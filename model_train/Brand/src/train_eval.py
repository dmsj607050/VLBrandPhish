from __future__ import annotations

import math
import os
import pickle
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel

from .datasets import LogoDataset, build_flat_splits, build_logo2k_splits
from .model import (
    ProtoCache,
    SigLIPDualFusionModel,
    SigLIPProtoModel,
    SigLIPTriSigModel,
    build_text_prototypes,
    freeze_all_text,
    freeze_all_vision,
    get_layer_id_for_siglip,
    get_num_layers,
    get_trainable_params_info,
    unfreeze_all_text,
    unfreeze_all_vision,
    unfreeze_text_last_n_layers,
    unfreeze_vision_last_n_layers,
)
from .prompts import build_label_to_prompts, summarize_label_prompts
from .transforms import build_transforms
from .utils import ensure_dir


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _state_dict_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}


def _safe_torch_load(path: Path) -> Any:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            raise EOFError("Checkpoint file missing or empty.")
        return torch.load(path, map_location="cpu", weights_only=False)
    except (FileNotFoundError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        size = path.stat().st_size if path.exists() else None
        print(f"[resume] Warning: failed to load checkpoint: {path} (size={size}) -> {type(exc).__name__}: {exc}")
        return None


def _safe_load_state_dict(
    module,
    state_dict: Any,
    *,
    module_name: str,
    strict: bool = False,
) -> bool:
    if not isinstance(state_dict, dict):
        return False
    try:
        if isinstance(module, nn.Module):
            missing, unexpected = module.load_state_dict(state_dict, strict=strict)
            print(f"[resume] Loaded {module_name}: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            module.load_state_dict(state_dict)
            print(f"[resume] Loaded {module_name}")
        return True
    except Exception as exc:
        print(f"[resume] Warning: failed to load {module_name} state -> {type(exc).__name__}: {exc}")
        return False


def _get_autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()

    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "autocast"):
        try:
            return amp_namespace.autocast(device_type=device.type, enabled=enabled)
        except TypeError:
            try:
                return amp_namespace.autocast(device.type, enabled=enabled)
            except TypeError:
                pass

    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)

    return nullcontext()


def _create_grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        cuda_amp = getattr(torch.cuda, "amp", None)
        if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
            return cuda_amp.GradScaler(enabled=False)
        amp_namespace = getattr(torch, "amp", None)
        if amp_namespace is not None and hasattr(amp_namespace, "GradScaler"):
            try:
                return amp_namespace.GradScaler(device.type, enabled=False)
            except TypeError:
                return amp_namespace.GradScaler(enabled=False)
        return None

    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "GradScaler"):
        try:
            return amp_namespace.GradScaler(device.type, enabled=enabled)
        except TypeError:
            try:
                return amp_namespace.GradScaler(enabled=enabled)
            except TypeError:
                pass

    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    raise RuntimeError("AMP is enabled, but no compatible GradScaler was found in this PyTorch build.")


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch
        if len(batch) == 2:
            images, labels = batch
            return images, labels, [""] * labels.size(0)
    raise ValueError("Unexpected batch format.")


def _is_loss_enabled(cfg: Dict[str, Any], key: str, default: bool = True) -> bool:
    loss_cfg = cfg.get("losses") or {}
    if not isinstance(loss_cfg, dict):
        return default
    value = loss_cfg.get(key)
    return default if value is None else bool(value)


def _compute_adaptive_margin_matrix(
    text_prototypes: torch.Tensor,
    *,
    min_margin: float,
    max_margin: float,
) -> torch.Tensor:
    prototypes = F.normalize(text_prototypes, dim=-1)
    similarity = prototypes @ prototypes.t()
    margins = min_margin + (max_margin - min_margin) * similarity.clamp(0.0, 1.0)
    margins.fill_diagonal_(0.0)
    return margins


def compute_adaptive_margin_sigmoid_loss(
    query_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    min_margin: float = 0.0,
    max_margin: float = 0.3,
) -> torch.Tensor:
    logits = torch.matmul(query_features, class_prototypes.t()) * logit_scale + logit_bias
    margins = _compute_adaptive_margin_matrix(
        class_prototypes,
        min_margin=float(min_margin),
        max_margin=float(max_margin),
    ).to(logits.device, dtype=logits.dtype)
    selected_margins = margins[labels.long()]
    adjusted_logits = logits - selected_margins
    targets = -torch.ones_like(adjusted_logits)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    return (-F.logsigmoid(targets * adjusted_logits).sum(dim=-1)).mean()


def compute_multibranch_consistency_loss(
    logits_text: torch.Tensor,
    logits_image: torch.Tensor,
) -> torch.Tensor:
    probs_text = torch.sigmoid(logits_text)
    probs_image = torch.sigmoid(logits_image)
    return F.mse_loss(probs_text, probs_image)


def compute_router_pseudo_supervision_loss(
    route_weights: torch.Tensor,
    logits_text: torch.Tensor,
    logits_image: torch.Tensor,
) -> torch.Tensor:
    text_conf = torch.sigmoid(logits_text.detach().amax(dim=-1))
    image_conf = torch.sigmoid(logits_image.detach().amax(dim=-1))
    evidence = torch.stack([text_conf, image_conf], dim=-1).to(route_weights.dtype)
    prior = torch.tensor([0.5, 0.5], device=route_weights.device, dtype=route_weights.dtype)
    target = prior.unsqueeze(0) * (0.5 + evidence)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="batchmean")


def compute_router_oracle_supervision_loss(
    route_weights: torch.Tensor,
    logits_text: torch.Tensor,
    logits_image: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    with torch.no_grad():
        loss_text = F.cross_entropy(logits_text, labels, reduction="none")
        loss_image = F.cross_entropy(logits_image, labels, reduction="none")
        branch_scores = torch.stack([-loss_text, -loss_image], dim=-1) / max(float(temperature), 1e-6)
        target = F.softmax(branch_scores, dim=-1)
    return F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="batchmean")


def compute_uncertainty_alignment_loss(
    route_weights: torch.Tensor,
    uncertainty_stats: Dict[str, Any],
) -> torch.Tensor:
    text_conf = uncertainty_stats["text"]["confidence"].detach()
    image_conf = uncertainty_stats["image"]["confidence"].detach()
    agreement = uncertainty_stats["agreement"].detach()
    target = torch.stack(
        [
            text_conf * (0.75 + 0.25 * agreement),
            image_conf * (0.75 + 0.25 * agreement),
        ],
        dim=-1,
    ).to(route_weights.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="batchmean")


class MetricsTracker:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_samples = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = logits.argmax(dim=-1)
        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            pred = int(pred)
            self.class_total[label] += 1
            if pred == label:
                self.class_correct[label] += 1
                self.total_correct_top1 += 1

        maxk = min(5, logits.size(1))
        _, topk = logits.topk(maxk, dim=1, largest=True, sorted=True)
        for idx, label in enumerate(labels.cpu().numpy()):
            if int(label) in topk[idx].cpu().numpy():
                self.total_correct_top5 += 1

        self.total_samples += labels.size(0)

    def get_accuracy(self) -> Tuple[float, float]:
        if self.total_samples == 0:
            return 0.0, 0.0
        return (
            self.total_correct_top1 / self.total_samples,
            self.total_correct_top5 / self.total_samples,
        )

    def get_mean_per_class_accuracy(self) -> float:
        values = []
        for class_id in range(self.num_classes):
            if self.class_total[class_id] > 0:
                values.append(self.class_correct[class_id] / self.class_total[class_id])
        return float(np.mean(values)) if values else 0.0

    def get_worst_classes(self, k: int = 10) -> List[Tuple[int, float, int]]:
        items = []
        for class_id in range(self.num_classes):
            if self.class_total[class_id] > 0:
                acc = self.class_correct[class_id] / self.class_total[class_id]
                items.append((class_id, acc, self.class_total[class_id]))
        items.sort(key=lambda x: x[1])
        return items[:k]


def _plot_loss_curves(
    *,
    out_path: Path,
    stage: str,
    objective: str,
    history: Dict[str, List[float]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[{stage}] Warning: matplotlib not available -> {type(exc).__name__}: {exc}")
        return

    epochs = list(range(1, len(history.get("train_losses", [])) + 1))
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].plot(epochs, history.get("train_losses", []), label="train")
    axes[0].plot(epochs, history.get("val_losses", []), label="val")
    axes[0].set_title(f"Loss | {stage} | {objective}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_acc_top1", []), label="train@1")
    axes[1].plot(epochs, history.get("val_acc_top1", []), label="val@1")
    axes[1].plot(epochs, history.get("train_acc_top5", []), label="train@5", linestyle="--", alpha=0.7)
    axes[1].plot(epochs, history.get("val_acc_top5", []), label="val@5", linestyle="--", alpha=0.7)
    axes[1].set_title(f"Accuracy | {stage}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("acc")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_splits(cfg: Dict[str, Any]):
    dataset_type = cfg["dataset_type"]
    common = {
        "val_ratio": float(cfg.get("val_ratio", 0.1)),
        "seed": int(cfg.get("seed", 42)),
        "debug_max_train_per_class": int(cfg.get("debug_max_train_per_class", 0)),
        "debug_max_val_per_class": int(cfg.get("debug_max_val_per_class", 0)),
    }
    if dataset_type == "logo2k":
        return build_logo2k_splits(cfg["data_root"], **common)
    if dataset_type == "flat":
        return build_flat_splits(
            cfg["data_root"],
            numeric_suffix_only=bool(cfg.get("flat_numeric_suffix_only", False)),
            excluded_filenames=cfg.get("flat_excluded_filenames") or [],
            ref_prefix_max=int(cfg.get("ref_images_per_brand_max", 0)),
            ref_exclusion_filenames=cfg.get("ref_exclusion_filenames") or [],
            **common,
        )
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


class ClassAwareBatchSampler(BatchSampler):
    def __init__(
        self,
        labels: List[int],
        *,
        batch_size: int,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int,
    ):
        self.labels = [int(x) for x in labels]
        self.batch_size = int(batch_size)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        if self.classes_per_batch * self.samples_per_class != self.batch_size:
            raise ValueError("batch_size must equal classes_per_batch * samples_per_class")
        self.seed = int(seed)
        self.epoch = 0
        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[int(label)].append(idx)
        self.classes = list(self.class_to_indices.keys())
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 10007)
        for _ in range(self.num_batches):
            if len(self.classes) >= self.classes_per_batch:
                chosen_classes = rng.sample(self.classes, self.classes_per_batch)
            else:
                chosen_classes = [rng.choice(self.classes) for _ in range(self.classes_per_batch)]

            batch: List[int] = []
            for class_id in chosen_classes:
                indices = self.class_to_indices[class_id]
                if len(indices) >= self.samples_per_class:
                    batch.extend(rng.sample(indices, self.samples_per_class))
                else:
                    batch.extend(rng.choices(indices, k=self.samples_per_class))
            yield batch


class WarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_idx = 0
        self._set_lr_scale(0.0 if self.warmup_steps > 0 else 1.0)

    def _lr_scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _set_lr_scale(self, scale: float) -> None:
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def step(self) -> None:
        self._set_lr_scale(self._lr_scale(self.step_idx))
        self.step_idx += 1


def create_optimizer_param_groups(
    model: nn.Module,
    cfg: Dict[str, Any],
    *,
    num_vision_layers: int,
    num_text_layers: int,
) -> List[Dict[str, Any]]:
    base_lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    layer_decay = float(cfg.get("layer_decay", 1.0))
    lr_scales = {
        "vision": float(cfg.get("lr_scale_vision", 1.0)),
        "text": float(cfg.get("lr_scale_text", 0.1)),
        "gate": float(cfg.get("lr_scale_gate", 1.0)),
        "router": float(cfg.get("lr_scale_router", cfg.get("lr_scale_gate", 1.0))),
        "adapter": float(cfg.get("lr_scale_adapter", 1.0)),
        "logit": float(cfg.get("lr_scale_logit", 0.1)),
        "other": 1.0,
    }
    no_decay_keywords = ("bias", "layer_norm", "norm", "embedding")
    grouped: Dict[str, Dict[str, Any]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        wd = 0.0 if any(keyword in name.lower() for keyword in no_decay_keywords) else weight_decay
        group_name = "other"
        lr_scale = lr_scales["other"]

        if "vision_model" in name:
            group_name = "vision"
            lr_scale = lr_scales["vision"]
            if layer_decay < 1.0:
                layer_id = get_layer_id_for_siglip(name, num_vision_layers, "vision")
                lr_scale *= layer_decay ** (num_vision_layers - layer_id)
        elif "text_model" in name:
            group_name = "text"
            lr_scale = lr_scales["text"]
            if layer_decay < 1.0:
                layer_id = get_layer_id_for_siglip(name, num_text_layers, "text")
                lr_scale *= layer_decay ** (num_text_layers - layer_id)
        elif "gate" in name:
            group_name = "gate"
            lr_scale = lr_scales["gate"]
        elif "router" in name:
            group_name = "router"
            lr_scale = lr_scales["router"]
        elif "adapter" in name:
            group_name = "adapter"
            lr_scale = lr_scales["adapter"]
        elif "logit_scale" in name or "logit_bias" in name:
            group_name = "logit"
            lr_scale = lr_scales["logit"]
            wd = 0.0

        key = f"{group_name}|{lr_scale:.8f}|{wd:.6f}"
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": base_lr * lr_scale,
                "weight_decay": wd,
                "name": group_name,
            }
        grouped[key]["params"].append(param)

    groups = list(grouped.values())
    print(f"[optimizer] Created {len(groups)} parameter groups")
    for idx, group in enumerate(groups):
        n_params = sum(p.numel() for p in group["params"])
        print(f"  - group {idx} [{group['name']}]: {n_params:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    return groups


def compute_siglip_pairwise_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_labels: torch.Tensor,
    text_labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    logits = torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias
    positives = image_labels.unsqueeze(1) == text_labels.unsqueeze(0)
    targets = torch.where(positives, 1.0, -1.0).to(logits.dtype)
    return (-F.logsigmoid(targets * logits).sum(dim=-1)).mean()


def compute_sigmoid_loss_class_proto(
    query_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    logits = torch.matmul(query_features, class_prototypes.t()) * logit_scale + logit_bias
    targets = -torch.ones_like(logits)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    return (-F.logsigmoid(targets * logits).sum(dim=-1)).mean()


def compute_multi_positive_sigmoid_loss(
    query_features: torch.Tensor,
    ref_bank: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    logits = torch.einsum("bd,ckd->bck", query_features, ref_bank) * logit_scale + logit_bias
    targets = -torch.ones_like(logits)
    batch_indices = torch.arange(labels.size(0), device=labels.device)
    targets[batch_indices, labels.long(), :] = 1.0
    return (-F.logsigmoid(targets * logits).sum(dim=(1, 2))).mean()


@torch.no_grad()
def _index_text_prototypes(proto_cache: ProtoCache, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    if proto_cache.label_to_index is None:
        return proto_cache.prototypes.to(device)[labels.long()]
    indices = torch.tensor(
        [proto_cache.label_to_index[int(label.item())] for label in labels],
        device=device,
        dtype=torch.long,
    )
    return proto_cache.prototypes.to(device)[indices]


def configure_freeze_strategy(model: nn.Module, cfg: Dict[str, Any], stage: str) -> None:
    freeze_vision = str(cfg.get("freeze_vision", "all")).lower()
    if freeze_vision == "all":
        freeze_all_vision(model)
        print(f"[{stage}] Vision: fully frozen")
    elif freeze_vision == "none":
        unfreeze_all_vision(model)
        print(f"[{stage}] Vision: fully trainable")
    elif freeze_vision == "except_last_n":
        freeze_all_vision(model)
        unfreeze_vision_last_n_layers(model, int(cfg.get("unfreeze_vision_last_n", 1)))
        print(f"[{stage}] Vision: unfreeze last {int(cfg.get('unfreeze_vision_last_n', 1))} layers")
    else:
        raise ValueError(f"Unknown freeze_vision: {freeze_vision}")

    freeze_text = str(cfg.get("freeze_text", "all")).lower()
    if freeze_text == "all":
        freeze_all_text(model)
        print(f"[{stage}] Text: fully frozen")
    elif freeze_text == "none":
        unfreeze_all_text(model)
        print(f"[{stage}] Text: fully trainable")
    elif freeze_text == "except_last_n":
        freeze_all_text(model)
        unfreeze_text_last_n_layers(model, int(cfg.get("unfreeze_text_last_n", 1)))
        print(f"[{stage}] Text: unfreeze last {int(cfg.get('unfreeze_text_last_n', 1))} layers")
    else:
        raise ValueError(f"Unknown freeze_text: {freeze_text}")

    train_logit = bool(cfg.get("train_logit", True))
    model.siglip.logit_scale.requires_grad = train_logit
    model.siglip.logit_bias.requires_grad = train_logit
    print(f"[{stage}] Logit params: {'trainable' if train_logit else 'frozen'}")

    if hasattr(model, "gate"):
        train_gate = bool(cfg.get("train_gate", True))
        for param in model.gate.parameters():
            param.requires_grad = train_gate
        print(f"[{stage}] Gate: {'trainable' if train_gate else 'frozen'}")

    if hasattr(model, "router"):
        train_router = bool(cfg.get("train_router", True))
        for param in model.router.parameters():
            param.requires_grad = train_router
        print(f"[{stage}] Router: {'trainable' if train_router else 'frozen'}")

    if hasattr(model, "adapter") and model.adapter is not None:
        train_adapter = bool(cfg.get("train_adapter", True))
        for param in model.adapter.parameters():
            param.requires_grad = train_adapter
        print(f"[{stage}] Adapter: {'trainable' if train_adapter else 'frozen'}")


def _is_text_encoder_trainable(cfg: Dict[str, Any]) -> bool:
    return str(cfg.get("freeze_text", "all")).lower() != "all"


@torch.no_grad()
def build_image_prototypes(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    *,
    num_classes: int,
    device: torch.device,
    transform,
    max_images_per_class: int = 8,
    use_query_features: bool = True,
) -> torch.Tensor:
    class_to_paths: Dict[int, List[str]] = defaultdict(list)
    for sample in samples:
        class_to_paths[int(sample["label"])].append(sample["image_path"])

    banks: List[torch.Tensor] = []
    max_images = int(max_images_per_class)
    was_training = model.training
    model.eval()

    for class_id in tqdm(range(num_classes), desc="Build image prototypes", leave=False):
        paths = class_to_paths.get(class_id, [])
        if not paths:
            raise RuntimeError(f"No samples found for class={class_id}")
        paths = paths[: min(len(paths), max_images)]

        features: List[torch.Tensor] = []
        for path in paths:
            from PIL import Image

            image = Image.open(path)
            if image.mode == "P" and "transparency" in image.info:
                image = image.convert("RGBA")
            image = image.convert("RGB")
            pixel_values = transform(image).unsqueeze(0).to(device)
            if use_query_features and hasattr(model, "encode_query_features"):
                feature = model.encode_query_features(pixel_values).squeeze(0)
            else:
                feature = model.encode_image_features(pixel_values).squeeze(0)
            features.append(F.normalize(feature, dim=-1))

        bank = torch.stack(features, dim=0)
        if bank.size(0) < max_images:
            bank = torch.cat([bank, bank[-1:].repeat(max_images - bank.size(0), 1)], dim=0)
        banks.append(bank)

    if was_training:
        model.train()
    return F.normalize(torch.stack(banks, dim=0), dim=-1)


def _maybe_refresh_text_prototypes(
    model: nn.Module,
    tokenizer,
    label_to_prompts: Dict[int, List[str]],
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    current_proto_cache: ProtoCache,
    stage: str,
) -> ProtoCache:
    if not _is_text_encoder_trainable(cfg):
        return current_proto_cache
    refresh_interval = int(cfg.get("proto_refresh_interval", 1))
    if refresh_interval <= 0 or epoch % refresh_interval != 0:
        return current_proto_cache
    print(f"[{stage}] Refreshing text prototypes at epoch {epoch}")
    new_cache = build_text_prototypes(
        model.siglip,
        tokenizer,
        label_to_prompts,
        device,
        prompt_ensemble=str(cfg.get("prompt_ensemble", "mean")),
    )
    model.set_text_prototypes(new_cache.prototypes.detach())
    return new_cache


def _maybe_refresh_image_prototypes(
    model: SigLIPDualFusionModel,
    train_samples: List[Dict[str, Any]],
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    val_transform,
    num_classes: int,
    stage: str,
) -> Optional[torch.Tensor]:
    interval = int(cfg.get("image_proto_rebuild_interval", 0))
    if interval <= 0 or epoch % interval != 0:
        return None
    print(f"[{stage}] Rebuilding image prototypes at epoch {epoch}")
    new_prototypes = build_image_prototypes(
        model,
        train_samples,
        num_classes=num_classes,
        device=device,
        transform=val_transform,
        max_images_per_class=int(cfg.get("image_proto_max_images_per_class", 8)),
        use_query_features=bool(cfg.get("use_adapter_for_prototypes", True)),
    )
    model.set_image_prototypes(new_prototypes)
    return new_prototypes


def _load_training_states(
    *,
    resume_ckpt: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineLRScheduler,
    scaler,
    stage: str,
) -> Tuple[int, float, int]:
    if not isinstance(resume_ckpt, dict):
        return 1, -1.0, 0

    ckpt_stage = str(resume_ckpt.get("stage", "")).lower()
    if ckpt_stage and ckpt_stage != str(stage).lower():
        print(f"[resume] Skip optimizer/scheduler restore: checkpoint stage={ckpt_stage}, current stage={stage}")
        return 1, -1.0, 0

    start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
    best_val_acc = float(resume_ckpt.get("best_val_acc", resume_ckpt.get("val_acc", -1.0)))
    bad_epochs = int(resume_ckpt.get("bad_epochs", 0))

    _safe_load_state_dict(
        optimizer,
        resume_ckpt.get("optimizer_state"),
        module_name=f"{stage} optimizer",
        strict=False,
    )
    if isinstance(resume_ckpt.get("scheduler_state"), dict):
        scheduler_state = resume_ckpt["scheduler_state"]
        scheduler.step_idx = int(scheduler_state.get("step_idx", scheduler.step_idx))
        print(f"[resume] Loaded {stage} scheduler step={scheduler.step_idx}")
    if isinstance(resume_ckpt.get("scaler_state"), dict):
        try:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
            print(f"[resume] Loaded {stage} AMP scaler state")
        except Exception as exc:
            print(f"[resume] Warning: failed to load scaler state -> {type(exc).__name__}: {exc}")

    return max(1, start_epoch), best_val_acc, max(0, bad_epochs)


def _evaluate_stage1_pairwise_loss(
    model: SigLIPProtoModel,
    loader,
    proto_cache: ProtoCache,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for step, batch in enumerate(loader, start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _get_autocast_context(device, amp):
            image_features = model.encode_image_features(images)
            text_features = F.normalize(_index_text_prototypes(proto_cache, labels, device=device).float(), dim=-1)
            loss = compute_siglip_pairwise_loss(
                image_features,
                text_features,
                labels,
                labels,
                logit_scale=model.get_logit_scale(),
                logit_bias=model.get_logit_bias(),
            )
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        if max_steps > 0 and step >= max_steps:
            break
    return total_loss / max(1, total_samples)


def _encode_trainable_text_features(
    model: SigLIPProtoModel,
    tokenizer,
    labels: torch.Tensor,
    label_to_prompts: Dict[int, List[str]],
    device: torch.device,
    *,
    prompt_ensemble: str = "mean",
) -> torch.Tensor:
    unique_labels, inverse = torch.unique(labels.detach(), sorted=True, return_inverse=True)
    label_ids = [int(x) for x in unique_labels.detach().cpu().tolist()]
    prompts_by_label = [label_to_prompts[label] for label in label_ids]
    flat_prompts = [prompt for prompts in prompts_by_label for prompt in prompts]
    tokens = tokenizer(
        flat_prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    prompt_features = model.siglip.get_text_features(**tokens).float()

    class_prototypes = []
    offset = 0
    for prompts in prompts_by_label:
        group = prompt_features[offset : offset + len(prompts)]
        offset += len(prompts)
        if str(prompt_ensemble).lower() == "max":
            group = group[group.norm(dim=-1).argmax()].unsqueeze(0)
        prototype = F.normalize(group.mean(dim=0), dim=-1)
        class_prototypes.append(prototype)
    prototypes = torch.stack(class_prototypes, dim=0)
    return F.normalize(prototypes[inverse], dim=-1)


def train_one_epoch_stage1(
    model: SigLIPProtoModel,
    loader,
    optimizer,
    scheduler: WarmupCosineLRScheduler,
    device: torch.device,
    epoch: int,
    *,
    amp: bool,
    proto_cache: ProtoCache,
    label_to_prompts: Dict[int, List[str]],
    tokenizer,
    train_text_encoder: bool,
    prompt_ensemble: str,
    grad_clip_norm: float,
    log_interval: int,
    scaler,
    num_classes: int,
    max_steps: int = 0,
    grad_accum_steps: int = 1,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    grad_accum_steps = max(1, int(grad_accum_steps))

    pbar = tqdm(loader, desc=f"Train(S1) E{epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(pbar, start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _get_autocast_context(device, amp):
            image_features = model.encode_image_features(images)
            if train_text_encoder:
                text_features = _encode_trainable_text_features(
                    model,
                    tokenizer,
                    labels,
                    label_to_prompts,
                    device,
                    prompt_ensemble=prompt_ensemble,
                )
            else:
                text_features = F.normalize(_index_text_prototypes(proto_cache, labels, device=device).float(), dim=-1)
            loss = compute_siglip_pairwise_loss(
                image_features,
                text_features,
                labels,
                labels,
                logit_scale=model.get_logit_scale(),
                logit_bias=model.get_logit_bias(),
            )
            logits = model.forward_logits_from_features(image_features)

        loss_to_backward = loss / grad_accum_steps
        scaler.scale(loss_to_backward).backward()
        if train_text_encoder and step == 1:
            grad_sq = 0.0
            for name, parameter in model.siglip.named_parameters():
                if parameter.requires_grad and (name.startswith("text_model.") or name.startswith("text_projection.")) and parameter.grad is not None:
                    grad_sq += float(parameter.grad.detach().float().pow(2).sum().item())
            text_grad_norm = grad_sq ** 0.5
            print(f"[stage1] Text encoder grad norm after first batch: {text_grad_norm:.6e}")
            if not text_grad_norm > 0.0:
                raise RuntimeError("Text encoder is configured trainable but received no gradient.")
        should_step = (step % grad_accum_steps == 0) or (max_steps > 0 and step >= max_steps) or (step == len(loader))
        if should_step:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        metrics.update(logits, labels)

        if step % int(log_interval) == 0:
            acc1, _ = metrics.get_accuracy()
            pbar.set_postfix({"loss": f"{total_loss / max(1, metrics.total_samples):.4f}", "acc@1": f"{acc1:.4f}"})

        if max_steps > 0 and step >= max_steps:
            break

    acc1, acc5 = metrics.get_accuracy()
    return total_loss / max(1, metrics.total_samples), acc1, acc5


def train_one_epoch_stage2(
    model: SigLIPDualFusionModel,
    loader,
    optimizer,
    scheduler: WarmupCosineLRScheduler,
    device: torch.device,
    epoch: int,
    *,
    amp: bool,
    lambda_cls: float,
    lambda_it: float,
    lambda_ii: float,
    label_smoothing: float,
    grad_clip_norm: float,
    log_interval: int,
    scaler,
    num_classes: int,
    max_steps: int = 0,
    grad_accum_steps: int = 1,
) -> Tuple[float, float, float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    loss_components = defaultdict(float)
    gate_sum = 0.0
    grad_accum_steps = max(1, int(grad_accum_steps))

    pbar = tqdm(loader, desc=f"Train(S2) E{epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _get_autocast_context(device, amp):
            logits_fused, _, _, gate_weights = model(images, return_gate_weights=True)
            query_features = model.encode_query_features(images)
            scale = model.get_logit_scale()
            bias = model.get_logit_bias()

            loss_cls = F.cross_entropy(logits_fused, labels, label_smoothing=float(label_smoothing))
            loss_it = compute_sigmoid_loss_class_proto(
                query_features,
                F.normalize(model.text_prototypes, dim=-1),
                labels,
                logit_scale=scale,
                logit_bias=bias,
            )
            if model.image_prototypes.ndim == 3:
                loss_ii = compute_multi_positive_sigmoid_loss(
                    query_features,
                    F.normalize(model.image_prototypes, dim=-1),
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                )
            else:
                loss_ii = compute_sigmoid_loss_class_proto(
                    query_features,
                    F.normalize(model.image_prototypes, dim=-1),
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                )
            loss = float(lambda_cls) * loss_cls + float(lambda_it) * loss_it + float(lambda_ii) * loss_ii

        loss_to_backward = loss / grad_accum_steps
        scaler.scale(loss_to_backward).backward()
        should_step = (step % grad_accum_steps == 0) or (max_steps > 0 and step >= max_steps) or (step == len(loader))
        if should_step:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        loss_components["cls"] += loss_cls.item() * labels.size(0)
        loss_components["it"] += loss_it.item() * labels.size(0)
        loss_components["ii"] += loss_ii.item() * labels.size(0)
        metrics.update(logits_fused, labels)
        gate_sum += gate_weights.mean().item() * labels.size(0)

        if step % int(log_interval) == 0:
            acc1, _ = metrics.get_accuracy()
            pbar.set_postfix({"loss": f"{total_loss / max(1, metrics.total_samples):.4f}", "acc@1": f"{acc1:.4f}"})

        if max_steps > 0 and step >= max_steps:
            break

    samples = max(1, metrics.total_samples)
    acc1, acc5 = metrics.get_accuracy()
    return total_loss / samples, acc1, acc5, {
        "loss_cls": loss_components["cls"] / samples,
        "loss_it": loss_components["it"] / samples,
        "loss_ii": loss_components["ii"] / samples,
        "gate_mean": gate_sum / samples,
    }


@torch.no_grad()
def evaluate_stage2(
    model: SigLIPDualFusionModel,
    loader,
    device: torch.device,
    *,
    amp: bool,
    lambda_cls: float,
    lambda_it: float,
    lambda_ii: float,
    num_classes: int,
    max_steps: int = 0,
) -> Tuple[float, float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    loss_components = defaultdict(float)
    gate_sum = 0.0

    for step, (images, labels) in enumerate(tqdm(loader, desc="Val(S2)", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _get_autocast_context(device, amp):
            logits_fused, _, _, gate_weights = model(images, return_gate_weights=True)
            query_features = model.encode_query_features(images)
            scale = model.get_logit_scale()
            bias = model.get_logit_bias()
            loss_cls = F.cross_entropy(logits_fused, labels)
            loss_it = compute_sigmoid_loss_class_proto(
                query_features,
                F.normalize(model.text_prototypes, dim=-1),
                labels,
                logit_scale=scale,
                logit_bias=bias,
            )
            if model.image_prototypes.ndim == 3:
                loss_ii = compute_multi_positive_sigmoid_loss(
                    query_features,
                    F.normalize(model.image_prototypes, dim=-1),
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                )
            else:
                loss_ii = compute_sigmoid_loss_class_proto(
                    query_features,
                    F.normalize(model.image_prototypes, dim=-1),
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                )
            loss = float(lambda_cls) * loss_cls + float(lambda_it) * loss_it + float(lambda_ii) * loss_ii

        total_loss += loss.item() * labels.size(0)
        loss_components["cls"] += loss_cls.item() * labels.size(0)
        loss_components["it"] += loss_it.item() * labels.size(0)
        loss_components["ii"] += loss_ii.item() * labels.size(0)
        gate_sum += gate_weights.mean().item() * labels.size(0)
        metrics.update(logits_fused, labels)

        if max_steps > 0 and step >= max_steps:
            break

    samples = max(1, metrics.total_samples)
    acc1, acc5 = metrics.get_accuracy()
    return total_loss / samples, acc1, acc5, {
        "loss_cls": loss_components["cls"] / samples,
        "loss_it": loss_components["it"] / samples,
        "loss_ii": loss_components["ii"] / samples,
        "gate_mean": gate_sum / samples,
        "mean_per_class_acc": metrics.get_mean_per_class_accuracy(),
    }


def train_one_epoch_trisig(
    model: SigLIPTriSigModel,
    loader,
    optimizer,
    scheduler: WarmupCosineLRScheduler,
    device: torch.device,
    epoch: int,
    *,
    amp: bool,
    lambda_cls: float,
    lambda_it: float,
    lambda_ii: float,
    lambda_consistency: float,
    lambda_margin: float,
    lambda_router: float,
    lambda_uncertainty: float,
    label_smoothing: float,
    grad_clip_norm: float,
    log_interval: int,
    scaler,
    num_classes: int,
    max_steps: int = 0,
    grad_accum_steps: int = 1,
    enable_cls_loss: bool = True,
    enable_it_loss: bool = True,
    enable_ii_loss: bool = True,
    enable_consistency_loss: bool = True,
    enable_margin_loss: bool = False,
    enable_router_loss: bool = False,
    enable_uncertainty_loss: bool = False,
    router_supervision: str = "pseudo",
    router_oracle_temperature: float = 0.5,
    margin_min: float = 0.0,
    margin_max: float = 0.3,
) -> Tuple[float, float, float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    loss_components = defaultdict(float)
    router_sum = torch.zeros(2, dtype=torch.float64)
    uncertainty_sums = torch.zeros(2, dtype=torch.float64)
    grad_accum_steps = max(1, int(grad_accum_steps))

    pbar = tqdm(loader, desc=f"Train(TriSig) E{epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(pbar, start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _get_autocast_context(device, amp):
            query_features = model.encode_query_features(images)
            logits_fused, logits_text, logits_image, route_weights, uncertainty_stats = model(
                images,
                return_route_weights=True,
            )
            scale = model.get_logit_scale()
            bias = model.get_logit_bias()

            text_prototypes = F.normalize(model.text_prototypes, dim=-1)
            image_prototypes = F.normalize(model.image_prototypes, dim=-1)

            loss_cls = (
                F.cross_entropy(logits_fused, labels, label_smoothing=float(label_smoothing))
                if enable_cls_loss else logits_fused.new_tensor(0.0)
            )
            loss_it = (
                compute_sigmoid_loss_class_proto(
                    query_features,
                    text_prototypes,
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                ) if enable_it_loss else logits_fused.new_tensor(0.0)
            )
            if enable_ii_loss:
                if model.image_prototypes.ndim == 3:
                    loss_ii = compute_multi_positive_sigmoid_loss(
                        query_features,
                        image_prototypes,
                        labels,
                        logit_scale=scale,
                        logit_bias=bias,
                    )
                else:
                    loss_ii = compute_sigmoid_loss_class_proto(
                        query_features,
                        image_prototypes,
                        labels,
                        logit_scale=scale,
                        logit_bias=bias,
                    )
            else:
                loss_ii = logits_fused.new_tensor(0.0)

            consistency = (
                compute_multibranch_consistency_loss(logits_text, logits_image)
                if enable_consistency_loss else logits_fused.new_tensor(0.0)
            )

            margin_loss = (
                compute_adaptive_margin_sigmoid_loss(
                    query_features,
                    text_prototypes,
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                    min_margin=float(margin_min),
                    max_margin=float(margin_max),
                ) if enable_margin_loss else logits_fused.new_tensor(0.0)
            )
            if enable_router_loss:
                if str(router_supervision).lower() == "oracle_ce":
                    router_loss = compute_router_oracle_supervision_loss(
                        route_weights,
                        logits_text,
                        logits_image,
                        labels,
                        temperature=float(router_oracle_temperature),
                    )
                else:
                    router_loss = compute_router_pseudo_supervision_loss(
                        route_weights,
                        logits_text,
                        logits_image,
                    )
            else:
                router_loss = logits_fused.new_tensor(0.0)
            uncertainty_loss = (
                compute_uncertainty_alignment_loss(route_weights, uncertainty_stats)
                if enable_uncertainty_loss and uncertainty_stats is not None else logits_fused.new_tensor(0.0)
            )

            loss = (
                float(lambda_cls) * loss_cls
                + float(lambda_it) * loss_it
                + float(lambda_ii) * loss_ii
                + float(lambda_consistency) * consistency
                + float(lambda_margin) * margin_loss
                + float(lambda_router) * router_loss
                + float(lambda_uncertainty) * uncertainty_loss
            )

        loss_to_backward = loss / grad_accum_steps
        scaler.scale(loss_to_backward).backward()
        should_step = (step % grad_accum_steps == 0) or (max_steps > 0 and step >= max_steps) or (step == len(loader))
        if should_step:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        loss_components["cls"] += loss_cls.item() * labels.size(0)
        loss_components["it"] += loss_it.item() * labels.size(0)
        loss_components["ii"] += loss_ii.item() * labels.size(0)
        loss_components["consistency"] += consistency.item() * labels.size(0)
        loss_components["margin"] += margin_loss.item() * labels.size(0)
        loss_components["router"] += router_loss.item() * labels.size(0)
        loss_components["uncertainty"] += uncertainty_loss.item() * labels.size(0)
        metrics.update(logits_fused, labels)
        router_sum += route_weights.detach().sum(dim=0).cpu().to(torch.float64)
        if uncertainty_stats is not None:
            uncertainty_sums[0] += uncertainty_stats["text"]["uncertainty"].detach().sum().cpu().to(torch.float64)
            uncertainty_sums[1] += uncertainty_stats["image"]["uncertainty"].detach().sum().cpu().to(torch.float64)

        if step % int(log_interval) == 0:
            acc1, _ = metrics.get_accuracy()
            pbar.set_postfix({"loss": f"{total_loss / max(1, metrics.total_samples):.4f}", "acc@1": f"{acc1:.4f}"})

        if max_steps > 0 and step >= max_steps:
            break

    samples = max(1, metrics.total_samples)
    acc1, acc5 = metrics.get_accuracy()
    router_mean = (router_sum / samples).tolist()
    return total_loss / samples, acc1, acc5, {
        "loss_cls": loss_components["cls"] / samples,
        "loss_it": loss_components["it"] / samples,
        "loss_ii": loss_components["ii"] / samples,
        "loss_consistency": loss_components["consistency"] / samples,
        "loss_margin": loss_components["margin"] / samples,
        "loss_router": loss_components["router"] / samples,
        "loss_uncertainty": loss_components["uncertainty"] / samples,
        "route_text": float(router_mean[0]),
        "route_image": float(router_mean[1]),
        "unc_text": float((uncertainty_sums[0] / samples).item()),
        "unc_image": float((uncertainty_sums[1] / samples).item()),
    }


@torch.no_grad()
def evaluate_trisig(
    model: SigLIPTriSigModel,
    loader,
    device: torch.device,
    *,
    amp: bool,
    lambda_cls: float,
    lambda_it: float,
    lambda_ii: float,
    lambda_consistency: float,
    lambda_margin: float,
    lambda_router: float,
    lambda_uncertainty: float,
    num_classes: int,
    max_steps: int = 0,
    enable_cls_loss: bool = True,
    enable_it_loss: bool = True,
    enable_ii_loss: bool = True,
    enable_consistency_loss: bool = True,
    enable_margin_loss: bool = False,
    enable_router_loss: bool = False,
    enable_uncertainty_loss: bool = False,
    router_supervision: str = "pseudo",
    router_oracle_temperature: float = 0.5,
    margin_min: float = 0.0,
    margin_max: float = 0.3,
) -> Tuple[float, float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    loss_components = defaultdict(float)
    router_sum = torch.zeros(2, dtype=torch.float64)
    uncertainty_sums = torch.zeros(2, dtype=torch.float64)

    for step, batch in enumerate(tqdm(loader, desc="Val(TriSig)", leave=False), start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _get_autocast_context(device, amp):
            query_features = model.encode_query_features(images)
            logits_fused, logits_text, logits_image, route_weights, uncertainty_stats = model(
                images,
                return_route_weights=True,
            )
            scale = model.get_logit_scale()
            bias = model.get_logit_bias()
            text_prototypes = F.normalize(model.text_prototypes, dim=-1)
            image_prototypes = F.normalize(model.image_prototypes, dim=-1)
            loss_cls = F.cross_entropy(logits_fused, labels) if enable_cls_loss else logits_fused.new_tensor(0.0)
            loss_it = (
                compute_sigmoid_loss_class_proto(
                    query_features,
                    text_prototypes,
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                ) if enable_it_loss else logits_fused.new_tensor(0.0)
            )
            if enable_ii_loss:
                if model.image_prototypes.ndim == 3:
                    loss_ii = compute_multi_positive_sigmoid_loss(
                        query_features,
                        image_prototypes,
                        labels,
                        logit_scale=scale,
                        logit_bias=bias,
                    )
                else:
                    loss_ii = compute_sigmoid_loss_class_proto(
                        query_features,
                        image_prototypes,
                        labels,
                        logit_scale=scale,
                        logit_bias=bias,
                    )
            else:
                loss_ii = logits_fused.new_tensor(0.0)
            consistency = (
                compute_multibranch_consistency_loss(logits_text, logits_image)
                if enable_consistency_loss else logits_fused.new_tensor(0.0)
            )
            margin_loss = (
                compute_adaptive_margin_sigmoid_loss(
                    query_features,
                    text_prototypes,
                    labels,
                    logit_scale=scale,
                    logit_bias=bias,
                    min_margin=float(margin_min),
                    max_margin=float(margin_max),
                ) if enable_margin_loss else logits_fused.new_tensor(0.0)
            )
            if enable_router_loss:
                if str(router_supervision).lower() == "oracle_ce":
                    router_loss = compute_router_oracle_supervision_loss(
                        route_weights,
                        logits_text,
                        logits_image,
                        labels,
                        temperature=float(router_oracle_temperature),
                    )
                else:
                    router_loss = compute_router_pseudo_supervision_loss(
                        route_weights,
                        logits_text,
                        logits_image,
                    )
            else:
                router_loss = logits_fused.new_tensor(0.0)
            uncertainty_loss = (
                compute_uncertainty_alignment_loss(route_weights, uncertainty_stats)
                if enable_uncertainty_loss and uncertainty_stats is not None else logits_fused.new_tensor(0.0)
            )
            loss = (
                float(lambda_cls) * loss_cls
                + float(lambda_it) * loss_it
                + float(lambda_ii) * loss_ii
                + float(lambda_consistency) * consistency
                + float(lambda_margin) * margin_loss
                + float(lambda_router) * router_loss
                + float(lambda_uncertainty) * uncertainty_loss
            )

        total_loss += loss.item() * labels.size(0)
        loss_components["cls"] += loss_cls.item() * labels.size(0)
        loss_components["it"] += loss_it.item() * labels.size(0)
        loss_components["ii"] += loss_ii.item() * labels.size(0)
        loss_components["consistency"] += consistency.item() * labels.size(0)
        loss_components["margin"] += margin_loss.item() * labels.size(0)
        loss_components["router"] += router_loss.item() * labels.size(0)
        loss_components["uncertainty"] += uncertainty_loss.item() * labels.size(0)
        router_sum += route_weights.detach().sum(dim=0).cpu().to(torch.float64)
        metrics.update(logits_fused, labels)
        if uncertainty_stats is not None:
            uncertainty_sums[0] += uncertainty_stats["text"]["uncertainty"].detach().sum().cpu().to(torch.float64)
            uncertainty_sums[1] += uncertainty_stats["image"]["uncertainty"].detach().sum().cpu().to(torch.float64)

        if max_steps > 0 and step >= max_steps:
            break

    samples = max(1, metrics.total_samples)
    acc1, acc5 = metrics.get_accuracy()
    router_mean = (router_sum / samples).tolist()
    return total_loss / samples, acc1, acc5, {
        "loss_cls": loss_components["cls"] / samples,
        "loss_it": loss_components["it"] / samples,
        "loss_ii": loss_components["ii"] / samples,
        "loss_consistency": loss_components["consistency"] / samples,
        "loss_margin": loss_components["margin"] / samples,
        "loss_router": loss_components["router"] / samples,
        "loss_uncertainty": loss_components["uncertainty"] / samples,
        "route_text": float(router_mean[0]),
        "route_image": float(router_mean[1]),
        "unc_text": float((uncertainty_sums[0] / samples).item()),
        "unc_image": float((uncertainty_sums[1] / samples).item()),
        "mean_per_class_acc": metrics.get_mean_per_class_accuracy(),
    }


@torch.no_grad()
def evaluate_full_classification_trisig(
    model: SigLIPTriSigModel,
    loader,
    device: torch.device,
    *,
    amp: bool,
    num_classes: int,
    label_to_name: Optional[Dict[int, str]] = None,
    max_steps: int = 0,
) -> Tuple[float, float, Dict[str, Any]]:
    model.eval()
    metrics = MetricsTracker(num_classes)

    for step, batch in enumerate(tqdm(loader, desc="FullEval(TriSig)", leave=False), start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _get_autocast_context(device, amp):
            logits, _, _, _, _ = model(images, return_route_weights=True)
        metrics.update(logits, labels)
        if max_steps > 0 and step >= max_steps:
            break

    acc1, acc5 = metrics.get_accuracy()
    worst = metrics.get_worst_classes(k=10)
    if label_to_name is None:
        worst_named = [(class_id, acc, count, f"class_{class_id}") for class_id, acc, count in worst]
    else:
        worst_named = [(class_id, acc, count, label_to_name.get(class_id, f"class_{class_id}")) for class_id, acc, count in worst]
    return acc1, acc5, {
        "mean_per_class_acc": metrics.get_mean_per_class_accuracy(),
        "worst_classes": worst_named,
    }


@torch.no_grad()
def evaluate_full_classification(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    amp: bool,
    num_classes: int,
    label_to_name: Optional[Dict[int, str]] = None,
    max_steps: int = 0,
) -> Tuple[float, float, Dict[str, Any]]:
    model.eval()
    metrics = MetricsTracker(num_classes)

    for step, batch in enumerate(tqdm(loader, desc="FullEval", leave=False), start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _get_autocast_context(device, amp):
            if isinstance(model, SigLIPTriSigModel):
                raise RuntimeError("TriSig full evaluation requires the router-fusion evaluator.")
            if isinstance(model, SigLIPDualFusionModel):
                logits, _, _, _ = model(images, return_gate_weights=True)
            else:
                logits = model(images)
        metrics.update(logits, labels)
        if max_steps > 0 and step >= max_steps:
            break

    acc1, acc5 = metrics.get_accuracy()
    worst = metrics.get_worst_classes(k=10)
    if label_to_name is None:
        worst_named = [(class_id, acc, count, f"class_{class_id}") for class_id, acc, count in worst]
    else:
        worst_named = [(class_id, acc, count, label_to_name.get(class_id, f"class_{class_id}")) for class_id, acc, count in worst]
    return acc1, acc5, {
        "mean_per_class_acc": metrics.get_mean_per_class_accuracy(),
        "worst_classes": worst_named,
    }


def run_training(cfg: Dict[str, Any]) -> Path:
    _set_seed(int(cfg.get("seed", 42)))
    stage = str(cfg.get("stage", "stage1")).lower()
    objective = str(cfg.get("objective", "pairwise")).lower()

    output_dir = ensure_dir(cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"[{stage}] Device: {device}")
    print(f"[{stage}] Objective: {objective}")
    print(f"{'=' * 60}\n")

    train_samples, val_samples, default_prompts, label_to_name = _build_splits(cfg)
    label_to_prompts = build_label_to_prompts(
        label_to_name=label_to_name,
        cfg=cfg,
        project_root=Path.cwd(),
        default_prompts=default_prompts,
    )
    num_classes = len(label_to_name)
    print(f"[{stage}] Train: {len(train_samples)} | Val: {len(val_samples)} | Classes: {num_classes}")

    prompt_stats = summarize_label_prompts(label_to_prompts)
    print(
        f"[{stage}] Prompts/class: min={int(prompt_stats['min'])}, "
        f"mean={prompt_stats['mean']:.1f}, max={int(prompt_stats['max'])}"
    )

    resume = cfg.get("resume")
    resume_ckpt = _safe_torch_load(Path(resume)) if resume else None
    init_from = cfg.get("init_from")
    init_ckpt = None
    if init_from and not resume_ckpt:
        init_ckpt = _safe_torch_load(Path(init_from))
        if isinstance(init_ckpt, dict):
            print(f"[{stage}] Initialize weights from: {init_from}")
    weight_ckpt = resume_ckpt if isinstance(resume_ckpt, dict) else init_ckpt

    processor = AutoProcessor.from_pretrained(cfg["model_name"])
    tokenizer = processor.tokenizer
    siglip_model = SiglipModel.from_pretrained(cfg["model_name"]).to(device)
    siglip_model = siglip_model.float()

    if isinstance(weight_ckpt, dict) and isinstance(weight_ckpt.get("siglip_state"), dict):
        missing, unexpected = siglip_model.load_state_dict(weight_ckpt["siglip_state"], strict=False)
        print(f"[{stage}] Loaded SigLIP state: missing={len(missing)}, unexpected={len(unexpected)}")

    if objective == "dual_fusion":
        model: nn.Module = SigLIPDualFusionModel(
            siglip_model,
            use_adapter=bool(cfg.get("use_adapter", True)),
            adapter_hidden_dim=int(cfg.get("adapter_hidden_dim", 256)),
            adapter_dropout=float(cfg.get("adapter_dropout", 0.1)),
            use_class_aware_gate=bool(cfg.get("use_class_aware_gate", True)),
            gate_hidden_dim=int(cfg.get("gate_hidden_dim", 256)),
            gate_dropout=float(cfg.get("gate_dropout", 0.1)),
            image_proto_agg=str(cfg.get("image_proto_agg", "attention")),
            image_proto_agg_tau=float(cfg.get("image_proto_agg_tau", 0.07)),
        ).to(device)
    elif objective == "tri_sig":
        model = SigLIPTriSigModel(
            siglip_model,
            use_adapter=bool(cfg.get("use_adapter", True)),
            adapter_hidden_dim=int(cfg.get("adapter_hidden_dim", 256)),
            adapter_dropout=float(cfg.get("adapter_dropout", 0.1)),
            router_hidden_dim=int(cfg.get("router_hidden_dim", 256)),
            router_dropout=float(cfg.get("router_dropout", 0.1)),
            image_proto_agg=str(cfg.get("image_proto_agg", "attention")),
            image_proto_agg_tau=float(cfg.get("image_proto_agg_tau", 0.07)),
        ).to(device)
    else:
        model = SigLIPProtoModel(siglip_model).to(device)

    if objective in {"dual_fusion", "tri_sig"} and isinstance(weight_ckpt, dict):
        if objective == "dual_fusion" and "gate_state" in weight_ckpt:
            try:
                model.gate.load_state_dict(weight_ckpt["gate_state"], strict=True)
                print(f"[{stage}] Resumed gate state")
            except Exception as exc:
                print(f"[{stage}] Warning: failed to load gate state -> {exc}")
        if objective == "tri_sig" and "router_state" in weight_ckpt:
            try:
                model.router.load_state_dict(weight_ckpt["router_state"], strict=True)
                print(f"[{stage}] Resumed router state")
            except Exception as exc:
                print(f"[{stage}] Warning: failed to load router state -> {exc}")
        if "adapter_state" in weight_ckpt and model.adapter is not None:
            try:
                model.adapter.load_state_dict(weight_ckpt["adapter_state"], strict=True)
                print(f"[{stage}] Resumed adapter state")
            except Exception as exc:
                print(f"[{stage}] Warning: failed to load adapter state -> {exc}")

    resumed_text = weight_ckpt.get("text_prototypes") if isinstance(weight_ckpt, dict) else None
    if isinstance(resumed_text, torch.Tensor) and int(resumed_text.size(0)) == len(label_to_prompts):
        resumed_text_prototypes = weight_ckpt["text_prototypes"].to(device).float()
        proto_cache = ProtoCache(
            prototypes=F.normalize(resumed_text_prototypes, dim=-1),
            label_to_prompts=label_to_prompts,
            labels=sorted(label_to_prompts.keys()),
            label_to_index={label: idx for idx, label in enumerate(sorted(label_to_prompts.keys()))},
        )
        print(f"[{stage}] Resumed text prototypes from checkpoint")
    else:
        if isinstance(resumed_text, torch.Tensor):
            print(
                f"[{stage}] Skip resumed text prototypes: ckpt_classes={int(resumed_text.size(0))}, "
                f"current_classes={len(label_to_prompts)}"
            )
        proto_cache = build_text_prototypes(
            model.siglip,
            tokenizer,
            label_to_prompts,
            device,
            prompt_ensemble=str(cfg.get("prompt_ensemble", "mean")),
        )
    model.set_text_prototypes(proto_cache.prototypes.detach())

    configure_freeze_strategy(model, cfg, stage)
    info = get_trainable_params_info(model)
    print(
        f"[{stage}] Trainable: {int(info['total_trainable']):,} / {int(info['total_params']):,} "
        f"({info['trainable_ratio']:.2%})"
    )

    image_processor = processor.image_processor
    train_tf = build_transforms(
        image_size_cfg=image_processor.size,
        image_mean=image_processor.image_mean,
        image_std=image_processor.image_std,
        train=True,
    )
    val_tf = build_transforms(
        image_size_cfg=image_processor.size,
        image_mean=image_processor.image_mean,
        image_std=image_processor.image_std,
        train=False,
    )

    train_ds = LogoDataset(train_samples, train_tf)
    val_ds = LogoDataset(val_samples, val_tf)

    num_workers = int(cfg.get("workers", 4))
    if os.name == "nt":
        num_workers = 0
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": _worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    batch_size = int(cfg.get("batch_size", 32))
    if bool(cfg.get("class_aware_sampler", False)):
        labels_train = [int(sample["label"]) for sample in train_samples]
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ClassAwareBatchSampler(
                labels_train,
                batch_size=batch_size,
                classes_per_batch=int(cfg.get("classes_per_batch", batch_size)),
                samples_per_class=int(cfg.get("samples_per_class", 1)),
                seed=int(cfg.get("seed", 42)),
            ),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    image_prototypes = None
    if objective in {"dual_fusion", "tri_sig"}:
        resumed_image = weight_ckpt.get("image_prototypes") if isinstance(weight_ckpt, dict) else None
        if isinstance(resumed_image, torch.Tensor) and int(resumed_image.size(0)) == num_classes:
            image_prototypes = F.normalize(resumed_image.to(device).float(), dim=-1)
            print(f"[{stage}] Resumed image prototypes from checkpoint")
        else:
            if isinstance(resumed_image, torch.Tensor):
                print(
                    f"[{stage}] Skip resumed image prototypes: ckpt_classes={int(resumed_image.size(0))}, "
                    f"current_classes={num_classes}"
                )
            image_prototypes = build_image_prototypes(
                model,
                train_samples,
                num_classes=num_classes,
                device=device,
                transform=val_tf,
                max_images_per_class=int(cfg.get("image_proto_max_images_per_class", 8)),
                use_query_features=bool(cfg.get("use_adapter_for_prototypes", True)),
            )
        model.set_image_prototypes(image_prototypes)

    num_vision_layers = get_num_layers(model.siglip, "vision")
    num_text_layers = get_num_layers(model.siglip, "text")
    optimizer = torch.optim.AdamW(
        create_optimizer_param_groups(
            model,
            cfg,
            num_vision_layers=num_vision_layers,
            num_text_layers=num_text_layers,
        )
    )

    epochs = int(cfg.get("epochs", 1))
    steps_per_epoch = max(1, len(train_loader))
    debug_max_steps = int(cfg.get("debug_max_steps_per_epoch", 0))
    if debug_max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, debug_max_steps)
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=int(epochs * steps_per_epoch * float(cfg.get("warmup_ratio", 0.05))),
        min_lr_ratio=float(cfg.get("min_lr_ratio", 0.05)),
    )

    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = _create_grad_scaler(device, amp)
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    log_interval = int(cfg.get("log_interval", 50))
    full_eval_interval = int(cfg.get("full_eval_interval", 1))
    patience = int(cfg.get("early_stop_patience", 0))
    min_delta = float(cfg.get("early_stop_min_delta", 0.0))
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    best_val_acc = -1.0
    bad_epochs = 0
    best_path = output_dir / "best.pt"

    loss_weights = cfg.get("loss_weights", {}) or {}
    lambda_cls = float(loss_weights.get("lambda_cls", 1.0))
    lambda_it = float(loss_weights.get("lambda_it", 0.3))
    lambda_ii = float(loss_weights.get("lambda_ii", 0.3))
    lambda_consistency = float(loss_weights.get("lambda_consistency", 0.1))
    lambda_margin = float(loss_weights.get("lambda_margin", 0.0))
    lambda_router = float(loss_weights.get("lambda_router", 0.0))
    lambda_uncertainty = float(loss_weights.get("lambda_uncertainty", 0.0))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    enable_cls_loss = _is_loss_enabled(cfg, "cls", True)
    enable_it_loss = _is_loss_enabled(cfg, "it", True)
    enable_ii_loss = _is_loss_enabled(cfg, "ii", True)
    enable_consistency_loss = _is_loss_enabled(cfg, "consistency", True)
    enable_margin_loss = _is_loss_enabled(cfg, "adaptive_margin", False)
    enable_router_loss = _is_loss_enabled(cfg, "router", False)
    enable_uncertainty_loss = _is_loss_enabled(cfg, "uncertainty", False)
    router_supervision = str(cfg.get("router_supervision", "pseudo"))
    router_oracle_temperature = float(cfg.get("router_oracle_temperature", 0.5))
    margin_cfg = cfg.get("adaptive_margin", {}) or {}
    margin_min = float(margin_cfg.get("min_margin", 0.0))
    margin_max = float(margin_cfg.get("max_margin", 0.3))

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_acc_top1": [],
        "val_acc_top1": [],
        "train_acc_top5": [],
        "val_acc_top5": [],
    }

    start_epoch = 1
    if resume_ckpt is not None:
        start_epoch, best_val_acc, bad_epochs = _load_training_states(
            resume_ckpt=resume_ckpt,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            stage=stage,
        )
        if start_epoch > epochs:
            print(f"[{stage}] Resume checkpoint already reached epoch {start_epoch - 1}, no training needed.")
            return best_path if best_path.exists() else Path(resume)

    print(f"[{stage}] Starting training for {epochs} epochs (from epoch {start_epoch}, grad_accum={grad_accum_steps})")
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"[{stage}] Epoch {epoch}/{epochs}")
        print(f"{'=' * 60}")

        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        proto_cache = _maybe_refresh_text_prototypes(
            model,
            tokenizer,
            label_to_prompts,
            device,
            cfg,
            epoch,
            proto_cache,
            stage,
        )

        if objective in {"dual_fusion", "tri_sig"}:
            refreshed = _maybe_refresh_image_prototypes(
                model,
                train_samples,
                device,
                cfg,
                epoch,
                val_tf,
                num_classes,
                stage,
            )
            if refreshed is not None:
                image_prototypes = refreshed

        if objective == "dual_fusion":
            train_loss, train_acc1, train_acc5, train_stats = train_one_epoch_stage2(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                amp=amp,
                lambda_cls=lambda_cls,
                lambda_it=lambda_it,
                lambda_ii=lambda_ii,
                label_smoothing=label_smoothing,
                grad_clip_norm=grad_clip_norm,
                log_interval=log_interval,
                scaler=scaler,
                num_classes=num_classes,
                max_steps=debug_max_steps,
                grad_accum_steps=grad_accum_steps,
            )
            val_loss, val_acc1, val_acc5, val_stats = evaluate_stage2(
                model,
                val_loader,
                device,
                amp=amp,
                lambda_cls=lambda_cls,
                lambda_it=lambda_it,
                lambda_ii=lambda_ii,
                num_classes=num_classes,
                max_steps=debug_max_steps,
            )
            print(
                f"[{stage}] Train loss={train_loss:.4f}, acc@1={train_acc1:.4f}, acc@5={train_acc5:.4f}, "
                f"cls={train_stats['loss_cls']:.4f}, it={train_stats['loss_it']:.4f}, ii={train_stats['loss_ii']:.4f}"
            )
            print(
                f"[{stage}] Val   loss={val_loss:.4f}, acc@1={val_acc1:.4f}, acc@5={val_acc5:.4f}, "
                f"mean_per_class={val_stats['mean_per_class_acc']:.4f}"
            )
            primary_acc = val_acc1
        elif objective == "tri_sig":
            train_loss, train_acc1, train_acc5, train_stats = train_one_epoch_trisig(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                amp=amp,
                lambda_cls=lambda_cls,
                lambda_it=lambda_it,
                lambda_ii=lambda_ii,
                lambda_consistency=lambda_consistency,
                lambda_margin=lambda_margin,
                lambda_router=lambda_router,
                lambda_uncertainty=lambda_uncertainty,
                label_smoothing=label_smoothing,
                grad_clip_norm=grad_clip_norm,
                log_interval=log_interval,
                scaler=scaler,
                num_classes=num_classes,
                max_steps=debug_max_steps,
                grad_accum_steps=grad_accum_steps,
                enable_cls_loss=enable_cls_loss,
                enable_it_loss=enable_it_loss,
                enable_ii_loss=enable_ii_loss,
                enable_consistency_loss=enable_consistency_loss,
                enable_margin_loss=enable_margin_loss,
                enable_router_loss=enable_router_loss,
                enable_uncertainty_loss=enable_uncertainty_loss,
                router_supervision=router_supervision,
                router_oracle_temperature=router_oracle_temperature,
                margin_min=margin_min,
                margin_max=margin_max,
            )
            val_loss, val_acc1, val_acc5, val_stats = evaluate_trisig(
                model,
                val_loader,
                device,
                amp=amp,
                lambda_cls=lambda_cls,
                lambda_it=lambda_it,
                lambda_ii=lambda_ii,
                lambda_consistency=lambda_consistency,
                lambda_margin=lambda_margin,
                lambda_router=lambda_router,
                lambda_uncertainty=lambda_uncertainty,
                num_classes=num_classes,
                max_steps=debug_max_steps,
                enable_cls_loss=enable_cls_loss,
                enable_it_loss=enable_it_loss,
                enable_ii_loss=enable_ii_loss,
                enable_consistency_loss=enable_consistency_loss,
                enable_margin_loss=enable_margin_loss,
                enable_router_loss=enable_router_loss,
                enable_uncertainty_loss=enable_uncertainty_loss,
                router_supervision=router_supervision,
                router_oracle_temperature=router_oracle_temperature,
                margin_min=margin_min,
                margin_max=margin_max,
            )
            print(
                f"[{stage}] Train loss={train_loss:.4f}, acc@1={train_acc1:.4f}, acc@5={train_acc5:.4f}, "
                f"cls={train_stats['loss_cls']:.4f}, it={train_stats['loss_it']:.4f}, "
                f"ii={train_stats['loss_ii']:.4f}, margin={train_stats['loss_margin']:.4f}, "
                f"router={train_stats['loss_router']:.4f}, uncertainty={train_stats['loss_uncertainty']:.4f}"
            )
            print(
                f"[{stage}] Val   loss={val_loss:.4f}, acc@1={val_acc1:.4f}, acc@5={val_acc5:.4f}, "
                f"mean_per_class={val_stats['mean_per_class_acc']:.4f}, route_image={val_stats['route_image']:.4f}, "
                f"unc_text={val_stats['unc_text']:.4f}, unc_image={val_stats['unc_image']:.4f}"
            )
            primary_acc = val_acc1
        else:
            train_loss, train_acc1, train_acc5 = train_one_epoch_stage1(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                amp=amp,
                proto_cache=proto_cache,
                label_to_prompts=label_to_prompts,
                tokenizer=tokenizer,
                train_text_encoder=_is_text_encoder_trainable(cfg),
                prompt_ensemble=str(cfg.get("prompt_ensemble", "mean")),
                grad_clip_norm=grad_clip_norm,
                log_interval=log_interval,
                scaler=scaler,
                num_classes=num_classes,
                max_steps=debug_max_steps,
                grad_accum_steps=grad_accum_steps,
            )
            if _is_text_encoder_trainable(cfg):
                print(f"[{stage}] Refreshing text prototypes after epoch {epoch}")
                proto_cache = build_text_prototypes(
                    model.siglip,
                    tokenizer,
                    label_to_prompts,
                    device,
                    prompt_ensemble=str(cfg.get("prompt_ensemble", "mean")),
                )
                model.set_text_prototypes(proto_cache.prototypes.detach())
            val_loss = _evaluate_stage1_pairwise_loss(
                model,
                val_loader,
                proto_cache,
                device,
                amp=amp,
                max_steps=debug_max_steps,
            )
            print(f"[{stage}] Train loss={train_loss:.4f}, acc@1={train_acc1:.4f}, acc@5={train_acc5:.4f}")
            primary_acc = 0.0
            val_acc1, val_acc5 = 0.0, 0.0

        full_acc1, full_acc5, detailed_stats = 0.0, 0.0, {}
        if epoch % full_eval_interval == 0 or epoch == epochs:
            if objective == "tri_sig":
                full_acc1, full_acc5, detailed_stats = evaluate_full_classification_trisig(
                    model,
                    val_loader,
                    device,
                    amp=amp,
                    num_classes=num_classes,
                    label_to_name=label_to_name,
                    max_steps=debug_max_steps,
                )
            else:
                full_acc1, full_acc5, detailed_stats = evaluate_full_classification(
                    model,
                    val_loader,
                    device,
                    amp=amp,
                    num_classes=num_classes,
                    label_to_name=label_to_name,
                    max_steps=debug_max_steps,
                )
            print(
                f"[{stage}] Full eval: acc@1={full_acc1:.4f}, acc@5={full_acc5:.4f}, "
                f"mean_per_class={detailed_stats['mean_per_class_acc']:.4f}"
            )
            primary_acc = full_acc1
            if detailed_stats.get("worst_classes"):
                for class_id, acc, count, name in detailed_stats["worst_classes"][:3]:
                    print(f"[{stage}] Worst: {name} (id={class_id}) acc={acc:.4f}, n={count}")

        history["train_losses"].append(float(train_loss))
        history["val_losses"].append(float(val_loss))
        history["train_acc_top1"].append(float(train_acc1))
        history["train_acc_top5"].append(float(train_acc5))
        history["val_acc_top1"].append(float(full_acc1 if full_acc1 > 0 else val_acc1))
        history["val_acc_top5"].append(float(full_acc5 if full_acc5 > 0 else val_acc5))

        if bool(cfg.get("plot_loss_curves", True)):
            _plot_loss_curves(
                out_path=output_dir / "loss_curves.png",
                stage=stage,
                objective=objective,
                history=history,
            )

        ckpt = {
            "stage": stage,
            "objective": objective,
            "epoch": epoch,
            "cfg": cfg,
            "model_name": cfg.get("model_name"),
            "val_acc": float(primary_acc),
            "best_val_acc": float(best_val_acc),
            "bad_epochs": int(bad_epochs),
            "siglip_state": _state_dict_to_cpu(model.siglip.state_dict()),
            "text_prototypes": proto_cache.prototypes.detach().cpu(),
            "image_prototypes": image_prototypes.detach().cpu() if image_prototypes is not None else None,
            "label_to_prompts": label_to_prompts,
            "label_to_name": label_to_name,
            "history": history,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": {
                "step_idx": int(scheduler.step_idx),
                "total_steps": int(scheduler.total_steps),
                "warmup_steps": int(scheduler.warmup_steps),
                "min_lr_ratio": float(scheduler.min_lr_ratio),
            },
            "scaler_state": scaler.state_dict(),
        }
        if objective == "dual_fusion":
            ckpt["gate_state"] = _state_dict_to_cpu(model.gate.state_dict())
            if model.adapter is not None:
                ckpt["adapter_state"] = _state_dict_to_cpu(model.adapter.state_dict())
        if objective == "tri_sig":
            ckpt["router_state"] = _state_dict_to_cpu(model.router.state_dict())
            if model.adapter is not None:
                ckpt["adapter_state"] = _state_dict_to_cpu(model.adapter.state_dict())

        if bool(cfg.get("save_each_epoch", False)):
            torch.save(ckpt, output_dir / f"ckpt_epoch{epoch}.pt")

        if primary_acc > best_val_acc + min_delta:
            best_val_acc = float(primary_acc)
            ckpt["best_val_acc"] = float(best_val_acc)
            ckpt["bad_epochs"] = int(bad_epochs)
            torch.save(ckpt, best_path)
            print(f"[{stage}] New best acc@1={best_val_acc:.4f}")
            bad_epochs = 0
        else:
            bad_epochs += 1

        if patience > 0 and bad_epochs >= patience:
            print(f"[{stage}] Early stopping after {bad_epochs} bad epochs")
            break

    if objective in {"dual_fusion", "tri_sig"} and image_prototypes is not None:
        torch.save(image_prototypes.detach().cpu(), output_dir / "image_prototypes.pt")

    print(f"[{stage}] Training complete. Best model: {best_path}")
    return best_path
