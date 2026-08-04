from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel

from .common.datasets import LogoDataset, build_flat_splits, build_logo2k_splits
from .common.losses import (
    is_loss_enabled as _is_loss_enabled,
)
from .common.model import (
    ProtoCache,
    SigLIPProtoModel,
    SigLIPTriSigModel,
    build_text_prototypes,
    freeze_all_text,
    freeze_all_vision,
    get_num_layers,
    get_trainable_params_info,
    unfreeze_all_text,
    unfreeze_all_vision,
    unfreeze_text_last_n_layers,
    unfreeze_vision_last_n_layers,
)
from .common.optimization import WarmupCosineLRScheduler, create_optimizer_param_groups
from .common.prompts import build_label_to_prompts, summarize_label_prompts
from .common.runtime import MetricsTracker, _get_autocast_context, _index_text_prototypes, _unpack_batch
from .common.transforms import build_transforms
from .common.utils import ensure_dir


def _set_seed(seed: int, *, deterministic: bool = False) -> None:
    seed = int(seed)
    deterministic = bool(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except (AttributeError, RuntimeError):
        # Older PyTorch builds may not expose this control; the cuDNN setting above
        # remains in effect and the run metadata records the installed version.
        pass


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
            **common,
        )
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


class ClassAwareBatchSampler(BatchSampler):
    """Batch sampler that draws a fixed number of classes per batch.

    Each batch contains ``classes_per_batch`` distinct classes with
    ``samples_per_class`` samples each.  If a class has fewer samples than
    requested, samples are drawn with replacement.
    """
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


def configure_freeze_strategy(model: nn.Module, cfg: Dict[str, Any], stage: str) -> None:
    """Apply per-module freeze/unfreeze rules based on config flags.

    Supported values for ``freeze_vision`` / ``freeze_text``:
      ``"all"``      – freeze every parameter in that encoder
      ``"none"``     – keep all parameters trainable
      ``"except_last_n"`` – freeze the encoder, then unfreeze its last N layers

    The ``train_logit`` flag controls whether ``logit_scale`` and
    ``logit_bias`` are trainable.
    """
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
    """Build per-class image prototype banks from training samples.

    For each class, up to ``max_images_per_class`` images are encoded and
    stacked into a bank.  Classes with fewer images are padded by repeating
    the last available image.  Returns a tensor of shape
    ``(num_classes, max_images_per_class, embed_dim)``.
    """
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
    model: SigLIPTriSigModel,
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
        use_query_features=True,
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
    # Stage modules import common helpers from this module. Importing them only
    # after this module has loaded keeps that dependency one-way at runtime.
    from .stage1.trainer import evaluate_stage1, train_one_epoch_stage1
    from .stage2.trainer import (
        evaluate_full_classification_trisig,
        evaluate_trisig,
        train_one_epoch_trisig,
    )
    _set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    stage = str(cfg.get("stage", "stage1")).lower()
    objective = str(cfg.get("objective", "pairwise")).lower()

    output_dir = ensure_dir(cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"[{stage}] Device: {device}")
    print(f"[{stage}] Objective: {objective}")
    print(f"[{stage}] Deterministic backend: {bool(cfg.get('deterministic', False))}")
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

    if objective == "tri_sig":
        model = SigLIPTriSigModel(
            siglip_model,
            image_proto_agg=str(cfg.get("image_proto_agg", "attention")),
            image_proto_agg_tau=float(cfg.get("image_proto_agg_tau", 0.07)),
            image_weight=float(cfg.get("image_weight", 0.8)),
            text_weight=float(cfg.get("text_weight", 0.2)),
        ).to(device)
    else:
        model = SigLIPProtoModel(siglip_model).to(device)

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
    if objective == "tri_sig":
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
                use_query_features=True,
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
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    enable_cls_loss = _is_loss_enabled(cfg, "cls", True)
    enable_it_loss = _is_loss_enabled(cfg, "it", True)
    enable_ii_loss = _is_loss_enabled(cfg, "ii", True)
    enable_consistency_loss = _is_loss_enabled(cfg, "consistency", True)
    enable_margin_loss = _is_loss_enabled(cfg, "adaptive_margin", False)
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

        if objective == "tri_sig":
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

        if objective == "tri_sig":
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
                num_classes=num_classes,
                max_steps=debug_max_steps,
                enable_cls_loss=enable_cls_loss,
                enable_it_loss=enable_it_loss,
                enable_ii_loss=enable_ii_loss,
                enable_consistency_loss=enable_consistency_loss,
                enable_margin_loss=enable_margin_loss,
                margin_min=margin_min,
                margin_max=margin_max,
            )
            print(
                f"[{stage}] Train loss={train_loss:.4f}, acc@1={train_acc1:.4f}, acc@5={train_acc5:.4f}, "
                f"cls={train_stats['loss_cls']:.4f}, it={train_stats['loss_it']:.4f}, "
                f"ii={train_stats['loss_ii']:.4f}, margin={train_stats['loss_margin']:.4f}"
            )
            print(
                f"[{stage}] Val   loss={val_loss:.4f}, acc@1={val_acc1:.4f}, acc@5={val_acc5:.4f}, "
                f"mean_per_class={val_stats['mean_per_class_acc']:.4f}"
            )
            primary_acc = val_acc1
        else:
            train_loss, train_acc1, train_acc5, _ = train_one_epoch_stage1(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                amp=amp,
                proto_cache=proto_cache,
                grad_clip_norm=grad_clip_norm,
                log_interval=log_interval,
                scaler=scaler,
                num_classes=num_classes,
                max_steps=debug_max_steps,
                grad_accum_steps=grad_accum_steps,
            )
            val_loss, val_acc1, val_acc5, _ = evaluate_stage1(
                model,
                val_loader,
                proto_cache,
                device,
                amp=amp,
                num_classes=num_classes,
                max_steps=debug_max_steps,
            )
            print(f"[{stage}] Train loss={train_loss:.4f}, acc@1={train_acc1:.4f}, acc@5={train_acc5:.4f}")
            print(f"[{stage}] Val   loss={val_loss:.4f}, acc@1={val_acc1:.4f}, acc@5={val_acc5:.4f}")
            primary_acc = val_acc1

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

    if objective == "tri_sig" and image_prototypes is not None:
        torch.save(image_prototypes.detach().cpu(), output_dir / "image_prototypes.pt")

    print(f"[{stage}] Training complete. Best model: {best_path}")
    return best_path
