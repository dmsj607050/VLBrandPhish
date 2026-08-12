"""Stage-2 fusion and TriSig training and validation routines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..common.runtime import MetricsTracker, _get_autocast_context, _unpack_batch
from ..common.losses import (
    adaptive_margin_sigmoid_loss as compute_adaptive_margin_sigmoid_loss,
    multibranch_consistency_loss as compute_multibranch_consistency_loss,
    multi_positive_sigmoid_loss as compute_multi_positive_sigmoid_loss,
    sigmoid_loss_class_proto as compute_sigmoid_loss_class_proto,
)
from ..common.model import SigLIPTriSigModel
from ..common.optimization import WarmupCosineLRScheduler

# Keys of the per-component loss dict returned by ``_compute_trisig_losses``.
LOSS_COMPONENT_KEYS: Tuple[str, ...] = ("cls", "it", "ii", "consistency", "margin")


def _compute_trisig_losses(
    model: SigLIPTriSigModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_cls: float,
    lambda_it: float,
    lambda_ii: float,
    lambda_consistency: float,
    lambda_margin: float,
    label_smoothing: float = 0.0,
    enable_cls_loss: bool = True,
    enable_it_loss: bool = True,
    enable_ii_loss: bool = True,
    enable_consistency_loss: bool = True,
    enable_margin_loss: bool = False,
    margin_min: float = 0.0,
    margin_max: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute all TriSig loss components and the weighted combination.

    Shared by training and validation so both paths use identical loss math.
    Returns ``(loss, logits_fused, components)`` where ``components`` maps each
    name in :data:`LOSS_COMPONENT_KEYS` to its scalar tensor.
    """
    query_features = model.encode_query_features(images)
    logits_fused, logits_text, logits_image = model.forward_with_features(query_features)
    scale = model.get_logit_scale()
    bias = model.get_logit_bias()

    text_prototypes = F.normalize(model.text_prototypes, dim=-1)
    image_prototypes = F.normalize(model.image_prototypes, dim=-1)

    loss_cls = (
        F.cross_entropy(logits_fused, labels, label_smoothing=float(label_smoothing))
        if enable_cls_loss
        else logits_fused.new_tensor(0.0)
    )
    loss_it = (
        compute_sigmoid_loss_class_proto(
            query_features, text_prototypes, labels,
            logit_scale=scale, logit_bias=bias,
        )
        if enable_it_loss
        else logits_fused.new_tensor(0.0)
    )
    if enable_ii_loss:
        if model.image_prototypes.ndim == 3:
            loss_ii = compute_multi_positive_sigmoid_loss(
                query_features, image_prototypes, labels,
                logit_scale=scale, logit_bias=bias,
            )
        else:
            loss_ii = compute_sigmoid_loss_class_proto(
                query_features, image_prototypes, labels,
                logit_scale=scale, logit_bias=bias,
            )
    else:
        loss_ii = logits_fused.new_tensor(0.0)

    consistency = (
        compute_multibranch_consistency_loss(logits_text, logits_image)
        if enable_consistency_loss
        else logits_fused.new_tensor(0.0)
    )
    margin_loss = (
        compute_adaptive_margin_sigmoid_loss(
            query_features, text_prototypes, labels,
            logit_scale=scale, logit_bias=bias,
            min_margin=float(margin_min), max_margin=float(margin_max),
        )
        if enable_margin_loss
        else logits_fused.new_tensor(0.0)
    )

    loss = (
        float(lambda_cls) * loss_cls
        + float(lambda_it) * loss_it
        + float(lambda_ii) * loss_ii
        + float(lambda_consistency) * consistency
        + float(lambda_margin) * margin_loss
    )
    components = {
        "cls": loss_cls,
        "it": loss_it,
        "ii": loss_ii,
        "consistency": consistency,
        "margin": margin_loss,
    }
    return loss, logits_fused, components


def _accumulate_loss_stats(
    loss_components: Dict[str, float],
    components: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    """Accumulate per-component loss values weighted by batch size.

    The combined ``loss`` is accumulated separately by the caller for ``total_loss``.
    """
    for key in LOSS_COMPONENT_KEYS:
        loss_components[key] += components[key].item() * batch_size


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
    margin_min: float = 0.0,
    margin_max: float = 0.3,
) -> Tuple[float, float, float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    loss_components: Dict[str, float] = defaultdict(float)
    grad_accum_steps = max(1, int(grad_accum_steps))

    pbar = tqdm(loader, desc=f"Train(TriSig) E{epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    for step, batch in enumerate(pbar, start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _get_autocast_context(device, amp):
            loss, logits_fused, components = _compute_trisig_losses(
                model, images, labels,
                lambda_cls=lambda_cls,
                lambda_it=lambda_it,
                lambda_ii=lambda_ii,
                lambda_consistency=lambda_consistency,
                lambda_margin=lambda_margin,
                label_smoothing=label_smoothing,
                enable_cls_loss=enable_cls_loss,
                enable_it_loss=enable_it_loss,
                enable_ii_loss=enable_ii_loss,
                enable_consistency_loss=enable_consistency_loss,
                enable_margin_loss=enable_margin_loss,
                margin_min=margin_min,
                margin_max=margin_max,
            )

        scaler.scale(loss).backward()
        accum_count += 1
        is_last = (max_steps > 0 and step >= max_steps) or (step == len(loader))
        should_step = (accum_count % grad_accum_steps == 0) or is_last
        if should_step:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
            # Normalise gradients by actual number of accumulated batches,
            # so the final partial step has the correct gradient magnitude.
            if accum_count > 1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(float(accum_count))
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _accumulate_loss_stats(loss_components, components, batch_size)
        metrics.update(logits_fused, labels)

        if step % int(log_interval) == 0:
            acc1, _ = metrics.get_accuracy()
            pbar.set_postfix({"loss": f"{total_loss / max(1, metrics.total_samples):.4f}", "acc@1": f"{acc1:.4f}"})

        if max_steps > 0 and step >= max_steps:
            break

    samples = max(1, metrics.total_samples)
    acc1, acc5 = metrics.get_accuracy()
    return total_loss / samples, acc1, acc5, {
        f"loss_{key}": loss_components[key] / samples for key in LOSS_COMPONENT_KEYS
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
    num_classes: int,
    max_steps: int = 0,
    enable_cls_loss: bool = True,
    enable_it_loss: bool = True,
    enable_ii_loss: bool = True,
    enable_consistency_loss: bool = True,
    enable_margin_loss: bool = False,
    margin_min: float = 0.0,
    margin_max: float = 0.3,
) -> Tuple[float, float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics = MetricsTracker(num_classes)
    loss_components: Dict[str, float] = defaultdict(float)

    for step, batch in enumerate(tqdm(loader, desc="Val(TriSig)", leave=False), start=1):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _get_autocast_context(device, amp):
            loss, logits_fused, components = _compute_trisig_losses(
                model, images, labels,
                lambda_cls=lambda_cls,
                lambda_it=lambda_it,
                lambda_ii=lambda_ii,
                lambda_consistency=lambda_consistency,
                lambda_margin=lambda_margin,
                enable_cls_loss=enable_cls_loss,
                enable_it_loss=enable_it_loss,
                enable_ii_loss=enable_ii_loss,
                enable_consistency_loss=enable_consistency_loss,
                enable_margin_loss=enable_margin_loss,
                margin_min=margin_min,
                margin_max=margin_max,
            )

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _accumulate_loss_stats(loss_components, components, batch_size)
        metrics.update(logits_fused, labels)

        if max_steps > 0 and step >= max_steps:
            break

    samples = max(1, metrics.total_samples)
    acc1, acc5 = metrics.get_accuracy()
    result = {f"loss_{key}": loss_components[key] / samples for key in LOSS_COMPONENT_KEYS}
    result["mean_per_class_acc"] = metrics.get_mean_per_class_accuracy()
    return total_loss / samples, acc1, acc5, result


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
            logits, _, _ = model(images)
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
