"""Stage-1 pairwise SigLIP training and validation routines."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..engine import MetricsTracker, _get_autocast_context, _index_text_prototypes, _unpack_batch
from ..common.losses import siglip_pairwise_loss as compute_siglip_pairwise_loss
from ..common.model import ProtoCache, SigLIPProtoModel
from ..common.optimization import WarmupCosineLRScheduler

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
