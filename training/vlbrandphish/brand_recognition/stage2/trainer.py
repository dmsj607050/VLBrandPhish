"""Stage-2 fusion and TriSig training and validation routines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..engine import MetricsTracker, _get_autocast_context, _unpack_batch
from ..common.losses import (
    adaptive_margin_sigmoid_loss as compute_adaptive_margin_sigmoid_loss,
    is_loss_enabled as _is_loss_enabled,
    multibranch_consistency_loss as compute_multibranch_consistency_loss,
    multi_positive_sigmoid_loss as compute_multi_positive_sigmoid_loss,
    router_oracle_supervision_loss as compute_router_oracle_supervision_loss,
    router_pseudo_supervision_loss as compute_router_pseudo_supervision_loss,
    sigmoid_loss_class_proto as compute_sigmoid_loss_class_proto,
    uncertainty_alignment_loss as compute_uncertainty_alignment_loss,
)
from ..common.model import SigLIPDualFusionModel, SigLIPTriSigModel
from ..common.optimization import WarmupCosineLRScheduler

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
