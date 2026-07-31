"""Objective functions used by the brand-recognizer training loops."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def is_loss_enabled(cfg: Dict[str, Any], key: str, default: bool = True) -> bool:
    loss_cfg = cfg.get("losses") or {}
    if not isinstance(loss_cfg, dict):
        return default
    value = loss_cfg.get(key)
    return default if value is None else bool(value)


def _adaptive_margin_matrix(text_prototypes: torch.Tensor, *, min_margin: float, max_margin: float) -> torch.Tensor:
    prototypes = F.normalize(text_prototypes, dim=-1)
    similarity = prototypes @ prototypes.t()
    margins = min_margin + (max_margin - min_margin) * similarity.clamp(0.0, 1.0)
    margins.fill_diagonal_(0.0)
    return margins


def adaptive_margin_sigmoid_loss(
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
    margins = _adaptive_margin_matrix(class_prototypes, min_margin=float(min_margin), max_margin=float(max_margin))
    adjusted_logits = logits - margins.to(logits.device, dtype=logits.dtype)[labels.long()]
    targets = -torch.ones_like(adjusted_logits)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    return (-F.logsigmoid(targets * adjusted_logits).sum(dim=-1)).mean()


def multibranch_consistency_loss(logits_text: torch.Tensor, logits_image: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(logits_text), torch.sigmoid(logits_image))


def router_pseudo_supervision_loss(
    route_weights: torch.Tensor, logits_text: torch.Tensor, logits_image: torch.Tensor
) -> torch.Tensor:
    text_confidence = torch.sigmoid(logits_text.detach().amax(dim=-1))
    image_confidence = torch.sigmoid(logits_image.detach().amax(dim=-1))
    evidence = torch.stack([text_confidence, image_confidence], dim=-1).to(route_weights.dtype)
    prior = torch.tensor([0.5, 0.5], device=route_weights.device, dtype=route_weights.dtype)
    target = prior.unsqueeze(0) * (0.5 + evidence)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="batchmean")


def router_oracle_supervision_loss(
    route_weights: torch.Tensor,
    logits_text: torch.Tensor,
    logits_image: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    with torch.no_grad():
        text_loss = F.cross_entropy(logits_text, labels, reduction="none")
        image_loss = F.cross_entropy(logits_image, labels, reduction="none")
        target = F.softmax(torch.stack([-text_loss, -image_loss], dim=-1) / max(float(temperature), 1e-6), dim=-1)
    return F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="batchmean")


def uncertainty_alignment_loss(route_weights: torch.Tensor, uncertainty_stats: Dict[str, Any]) -> torch.Tensor:
    text_confidence = uncertainty_stats["text"]["confidence"].detach()
    image_confidence = uncertainty_stats["image"]["confidence"].detach()
    agreement = uncertainty_stats["agreement"].detach()
    target = torch.stack(
        [
            text_confidence * (0.75 + 0.25 * agreement),
            image_confidence * (0.75 + 0.25 * agreement),
        ],
        dim=-1,
    ).to(route_weights.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(route_weights.clamp_min(1e-6).log(), target, reduction="batchmean")


def siglip_pairwise_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_labels: torch.Tensor,
    text_labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    logits = torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias
    targets = torch.where(image_labels.unsqueeze(1) == text_labels.unsqueeze(0), 1.0, -1.0).to(logits.dtype)
    return (-F.logsigmoid(targets * logits).sum(dim=-1)).mean()


def sigmoid_loss_class_proto(
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


def multi_positive_sigmoid_loss(
    query_features: torch.Tensor,
    ref_bank: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    logits = torch.einsum("bd,ckd->bck", query_features, ref_bank) * logit_scale + logit_bias
    targets = -torch.ones_like(logits)
    targets[torch.arange(labels.size(0), device=labels.device), labels.long(), :] = 1.0
    return (-F.logsigmoid(targets * logits).sum(dim=(1, 2))).mean()
