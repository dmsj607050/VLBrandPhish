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
    """Build a margin matrix from pairwise text-prototype similarity.

    Classes whose text prototypes are more similar receive larger margins,
    pushing the model to learn finer distinctions between visually close brands.
    """
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
