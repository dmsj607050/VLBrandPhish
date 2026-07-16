"""Optimizer grouping and learning-rate scheduling for brand training."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .model import get_layer_id_for_siglip


class WarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
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

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        weight_decay_for_param = 0.0 if any(keyword in name.lower() for keyword in no_decay_keywords) else weight_decay
        group_name = "other"
        lr_scale = lr_scales["other"]
        if "vision_model" in name:
            group_name, lr_scale = "vision", lr_scales["vision"]
            if layer_decay < 1.0:
                layer_id = get_layer_id_for_siglip(name, num_vision_layers, "vision")
                lr_scale *= layer_decay ** (num_vision_layers - layer_id)
        elif "text_model" in name:
            group_name, lr_scale = "text", lr_scales["text"]
            if layer_decay < 1.0:
                layer_id = get_layer_id_for_siglip(name, num_text_layers, "text")
                lr_scale *= layer_decay ** (num_text_layers - layer_id)
        elif "gate" in name:
            group_name, lr_scale = "gate", lr_scales["gate"]
        elif "router" in name:
            group_name, lr_scale = "router", lr_scales["router"]
        elif "adapter" in name:
            group_name, lr_scale = "adapter", lr_scales["adapter"]
        elif "logit_scale" in name or "logit_bias" in name:
            group_name, lr_scale, weight_decay_for_param = "logit", lr_scales["logit"], 0.0

        key = f"{group_name}|{lr_scale:.8f}|{weight_decay_for_param:.6f}"
        grouped.setdefault(
            key,
            {"params": [], "lr": base_lr * lr_scale, "weight_decay": weight_decay_for_param, "name": group_name},
        )["params"].append(parameter)

    groups = list(grouped.values())
    print(f"[optimizer] Created {len(groups)} parameter groups")
    for index, group in enumerate(groups):
        count = sum(parameter.numel() for parameter in group["params"])
        print(f"  - group {index} [{group['name']}]: {count:,} params, lr={group['lr']:.2e}, wd={group['weight_decay']}")
    return groups
