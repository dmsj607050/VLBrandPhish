from __future__ import annotations

from typing import Any, Optional


def patch_loss_cfg(cfg: Any, num_classes: int) -> Any:
    model = cfg.get("model")
    if model is None:
        return cfg

    bbox_head = model.get("bbox_head") if isinstance(model, dict) else None
    if bbox_head is None:
        return cfg

    if isinstance(bbox_head, dict):
        if "loss_cls" in bbox_head and isinstance(bbox_head["loss_cls"], dict):
            bbox_head["loss_cls"].setdefault("type", "FocalLoss")
            bbox_head["loss_cls"]["use_sigmoid"] = True

        if "num_classes" in bbox_head:
            bbox_head["num_classes"] = int(num_classes)

    return cfg
