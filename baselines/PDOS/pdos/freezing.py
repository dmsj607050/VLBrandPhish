"""Explicit implementation of the paper's first-10-convolution freeze."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def freeze_first_backbone_convolutions(model: nn.Module, count: int = 10) -> list[str]:
    """Freeze parameters belonging to the first ``count`` Conv2d layers.

    The traversal is restricted to the YOLO backbone modules before SPPF and
    returns the exact convolution module names for auditability.
    """
    backbone = model.model[:11] if hasattr(model, "model") else model
    convolution_names: list[str] = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            convolution_names.append(name)
    selected = convolution_names[:count]
    for name, module in backbone.named_modules():
        if name in selected:
            for parameter in module.parameters():
                parameter.requires_grad = False
    return selected
