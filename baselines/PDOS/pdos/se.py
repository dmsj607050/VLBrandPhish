"""Stable import path for the PDOS Squeeze-and-Excitation block.

The module path is intentionally part of the checkpoint format: PyTorch
stores the class path when serializing a complete model object.
"""

from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    """Channel attention with the reduction ratio specified by the paper."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        hidden = max(channels // reduction, 1)
        self.channels = channels
        self.reduction = reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"SEBlock expected [B, {self.channels}, H, W], got {tuple(x.shape)}"
            )
        scale = self.pool(x)
        scale = self.fc2(self.relu(self.fc1(scale)))
        return x * self.sigmoid(scale)
