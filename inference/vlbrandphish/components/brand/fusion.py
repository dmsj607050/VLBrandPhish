"""Shared adapter and router layers used by training and checkpoint inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ResidualAdapter(nn.Module):
    """Learn a residual feature correction for query logo embeddings."""
    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the learned residual projection to an embedding batch."""
        return self.net(features)


class LogoTypeRouter(nn.Module):
    """Adaptively weight image/text evidence while supporting legacy checkpoints."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        stats_dim: int = 2,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.stats_dim = int(stats_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + self.stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        ocr_available: torch.Tensor | None = None,
        ocr_strength: torch.Tensor | None = None,
        branch_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized branch weights for the available recognition evidence."""
        if self.out_dim == 2:
            if branch_stats is None:
                branch_stats = torch.zeros(
                    (int(query.shape[0]), self.stats_dim), dtype=query.dtype, device=query.device
                )
            return functional.softmax(self.net(torch.cat([query, branch_stats.float()], dim=-1)), dim=-1)

        if ocr_available is None:
            ocr_available = torch.zeros((int(query.shape[0]),), dtype=torch.bool, device=query.device)
        if ocr_strength is None:
            ocr_strength = ocr_available.float()
        features = torch.cat(
            [query, ocr_available.float().unsqueeze(-1), ocr_strength.float().unsqueeze(-1)], dim=-1
        )
        weights = functional.softmax(self.net(features), dim=-1)
        if (~ocr_available).any():
            # Redistribute unavailable OCR weight to image and text while retaining a valid distribution.
            redistributed = weights.clone()
            missing = ~ocr_available
            redistributed[missing, 0] += weights[missing, 2] * 0.4
            redistributed[missing, 1] += weights[missing, 2] * 0.6
            redistributed[missing, 2] = 0.0
            weights = redistributed / redistributed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights
