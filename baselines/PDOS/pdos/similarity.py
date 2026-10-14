"""Feature normalization and cosine similarity."""

from __future__ import annotations

import torch


def l2_normalize(feature: torch.Tensor) -> torch.Tensor:
    feature = feature.flatten(start_dim=1) if feature.ndim > 1 else feature.flatten().unsqueeze(0)
    return torch.nn.functional.normalize(feature, p=2, dim=1).squeeze(0)


def cosine_similarity(feature_a: torch.Tensor, feature_b: torch.Tensor) -> float:
    a = l2_normalize(feature_a).float()
    b = l2_normalize(feature_b).float()
    return float(torch.dot(a, b).item())


def max_reference_similarity(query: torch.Tensor, references: list[torch.Tensor]) -> tuple[float, int]:
    if not references:
        raise ValueError("At least one reference feature is required")
    scores = [cosine_similarity(query, reference) for reference in references]
    index = max(range(len(scores)), key=scores.__getitem__)
    return scores[index], index
