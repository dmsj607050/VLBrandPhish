"""Shared training runtime helpers used by engine and stage trainers.

Extracted from engine.py to break the circular import: stage1/stage2 trainers
previously imported these helpers from engine.py, while engine.py imported the
trainers inside run_training(). Housing them here keeps the dependency one-way.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch

from .model import ProtoCache


def _get_autocast_context(device: torch.device, enabled: bool):
    """Return an AMP autocast context manager compatible with the installed PyTorch."""
    if not enabled:
        return nullcontext()

    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "autocast"):
        try:
            return amp_namespace.autocast(device_type=device.type, enabled=enabled)
        except TypeError:
            try:
                return amp_namespace.autocast(device.type, enabled=enabled)
            except TypeError:
                pass

    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)

    return nullcontext()


def _unpack_batch(batch):
    """Unpack a DataLoader batch into (images, labels, extra) 3-tuple."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch
        if len(batch) == 2:
            images, labels = batch
            return images, labels, [""] * labels.size(0)
    raise ValueError("Unexpected batch format.")


class MetricsTracker:
    """Accumulate top-1/top-5 accuracy and per-class statistics."""

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        self.total_correct_top1 = 0
        self.total_correct_top5 = 0
        self.total_samples = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = logits.argmax(dim=-1)
        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            pred = int(pred)
            self.class_total[label] += 1
            if pred == label:
                self.class_correct[label] += 1
                self.total_correct_top1 += 1

        maxk = min(5, logits.size(1))
        _, topk = logits.topk(maxk, dim=1, largest=True, sorted=True)
        for idx, label in enumerate(labels.cpu().numpy()):
            if int(label) in topk[idx].cpu().numpy():
                self.total_correct_top5 += 1

        self.total_samples += labels.size(0)

    def get_accuracy(self) -> Tuple[float, float]:
        if self.total_samples == 0:
            return 0.0, 0.0
        return (
            self.total_correct_top1 / self.total_samples,
            self.total_correct_top5 / self.total_samples,
        )

    def get_mean_per_class_accuracy(self) -> float:
        values = []
        for class_id in range(self.num_classes):
            if self.class_total[class_id] > 0:
                values.append(self.class_correct[class_id] / self.class_total[class_id])
        return float(np.mean(values)) if values else 0.0

    def get_worst_classes(self, k: int = 10) -> List[Tuple[int, float, int]]:
        items = []
        for class_id in range(self.num_classes):
            if self.class_total[class_id] > 0:
                acc = self.class_correct[class_id] / self.class_total[class_id]
                items.append((class_id, acc, self.class_total[class_id]))
        items.sort(key=lambda x: x[1])
        return items[:k]


@torch.no_grad()
def _index_text_prototypes(proto_cache: ProtoCache, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Index text prototypes by label, handling custom label-to-index mappings."""
    if proto_cache.label_to_index is None:
        return proto_cache.prototypes.to(device)[labels.long()]
    indices = torch.tensor(
        [proto_cache.label_to_index[int(label.item())] for label in labels],
        device=device,
        dtype=torch.long,
    )
    return proto_cache.prototypes.to(device)[indices]
