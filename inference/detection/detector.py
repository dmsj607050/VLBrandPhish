"""Thin, lazy MMDetection adapter for logo detection."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


class MMDetTopKDetector:
    """Run an MMDetection model and return the highest-scoring logo boxes."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0") -> None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"detector config not found: {config_path}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"detector checkpoint not found: {checkpoint_path}")
        try:
            from mmdet.apis import init_detector  # type: ignore
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("Missing optional dependency 'mmdet'.") from error

        self._model = init_detector(
            os.path.abspath(config_path),
            os.path.abspath(checkpoint_path),
            device=str(device),
        )

    def detect_topk(
        self,
        image_path: str,
        score_thr: float = 0.05,
        topk: int = 3,
    ) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """Detect and rank the highest-confidence logo boxes in one screenshot."""
        try:
            from mmdet.apis import inference_detector  # type: ignore
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("Missing optional dependency 'mmdet'.") from error

        result = inference_detector(self._model, image_path)
        # MMDetection 3.x stores predictions in the pred_instances attribute.
        instances = getattr(result, "pred_instances", None)
        if instances is None:
            return []
        bboxes = getattr(instances, "bboxes", None)
        scores = getattr(instances, "scores", None)
        if bboxes is None or scores is None:
            return []

        boxes = bboxes.detach().cpu().numpy().astype(np.float32)
        values = scores.detach().cpu().numpy().astype(np.float32)
        # Filter by confidence before retaining the highest-scoring top-k boxes.
        keep = values >= float(score_thr)
        boxes, values = boxes[keep], values[keep]
        if boxes.size == 0:
            return []

        ordered = np.argsort(-values)
        return [
            (tuple(float(v) for v in boxes[int(index)][:4]), float(values[int(index)]))
            for index in ordered[: max(1, int(topk))].tolist()
        ]
