"""YOLOv5 v7.0 detector wrapper with a paper-specified SE insertion."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .preprocessing import load_rgb_image
from .freezing import freeze_first_backbone_convolutions


CLASS_NAMES = [
    "google", "search", "chrome", "mail", "map", "play", "chat", "picture",
    "meeting", "amazon_1", "amazon_2", "alibaba", "twitter_1", "twitter_2",
    "facebook_1", "facebook_2",
]


class YOLOv5LogoDetector:
    def __init__(
        self,
        project_root: str | Path,
        weights: str | Path | None = None,
        device: str | torch.device = "auto",
        confidence: float = 0.25,
        iou: float = 0.45,
        config: str | Path | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.yolov5_root = self.project_root / "third_party" / "yolov5"
        self.config = Path(config) if config else self.project_root / "configs" / "pdos_yolov5s_se.yaml"
        self.class_names = class_names or CLASS_NAMES
        self.device = _resolve_device(device)
        self.confidence = confidence
        self.iou = iou
        if not self.yolov5_root.exists():
            raise FileNotFoundError(f"Vendored YOLOv5 checkout missing: {self.yolov5_root}")
        self._import_yolov5()
        self.model = self._build_model()
        self.weights_source = None
        self.weights_loaded = False
        if weights:
            self._load_weights(Path(weights))
        self.model.to(self.device).eval()

    def _import_yolov5(self) -> None:
        _install_ipython_compat()
        sys.path.insert(0, str(self.yolov5_root))
        import models.yolo as yolo_module  # type: ignore
        from utils.general import non_max_suppression, scale_boxes  # type: ignore
        from utils.augmentations import letterbox  # type: ignore

        # Keep the class on a stable import path so complete torch checkpoints
        # can be unpickled after training.
        from .se import SEBlock
        yolo_module.SEBlock = SEBlock
        self._yolo_module = yolo_module
        self._non_max_suppression = non_max_suppression
        self._scale_boxes = scale_boxes
        self._letterbox = letterbox

    def _build_model(self):
        model = self._yolo_module.DetectionModel(str(self.config), nc=len(self.class_names))
        model.names = self.class_names
        self.frozen_conv_names = freeze_first_backbone_convolutions(model, count=10)
        return model

    def _load_weights(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {path}")
        try:
            # The YOLOv5 checkpoint is a trusted complete model object; this
            # is required for PyTorch 2.6+ where weights_only defaults True.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("ultralytics"):
                raise RuntimeError(
                    "The supplied checkpoint is an Ultralytics-format model "
                    "(for example yolov5su.pt), but this reproduction fixes "
                    "the YOLOv5 v7.0 codebase. Provide a YOLOv5 v7.0 "
                    "yolov5s.pt or a trained checkpoint exported for "
                    "configs/pdos_yolov5s_se.yaml; do not mix the formats."
                ) from exc
            raise
        source_model = checkpoint.get("ema") or checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if hasattr(source_model, "float"):
            source_model = source_model.float()
        source_state = source_model.state_dict() if hasattr(source_model, "state_dict") else source_model
        target_state = self.model.state_dict()
        compatible = {k: v for k, v in source_state.items() if k in target_state and target_state[k].shape == v.shape}
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        self.weights_source = str(path)
        self.weights_loaded = True
        self.partial_weight_load = {
            "compatible_tensors": len(compatible),
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
            "note": "COCO weights initialize the shared backbone; the 16-class logo head still requires logo training.",
        }

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        image = load_rgb_image(image_path)
        original = (image.height, image.width)
        array = __import__("numpy").asarray(image)
        letterboxed = self._letterbox(array, new_shape=(640, 640), auto=False, stride=32)[0]
        tensor = torch.from_numpy(letterboxed.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)
        prediction = self.model(tensor)
        prediction = prediction[0] if isinstance(prediction, tuple) else prediction
        detections = self._non_max_suppression(prediction, self.confidence, self.iou, max_det=100)[0]
        if len(detections):
            detections[:, :4] = self._scale_boxes(tensor.shape[2:], detections[:, :4], original).round()
        output = []
        for *xyxy, confidence, class_id in detections.detach().cpu().tolist():
            class_index = int(class_id)
            output.append({
                "bbox": [float(v) for v in xyxy],
                "confidence": float(confidence),
                "class_id": class_index,
                "category": self.class_names[class_index] if class_index < len(self.class_names) else str(class_index),
            })
        return output


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _install_ipython_compat() -> None:
    """Keep YOLOv5's optional notebook import optional on a minimal runtime."""
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
    if "IPython" in sys.modules:
        return
    ipython = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = lambda *args, **kwargs: None
    ipython.display = display_module
    ipython.version_info = (0, 0)
    ipython.get_ipython = lambda: None
    sys.modules["IPython"] = ipython
    sys.modules["IPython.display"] = display_module
    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(available=8 * (1 << 30), total=16 * (1 << 30))
        sys.modules["psutil"] = psutil
    if "seaborn" not in sys.modules:
        # YOLOv5 imports seaborn from plotting helpers, although inference does
        # not call the plotting-only functions that use it.
        seaborn = types.ModuleType("seaborn")
        seaborn.pairplot = lambda *args, **kwargs: None
        seaborn.histplot = lambda *args, **kwargs: None
        sys.modules["seaborn"] = seaborn
