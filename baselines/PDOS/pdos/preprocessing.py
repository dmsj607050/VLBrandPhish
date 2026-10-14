"""Image transforms shared by detector, cropper, and feature extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_to_yolo_tensor(image: Image.Image, size: int = 640) -> torch.Tensor:
    """Return a normalized CHW tensor for a YOLOv5 forward pass.

    Letterboxing is performed by the vendored YOLOv5 implementation in
    ``pdos.detector`` so that boxes can be mapped back exactly.
    """
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def crop_box(image: Image.Image, xyxy: Iterable[float], padding: float = 0.0) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    box_width, box_height = x2 - x1, y2 - y1
    x1 -= box_width * padding
    y1 -= box_height * padding
    x2 += box_width * padding
    y2 += box_height * padding
    clipped = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Invalid crop box after clipping: {clipped}")
    return image.crop(clipped)


def logo_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
