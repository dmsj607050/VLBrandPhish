from __future__ import annotations

import random
from typing import Dict, Tuple

from PIL import Image, ImageFilter
from torchvision import transforms


class RandomGaussianBlur:
    def __init__(self, p: float = 0.3, radius_min: float = 0.1, radius_max: float = 1.2):
        self.p = float(p)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        radius = self.radius_min + (self.radius_max - self.radius_min) * random.random()
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomLowRes:
    def __init__(self, p: float = 0.6, min_scale: float = 0.2, max_scale: float = 0.6):
        self.p = float(p)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        width, height = image.size
        scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
        new_width = max(8, int(width * scale))
        new_height = max(8, int(height * scale))
        low_res = image.resize((new_width, new_height), resample=Image.BILINEAR)
        return low_res.resize((width, height), resample=Image.BILINEAR)


def _resolve_size(image_size_cfg: Dict) -> Tuple[int, int]:
    if "height" in image_size_cfg and "width" in image_size_cfg:
        return int(image_size_cfg["height"]), int(image_size_cfg["width"])
    if "shortest_edge" in image_size_cfg:
        size = int(image_size_cfg["shortest_edge"])
        return size, size
    if len(image_size_cfg) == 1:
        size = int(next(iter(image_size_cfg.values())))
        return size, size
    raise ValueError(f"Unsupported image size config: {image_size_cfg}")


def build_transforms(*, image_size_cfg: Dict, image_mean, image_std, train: bool):
    height, width = _resolve_size(image_size_cfg)
    normalize = transforms.Normalize(tuple(image_mean), tuple(image_std))

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop((height, width), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            RandomLowRes(p=0.6, min_scale=0.2, max_scale=0.6),
            transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.25, 0.05)], p=0.6),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
            RandomGaussianBlur(p=0.3, radius_min=0.1, radius_max=1.2),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
