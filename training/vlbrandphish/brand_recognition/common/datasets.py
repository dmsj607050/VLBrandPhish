from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _stem_ends_with_digit(filename: str) -> bool:
    stem = os.path.splitext(filename)[0]
    return bool(stem) and stem[-1].isdigit()


def _build_prompts_for_brand(brand_clean: str) -> List[str]:
    templates = [
        "{b}",
        "{b} logo",
        "logo of {b}",
        "official logo of {b}",
        "{b} brand logo",
        "a photo of the {b} logo",
        "a clean logo of {b}",
        "a vector logo of {b}",
        "{b} logomark",
        "{b} wordmark",
        "the {b} logo on a website",
        "the {b} logo in a webpage screenshot",
        "a cropped {b} logo",
    ]
    prompts: List[str] = []
    seen = set()
    for template in templates:
        prompt = " ".join(template.format(b=brand_clean).split())
        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def _limit_samples_per_class(samples: List[Dict], *, max_per_class: int) -> List[Dict]:
    if max_per_class <= 0:
        return samples
    buckets: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        buckets[int(sample["label"])].append(sample)
    limited: List[Dict] = []
    for label in sorted(buckets.keys()):
        limited.extend(buckets[label][:max_per_class])
    return limited


def build_logo2k_splits(
    root: str,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
    debug_max_train_per_class: int = 0,
    debug_max_val_per_class: int = 0,
) -> Tuple[List[Dict], List[Dict], Dict[int, List[str]], Dict[int, str]]:
    random.seed(seed)
    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    label_to_prompts: Dict[int, List[str]] = {}
    label_to_name: Dict[int, str] = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Logo2K+ root not found: {root}")

    label = 0
    top_categories = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for category in top_categories:
        category_dir = os.path.join(root, category)
        brand_dirs = sorted(d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d)))
        for brand in brand_dirs:
            brand_dir = os.path.join(category_dir, brand)
            files = [f for f in os.listdir(brand_dir) if _is_image_file(f)]
            if not files:
                continue
            files.sort()
            random.shuffle(files)

            brand_clean = brand.replace("_", " ").strip()
            label_to_name[label] = brand_clean
            label_to_prompts[label] = _build_prompts_for_brand(brand_clean)

            n = len(files)
            n_val = max(1, int(n * float(val_ratio)))
            n_val = min(n_val, n - 1) if n > 1 else 0
            cut = n - n_val

            for idx, filename in enumerate(files):
                sample = {"image_path": os.path.join(brand_dir, filename), "label": label}
                if idx < cut:
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)
            label += 1

    train_samples = _limit_samples_per_class(train_samples, max_per_class=int(debug_max_train_per_class))
    val_samples = _limit_samples_per_class(val_samples, max_per_class=int(debug_max_val_per_class))

    if not train_samples or not val_samples:
        raise RuntimeError("No samples found for training/validation in Logo2K+.")

    print(f"[Logo2K+] Loaded {len(train_samples)} train, {len(val_samples)} val samples, {label} classes")
    return train_samples, val_samples, label_to_prompts, label_to_name


def build_flat_splits(
    root: str,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    numeric_suffix_only: bool = False,
    excluded_filenames: Optional[List[str]] = None,
    debug_max_train_per_class: int = 0,
    debug_max_val_per_class: int = 0,
) -> Tuple[List[Dict], List[Dict], Dict[int, List[str]], Dict[int, str]]:
    random.seed(seed)
    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    label_to_prompts: Dict[int, List[str]] = {}
    label_to_name: Dict[int, str] = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    excluded = {str(x).strip().lower() for x in (excluded_filenames or []) if str(x).strip()}
    brand_dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    label = 0
    for brand in brand_dirs:
        brand_dir = os.path.join(root, brand)
        files = [
            f
            for f in os.listdir(brand_dir)
            if _is_image_file(f)
            and (not excluded or f.lower() not in excluded)
            and (not numeric_suffix_only or _stem_ends_with_digit(f))
        ]
        if not files:
            continue
        files.sort()
        random.shuffle(files)

        brand_clean = brand.replace("_", " ").strip()
        label_to_name[label] = brand_clean
        label_to_prompts[label] = _build_prompts_for_brand(brand_clean)

        n = len(files)
        n_val = max(1, int(n * float(val_ratio)))
        n_val = min(n_val, n - 1) if n > 1 else 0
        cut = n - n_val
        for idx, filename in enumerate(files):
            sample = {"image_path": os.path.join(brand_dir, filename), "label": label}
            if idx < cut:
                train_samples.append(sample)
            else:
                val_samples.append(sample)
        label += 1

    train_samples = _limit_samples_per_class(train_samples, max_per_class=int(debug_max_train_per_class))
    val_samples = _limit_samples_per_class(val_samples, max_per_class=int(debug_max_val_per_class))

    if not train_samples or not val_samples:
        raise RuntimeError("No samples found for training/validation in flat dataset.")

    print(f"[Flat] Loaded {len(train_samples)} train, {len(val_samples)} val samples, {label} classes")
    return train_samples, val_samples, label_to_prompts, label_to_name


class LogoDataset(Dataset):
    def __init__(self, samples: List[Dict], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        image = image.convert("RGB")

        bbox = sample.get("bbox")
        if bbox is not None and isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            try:
                pad = int(sample.get("pad", 0))
            except Exception:
                pad = 0
            width, height = image.size
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1 - pad, width))
            y1 = max(0, min(y1 - pad, height))
            x2 = max(0, min(x2 + pad, width))
            y2 = max(0, min(y2 + pad, height))
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))

        return (
            self.transform(image),
            int(sample["label"]),
            str(sample["image_path"]),
        )
