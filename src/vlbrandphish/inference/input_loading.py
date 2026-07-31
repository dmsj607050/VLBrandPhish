"""Load inference inputs and resolve screenshots from detection records."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .data_contracts import DetectionSample

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _iter_items(items, *, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        yield from tqdm(items, desc=desc)
        return
    yield from items

def _safe_crop(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Optional[Image.Image]:
    w, h = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, float(w)))
    x2 = max(0.0, min(x2, float(w)))
    y1 = max(0.0, min(y1, float(h)))
    y2 = max(0.0, min(y2, float(h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))

def _resolve_benign_path(benign_root: str, sample_key: str) -> str:
    direct = os.path.join(benign_root, sample_key, f"{sample_key}.png")
    if os.path.isfile(direct):
        return direct
    sample_dir = os.path.join(benign_root, sample_key)
    if os.path.isdir(sample_dir):
        for fn in sorted(os.listdir(sample_dir)):
            if os.path.splitext(fn)[0] == sample_key:
                p = os.path.join(sample_dir, fn)
                if os.path.isfile(p):
                    return p
    return ""

def _resolve_phish_path(phish_root: str, sample_key: str, gt_brand: Optional[str]) -> str:
    if "/" in sample_key:
        brand, stem = sample_key.split("/", 1)
    else:
        brand = str(gt_brand or "").strip()
        stem = sample_key

    if not brand or not stem:
        return ""

    screens_dir = os.path.join(phish_root, brand, "screens")
    if not os.path.isdir(screens_dir):
        return ""

    direct = os.path.join(screens_dir, f"{stem}.png")
    if os.path.isfile(direct):
        return direct

    for fn in sorted(os.listdir(screens_dir)):
        if os.path.splitext(fn)[0] == stem:
            p = os.path.join(screens_dir, fn)
            if os.path.isfile(p):
                return p
    return ""

def resolve_image_path(det: DetectionSample, benign_root: str, phish_root: str) -> str:
    if det.image_path and os.path.isfile(det.image_path):
        return os.path.abspath(det.image_path)

    if det.gt_is_phishing:
        p = _resolve_phish_path(phish_root, det.sample_key, det.gt_brand)
    else:
        p = _resolve_benign_path(benign_root, det.sample_key)

    if p and os.path.isfile(p):
        return os.path.abspath(p)

    raise FileNotFoundError(
        "Cannot resolve screenshot path for sample "
        f"sample_key='{det.sample_key}', gt_is_phishing={int(det.gt_is_phishing)}, image_path='{det.image_path}'"
    )

def _inspect_detection_csv_format(path: str) -> Dict[str, object]:
    row_count = 0
    min_cols = 10**9
    max_cols = 0
    first_nonempty: List[str] = []

    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not first_nonempty:
                first_nonempty = [str(x) for x in row]
            n = len(row)
            row_count += 1
            min_cols = min(min_cols, n)
            max_cols = max(max_cols, n)

    if row_count == 0:
        min_cols = 0

    has_header = bool(first_nonempty) and str(first_nonempty[0]).strip().lower() == "image_path"
    return {
        "row_count": int(row_count),
        "has_header": bool(has_header),
        "first_row_ncols": int(len(first_nonempty)),
        "min_cols": int(min_cols),
        "max_cols": int(max_cols),
        "first_row_preview": first_nonempty[:8],
    }
