from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TestBSample:
    image_path: str
    url: str
    gt_is_phishing: bool
    gt_brand: Optional[str]
    sample_key: str


def _first_image_by_stem(directory: str, stem: str) -> str:
    for fn in sorted(os.listdir(directory)):
        if os.path.splitext(fn)[0] == stem:
            return os.path.join(directory, fn)
    return ""


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return dict(json.load(f))


def load_testb_samples(
    benign_root: str,
    phish_root: str,
    max_benign: int = -1,
    max_phish: int = -1,
) -> List[TestBSample]:
    out: List[TestBSample] = []

    benign_count = 0
    if benign_root and os.path.isdir(benign_root):
        for key in sorted(os.listdir(benign_root)):
            if max_benign >= 0 and benign_count >= int(max_benign):
                break
            sample_dir = os.path.join(benign_root, key)
            if not os.path.isdir(sample_dir):
                continue

            image_path = os.path.join(sample_dir, f"{key}.png")
            if not os.path.isfile(image_path):
                continue

            meta_path = os.path.join(sample_dir, f"{key}.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"Missing benign json: {meta_path}")
            meta = _load_json(meta_path)

            out.append(
                TestBSample(
                    image_path=image_path,
                    url=str(meta.get("url", "") or ""),
                    gt_is_phishing=False,
                    gt_brand=None,
                    sample_key=str(key),
                )
            )
            benign_count += 1

    phish_count = 0
    if phish_root and os.path.isdir(phish_root):
        for brand in sorted(os.listdir(phish_root)):
            brand_dir = os.path.join(phish_root, brand)
            if not os.path.isdir(brand_dir):
                continue

            jsons_dir = os.path.join(brand_dir, "jsons")
            screens_dir = os.path.join(brand_dir, "screens")
            if not (os.path.isdir(jsons_dir) and os.path.isdir(screens_dir)):
                continue

            json_files = [fn for fn in sorted(os.listdir(jsons_dir)) if fn.lower().endswith(".json")]
            for jf in json_files:
                if max_phish >= 0 and phish_count >= int(max_phish):
                    break
                stem = os.path.splitext(jf)[0]
                meta_path = os.path.join(jsons_dir, jf)
                image_path = os.path.join(screens_dir, f"{stem}.png")
                if not os.path.isfile(image_path):
                    image_path = _first_image_by_stem(screens_dir, stem)
                if not image_path or (not os.path.isfile(image_path)):
                    raise FileNotFoundError(f"Missing phishing screen for stem '{stem}' under: {screens_dir}")

                meta = _load_json(meta_path)
                out.append(
                    TestBSample(
                        image_path=image_path,
                        url=str(meta.get("url", "") or ""),
                        gt_is_phishing=True,
                        gt_brand=str(brand),
                        sample_key=f"{brand}/{stem}",
                    )
                )
                phish_count += 1

            if max_phish >= 0 and phish_count >= int(max_phish):
                break

    return out
