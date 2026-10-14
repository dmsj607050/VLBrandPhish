from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_ocr_cache import (
    build_paddle_ocr,
    extract_text_from_result,
    run_ocr,
)


def load_existing_cache(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Existing cache must be a JSON object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def save_cache(path: Path, cache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def should_skip_by_size(image: Image.Image, max_side: int, max_pixels: int) -> tuple[bool, str]:
    width, height = image.size
    if max_side > 0 and max(width, height) > max_side:
        return True, f"max_side={max(width, height)} > {max_side}"
    if max_pixels > 0 and width * height > max_pixels:
        return True, f"pixels={width * height} > {max_pixels}"
    return False, ""


def _iter_lines_without_nul(handle):
    for line in handle:
        yield line.replace("\x00", "")


def collect_eval_image_paths(det_csv: Path, phish_root: Path, benign_root: Path) -> list[Path]:
    image_paths: list[Path] = []
    seen: set[str] = set()

    with det_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(_iter_lines_without_nul(f))
        for row in reader:
            sample_key = str(row.get("sample_key", "") or "").strip()
            if not sample_key:
                continue
            gt_raw = str(row.get("gt_is_phish", row.get("gt_is_phishing", "0")) or "0").strip()
            root = phish_root if gt_raw in {"1", "true", "True"} else benign_root
            image_path = (root / sample_key / "shot.png").resolve()
            key = str(image_path)
            if key in seen:
                continue
            seen.add(key)
            image_paths.append(image_path)
    return image_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OCR cache JSON for exactly the images referenced by an evaluation det-csv."
    )
    parser.add_argument("--det-csv", required=True, help="Detection csv used by eval_system_trisig.py.")
    parser.add_argument("--phish-root", required=True, help="Root directory of phish_sample_30k.")
    parser.add_argument("--benign-root", required=True, help="Root directory of benign_25k.")
    parser.add_argument("--output", required=True, help="Output OCR cache json path.")
    parser.add_argument("--lang", default="en", help="PaddleOCR language, e.g. en, ch, latin.")
    parser.add_argument(
        "--use-angle-cls",
        action="store_true",
        help="Enable PaddleOCR angle classifier when supported.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save partial cache every N processed images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSON if present.",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=6000,
        help="Skip OCR for images whose max(width, height) exceeds this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=16000000,
        help="Skip OCR for images whose width*height exceeds this value. Use 0 to disable.",
    )
    args = parser.parse_args()

    det_csv = Path(args.det_csv).resolve()
    phish_root = Path(args.phish_root).resolve()
    benign_root = Path(args.benign_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not det_csv.is_file():
        raise FileNotFoundError(f"det-csv not found: {det_csv}")
    if not phish_root.is_dir():
        raise FileNotFoundError(f"phish-root not found: {phish_root}")
    if not benign_root.is_dir():
        raise FileNotFoundError(f"benign-root not found: {benign_root}")

    cache = load_existing_cache(output) if args.resume else {}
    if cache:
        print(f"Loaded existing cache with {len(cache)} entries from {output}")

    image_paths = collect_eval_image_paths(det_csv, phish_root, benign_root)
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. First missing: {missing[0]}")

    print(f"Resolved {len(image_paths)} unique eval images from {det_csv}")
    reader = build_paddle_ocr(lang=str(args.lang), use_angle_cls=bool(args.use_angle_cls))

    processed = 0
    skipped_existing = 0
    for idx, path in enumerate(image_paths, start=1):
        path_key = str(path)
        if path_key in cache:
            skipped_existing += 1
            continue

        try:
            image = Image.open(path).convert("RGB")
            skip_by_size, reason = should_skip_by_size(
                image, args.max_image_side, args.max_image_pixels
            )
            if skip_by_size:
                cache[path_key] = ""
                processed += 1
                print(f"[skip:size] {path} ({reason})")
            else:
                print(f"[ocr] {idx}/{len(image_paths)} {path}")
                results = run_ocr(reader, np.array(image), use_angle_cls=bool(args.use_angle_cls))
                text = extract_text_from_result(results)
                cache[path_key] = " ".join(str(text).split())
                processed += 1
        except Exception as exc:
            print(f"[warn] OCR failed for {path}: {exc}")
            cache[path_key] = ""
            processed += 1

        if processed % max(1, args.save_every) == 0:
            save_cache(output, cache)
            print(f"Saved partial cache with {len(cache)} entries to {output}")

        if idx % 20 == 0:
            print(
                f"Scanned {idx}/{len(image_paths)} | newly processed={processed} | resumed={skipped_existing}"
            )

    save_cache(output, cache)
    print(f"Saved OCR cache to {output} with {len(cache)} entries")


if __name__ == "__main__":
    main()
