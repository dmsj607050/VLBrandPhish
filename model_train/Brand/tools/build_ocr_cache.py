from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from PIL import Image


def iter_image_paths(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            yield path


def build_paddle_ocr(lang: str, use_angle_cls: bool):
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        print(f"[debug] importing paddleocr failed: {type(exc).__name__}: {exc}")
        raise


    ctor_attempts = [
        dict(
            lang=str(lang),
            device="cpu",
            use_textline_orientation=bool(use_angle_cls),
            show_log=False,
        ),
        dict(
            lang=str(lang),
            device="cpu",
            use_angle_cls=bool(use_angle_cls),
            show_log=False,
        ),
        dict(
            lang=str(lang),
            device="cpu",
        ),
        dict(
            lang=str(lang),
            use_gpu=False,
            use_textline_orientation=bool(use_angle_cls),
            show_log=False,
        ),
        dict(
            lang=str(lang),
            use_gpu=False,
            use_angle_cls=bool(use_angle_cls),
            show_log=False,
        ),
        dict(
            lang=str(lang),
            use_gpu=False,
        ),
    ]
    last_error = None
    for kwargs in ctor_attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to initialize PaddleOCR with compatible arguments: {last_error}")


def run_ocr(reader, image_np: np.ndarray, use_angle_cls: bool):
    call_attempts = [
        dict(cls=bool(use_angle_cls)),
        dict(det=True, rec=True, cls=bool(use_angle_cls)),
        {},
    ]
    last_error = None
    for kwargs in call_attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return reader.ocr(image_np, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to run PaddleOCR with compatible call arguments: {last_error}")


def extract_text_from_result(results) -> str:
    lines = []
    for block in results or []:
        if not block:
            continue
        if isinstance(block, dict):
            rec_texts = block.get("rec_texts") or []
            for text_part in rec_texts:
                text_part = str(text_part).strip()
                if text_part:
                    lines.append(text_part)
            continue
        for item in block or []:
            if not item:
                continue
            if isinstance(item, dict):
                text_part = item.get("text") or item.get("rec_text") or ""
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text_part = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            else:
                text_part = ""
            text_part = str(text_part).strip()
            if text_part:
                lines.append(text_part)
    return " ".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR cache JSON for TriSig training with PaddleOCR.")
    parser.add_argument("--data-root", required=True, help="Dataset root that contains brand folders.")
    parser.add_argument("--output", required=True, help="Output OCR cache json path.")
    parser.add_argument("--lang", default="en", help="PaddleOCR language, e.g. en, ch, latin.")
    parser.add_argument("--use-angle-cls", action="store_true", help="Enable PaddleOCR angle classifier when supported.")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    reader = build_paddle_ocr(lang=str(args.lang), use_angle_cls=bool(args.use_angle_cls))
    cache = {}
    paths = list(iter_image_paths(root))
    print(f"Found {len(paths)} images under {root}")

    for idx, path in enumerate(paths, start=1):
        try:
            image = Image.open(path).convert("RGB")
            results = run_ocr(reader, np.array(image), use_angle_cls=bool(args.use_angle_cls))
            text = extract_text_from_result(results)
        except Exception as exc:
            print(f"[warn] OCR failed for {path}: {exc}")
            text = ""
        cache[str(path)] = " ".join(text.split())
        if idx % 200 == 0:
            print(f"Processed {idx}/{len(paths)}")

    with output.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Saved OCR cache to {output}")


if __name__ == "__main__":
    main()
