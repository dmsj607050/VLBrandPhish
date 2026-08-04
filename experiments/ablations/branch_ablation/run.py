"""Branch ablation: image-only vs text-only vs 0.8*img + 0.2*txt fusion.

Dataset: ~2,700 cropped logos (phish + benign, ~100 brands)
    at data/VLBrandPhish/05_logo_crops/crops

Evaluates 3 configurations:
    image_weight=1.0  (image branch only)
    image_weight=0.8  (0.8 img + 0.2 text, default public inference)
    image_weight=0.0  (text branch only)

Metrics: Match Acc, AUROC, Brand Top-1, F1, Precision, Recall
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the shared first-existing resolver for portable defaults
from experiments._shared.e2e import _first_existing

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except Exception:
    average_precision_score = None
    f1_score = None
    roc_auc_score = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.components.brand.encoders import CLIPEncoder
from inference.components.brand.naming import normalize_brand_key
from inference.components.brand.repository import (
    BrandRepository,
    build_brand_repository,
    score_query_features,
)


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SKIP_PREFIXES = ("loginpage", "homepage")


def parse_brand_from_filename(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    if name.startswith("benign_"):
        return "benign"
    m = re.match(r"^(.+)_(\d{4})$", name)
    brand = m.group(1) if m else name
    return brand.replace("_", " ")


def load_crops(crop_dir: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for fn in sorted(os.listdir(crop_dir)):
        if not fn.lower().endswith(".png"):
            continue
        path = os.path.join(crop_dir, fn)
        brand = parse_brand_from_filename(fn)
        items.append((path, brand, fn))
    return items


def pad_to_square(img: Image.Image, pad_color=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    pad = ImageOps.expand(img, ((m - w) // 2, (m - h) // 2, (m - w) // 2, (m - h) // 2), fill=pad_color)
    return pad


def compute_metrics(
    best_scores: np.ndarray,
    pred_brands: List[str],
    gt_brands: List[str],
    threshold: float,
) -> dict:
    n = len(gt_brands)
    if n == 0:
        return {
            "match_acc": 0.0, "n_total": 0, "n_benign": 0, "n_phish": 0,
            "benign_correct": 0, "phish_correct": 0, "benign_acc": 0.0,
            "phish_acc": 0.0, "brand_top1": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "auroc": None, "threshold": threshold,
        }

    benign_correct = 0
    phish_correct = 0
    n_benign = 0
    n_phish = 0
    brand_top1_correct = 0
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    y_score_bin: List[float] = []

    for i in range(n):
        gt = gt_brands[i]
        score = float(best_scores[i])
        pred_brand = pred_brands[i]

        if gt == "benign":
            n_benign += 1
            is_phish = 1 if score >= threshold else 0
            y_true_bin.append(0)
            y_pred_bin.append(is_phish)
            y_score_bin.append(score)
            if score < threshold:
                benign_correct += 1
        else:
            n_phish += 1
            y_true_bin.append(1)
            is_phish = 1 if score >= threshold else 0
            y_pred_bin.append(is_phish)
            y_score_bin.append(score)
            gt_norm = normalize_brand_key(gt)
            pred_norm = normalize_brand_key(pred_brand) if pred_brand else ""
            if pred_norm == gt_norm:
                brand_top1_correct += 1
                if score >= threshold:
                    phish_correct += 1

    total = n_benign + n_phish
    total_correct = benign_correct + phish_correct
    match_acc = float(total_correct) / float(total) if total > 0 else 0.0
    brand_top1 = float(brand_top1_correct) / float(n_phish) if n_phish > 0 else 0.0

    y_true_np = np.array(y_true_bin, dtype=np.uint8)
    y_pred_np = np.array(y_pred_bin, dtype=np.uint8)
    y_score_np = np.array(y_score_bin, dtype=np.float32)

    tp = int(((y_pred_np == 1) & (y_true_np == 1)).sum())
    fp = int(((y_pred_np == 1) & (y_true_np == 0)).sum())
    fn = int(((y_pred_np == 0) & (y_true_np == 1)).sum())
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    auroc = None
    if roc_auc_score is not None and len(np.unique(y_true_np)) >= 2:
        try:
            auroc = float(roc_auc_score(y_true_np, y_score_np))
        except Exception:
            auroc = None

    return {
        "match_acc": match_acc,
        "n_total": int(total),
        "n_benign": int(n_benign),
        "n_phish": int(n_phish),
        "benign_correct": int(benign_correct),
        "phish_correct": int(phish_correct),
        "benign_acc": float(benign_correct) / float(n_benign) if n_benign > 0 else 0.0,
        "phish_acc": float(phish_correct) / float(n_phish) if n_phish > 0 else 0.0,
        "brand_top1": brand_top1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "threshold": float(threshold),
    }


def evaluate_branch(
    crop_items: List[Tuple[str, str, str]],
    reference_dir: str,
    prompt_dir: str,
    normal_urls_csv: str,
    checkpoint: str,
    model_dir: str,
    device: str,
    cache_dir: str,
    image_weight: float,
    threshold: float = 0.62,
    image_batch_size: int = 64,
    text_batch_size: int = 256,
    max_logo_per_brand: int = -1,
    brand_text_mode: str = "dbi5",
) -> dict:
    text_weight = 1.0 - image_weight
    label = (
        "image-only (w=1.0)" if image_weight >= 0.999
        else "text-only (w=0.0)" if image_weight <= 0.001
        else f"fusion (w_img={image_weight:.2f}, w_txt={text_weight:.2f})"
    )
    print("\n" + "=" * 70)
    print(f"Branch ablation: {label}")
    print("=" * 70)

    t0 = time.perf_counter()

    ref_brands = {d for d in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, d))}
    prompt_brands = set()
    for d in os.listdir(prompt_dir):
        dpath = os.path.join(prompt_dir, d)
        if os.path.isdir(dpath) and any(fn.lower().endswith("_logo_description.txt") for fn in os.listdir(dpath)):
            prompt_brands.add(d)
    common_brands = ref_brands & prompt_brands
    print(f"[branch] Reference={len(ref_brands)}, Prompt={len(prompt_brands)}, Common={len(common_brands)}")

    import tempfile
    import subprocess as sp
    temp_ref = tempfile.mkdtemp(prefix="vlp_ref_")
    temp_prompt = tempfile.mkdtemp(prefix="vlp_prompt_")
    for brand in sorted(common_brands):
        for src_root, dst_root in [(reference_dir, temp_ref), (prompt_dir, temp_prompt)]:
            src = os.path.join(src_root, brand)
            dst = os.path.join(dst_root, brand)
            sp.call(["cmd", "/c", "mklink", "/J", dst, src],
                    stdout=sp.DEVNULL, stderr=sp.DEVNULL, shell=True)

    encoder = CLIPEncoder(checkpoint, device=device, model_dir=model_dir)
    print(f"[branch] Encoder loaded (backend={encoder.backend}, dim={encoder.embed_dim})")

    os.makedirs(cache_dir, exist_ok=True)
    from inference.runtime.cache import (
        build_repository_cache_key, load_brand_repository_cache, save_brand_repository_cache,
    )
    repo_key = build_repository_cache_key(
        brand_ckpt=checkpoint, model_dir=model_dir,
        logo_library_dir=temp_ref, dbi5_dir=temp_prompt,
        normal_urls_csv=normal_urls_csv,
        max_logo_per_brand=max_logo_per_brand, brand_text_mode=brand_text_mode,
    )
    repo_cache = os.path.join(cache_dir, f"branch_repo_{repo_key}.pt")
    repo: Optional[BrandRepository] = None
    if os.path.isfile(repo_cache):
        try:
            repo = load_brand_repository_cache(repo_cache, encoder.device)
            print(f"[branch] Repository from cache")
        except Exception:
            repo = None
    if repo is None:
        repo = build_brand_repository(
            logo_library_dir=temp_ref, dbi5_dir=temp_prompt,
            normal_urls_csv=normal_urls_csv, encoder=encoder,
            max_logo_per_brand=max_logo_per_brand,
            image_batch_size=image_batch_size, text_batch_size=text_batch_size,
            brand_text_mode=brand_text_mode,
        )
        try:
            save_brand_repository_cache(repo_cache, repo)
        except Exception:
            pass
    print(f"[branch] Repository: {len(repo.brand_names)} brands")

    crop_paths = [p for p, _, _ in crop_items]
    gt_brands = [gt for _, gt, _ in crop_items]
    crop_feats: List[torch.Tensor] = []
    for start in range(0, len(crop_paths), image_batch_size):
        chunk = crop_paths[start:start + image_batch_size]
        imgs = []
        for p in chunk:
            try:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            except (UnidentifiedImageError, OSError):
                imgs.append(Image.new("RGB", (224, 224)))
        crop_feats.append(encoder.encode_images(imgs, batch_size=len(imgs)))
    crop_features = torch.cat(crop_feats, dim=0)
    print(f"[branch] Crops encoded: {tuple(crop_features.shape)}")

    sim_img, sim_txt = score_query_features(crop_features, repo)
    scores = image_weight * sim_img + text_weight * sim_txt

    best_scores, best_indices = scores.max(dim=1)
    best_scores_np = best_scores.cpu().numpy()
    repo_brand_norms = [normalize_brand_key(b) for b in repo.brand_names]
    pred_brands = [
        repo.brand_names[int(idx)] if s >= threshold else ""
        for s, idx in zip(best_scores_np, best_indices.cpu().numpy())
    ]

    result = compute_metrics(best_scores_np, pred_brands, gt_brands, threshold)
    print(f"[branch] Match Acc:  {result['match_acc']:.4f} "
          f"(benign={result['benign_acc']:.4f}, phish={result['phish_acc']:.4f})")
    print(f"[branch] Brand Top1: {result['brand_top1']:.4f}  "
          f"Precision: {result['precision']:.4f}  Recall: {result['recall']:.4f}  F1: {result['f1']:.4f}")
    print(f"[branch] AUROC:      {result['auroc']:.4f}" if result['auroc'] is not None else "[branch] AUROC:      N/A")
    print(f"[branch] Time: {time.perf_counter() - t0:.1f}s")
    return {
        "image_weight": float(image_weight),
        "text_weight": float(text_weight),
        **result,
    }


def save_results(results: List[dict], out_dir: str, crop_stats: dict) -> None:
    summary = {
        "experiment": "branch_ablation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": crop_stats,
        "results": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "image_weight", "text_weight", "threshold",
        "match_acc", "brand_top1", "precision", "recall", "f1", "auroc",
        "benign_acc", "phish_acc",
        "n_total", "n_benign", "n_phish",
        "benign_correct", "phish_correct",
    ]
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                **{k: f"{r[k]:.6f}" if isinstance(r[k], float) and k != "n_total"
                               and k != "n_benign" and k != "n_phish"
                               and k != "benign_correct" and k != "phish_correct"
                    else r[k]
                    for k in fieldnames if k in r}
            })

    # LaTeX table style: image_weight, match_acc, brand_top1, f1, auroc
    cols_md = ["image_weight", "text_weight", "match_acc", "brand_top1", "f1", "auroc"]
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols_md) + " |\n")
        f.write("| " + " | ".join(["---" for _ in cols_md]) + " |\n")
        for r in results:
            def _fmt(v):
                if v is None:
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("| " + " | ".join(_fmt(r.get(c, "")) for c in cols_md) + " |\n")


def main(argv: list[str] | None = None) -> None:
    default_data_root = str(REPO_ROOT / "data" / "VLBrandPhish")
    default_ckpt = _first_existing(
        os.environ.get("VLBRANDPHISH_BRAND_CKPT", ""),
        REPO_ROOT / "models" / "vlbrandphish" / "brand_encoder_inference.pt",
        REPO_ROOT / "models" / "vlbrandphish" / "brand_identify.pt",
    )
    default_siglip = _first_existing(
        os.environ.get("VLBRANDPHISH_SIGLIP_MODEL_DIR", ""),
        REPO_ROOT / "models" / "siglip-base-patch16-224",
    )

    parser = argparse.ArgumentParser(description="Branch ablation: image-only / fusion / text-only")
    parser.add_argument("--crop-dir", default=os.path.join(default_data_root, "05_logo_crops", "crops"))
    parser.add_argument("--reference-dir", default=os.path.join(default_data_root, "03_logo_reference_library", "logo_reference_library"))
    parser.add_argument("--prompt-dir", default=os.path.join(default_data_root, "04_prompt_libraries_152", "visual_all"))
    parser.add_argument("--normal-urls-csv", default=os.path.join(default_data_root, "07_normal_urls_testb.csv"))
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--model-dir", default=default_siglip)
    parser.add_argument("--image-weights", default="1.0,0.8,0.0", help="Comma-separated image weights")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--max-logo-per-brand", type=int, default=-1)
    parser.add_argument("--brand-text-mode", default="dbi5")
    args = parser.parse_args(argv)

    device = args.device if torch.cuda.is_available() else "cpu"

    for label, path in [("crop-dir", args.crop_dir), ("reference-dir", args.reference_dir),
                         ("checkpoint", args.checkpoint)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__) or ".", "results", f"branch_ablation_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[branch_ablation] device={device}")
    print(f"[branch_ablation] crop_dir={args.crop_dir}")
    print(f"[branch_ablation] reference_dir={args.reference_dir}")
    print(f"[branch_ablation] prompt_dir={args.prompt_dir}")
    print(f"[branch_ablation] out_dir={out_dir}")

    crop_items = load_crops(args.crop_dir)
    n_benign = sum(1 for _, gt, _ in crop_items if gt == "benign")
    n_phish = len(crop_items) - n_benign
    unique_brands = sorted(set(gt for _, gt, _ in crop_items if gt != "benign"))
    print(f"[branch_ablation] Crops: {len(crop_items)} (benign={n_benign}, phish={n_phish}, brands={len(unique_brands)})")
    crop_stats = {
        "crop_dir": args.crop_dir,
        "n_total": len(crop_items),
        "n_benign": n_benign,
        "n_phish": n_phish,
        "n_unique_brands": len(unique_brands),
    }

    try:
        weights_raw = [float(x.strip()) for x in str(args.image_weights).split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Invalid --image-weights: {args.image_weights!r} ({e})")

    results: List[dict] = []
    for w in weights_raw:
        if w < 0.0 or w > 1.0:
            print(f"[branch_ablation] skip invalid image_weight={w} (must be in [0, 1])")
            continue
        try:
            r = evaluate_branch(
                crop_items, args.reference_dir, args.prompt_dir, args.normal_urls_csv,
                args.checkpoint, args.model_dir, device, cache_dir,
                image_weight=w, threshold=args.threshold,
                image_batch_size=args.image_batch_size,
                text_batch_size=args.text_batch_size,
                max_logo_per_brand=args.max_logo_per_brand,
                brand_text_mode=args.brand_text_mode,
            )
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[branch_ablation] weight={w} FAILED: {e}")
            results.append({"image_weight": float(w), "text_weight": 1.0 - float(w), "error": str(e)})

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY — Branch ablation (image_weight sweep)")
    print("=" * 80)
    header = f"{'w_img':>6} {'w_txt':>6} {'MatchAcc':>10} {'BrandTop1':>10} {'P':>7} {'R':>7} {'F1':>7} {'AUROC':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"  w={r['image_weight']:.2f}  ERROR: {r['error']}")
            continue
        auroc = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "  N/A   "
        print(f"{r['image_weight']:6.2f} {r['text_weight']:6.2f} "
              f"{r['match_acc']:10.4f} {r['brand_top1']:10.4f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f} {auroc:>8}")
    print("-" * 80)
    print(f"Total: {len(crop_items)} crops (benign={n_benign}, phish={n_phish})")

    save_results(results, out_dir, crop_stats)
    print(f"\n[branch_ablation] Results saved to {out_dir}/")
    print(f"  summary.json  results.csv  results.md")


if __name__ == "__main__":
    main()
