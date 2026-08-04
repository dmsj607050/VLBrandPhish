"""Brand identification on the unified cropped-logo set.

Research question: RQ2 — Brand Identification
Dataset: Cropped-logo eval set (~2,700 crops: phish + benign, ~100 brands)
Metric: Match Acc (benign → benign, phish → correct target brand)

Methods:
  - VLBrandPhish:    SigLIP encoder, 0.8*img + 0.2*txt fusion, threshold 0.62
  - PhishIntention:  BiT-M-R50x1 + OCR(ASTER) siamese, threshold 0.87
  - PhishPedia:      BiT-M-R50x1 siamese (image-only), threshold 0.87
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the shared first-existing resolver for portable defaults
from experiments._shared.e2e import _first_existing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.components.brand.encoders import CLIPEncoder
from inference.components.brand.naming import normalize_brand_key
from inference.components.brand.repository import (
    BrandRepository,
    build_brand_repository,
    score_query_features,
)

# ── Method source paths ──────────────────────────────────────────────
PPD_WEIGHTS = _first_existing(
    os.environ.get("VLBRANDPHISH_PP_WEIGHTS", ""),
    REPO_ROOT.parent / "methods" / "02_phishpedia" / "Phishpedia-main" / "models" / "resnetv2_rgb_new.pth.tar",
)
PI_SIAMESS_WEIGHTS = _first_existing(
    os.environ.get("VLBRANDPHISH_PI_SIAMESE_WEIGHTS", ""),
    REPO_ROOT.parent / "methods" / "01_PhishIntention" / "PhishIntention-main" / "models" / "ocr_siamese.pth.tar",
)
PI_OCR_WEIGHTS = _first_existing(
    os.environ.get("VLBRANDPHISH_PI_OCR_WEIGHTS", ""),
    REPO_ROOT.parent / "methods" / "01_PhishIntention" / "PhishIntention-main" / "models" / "ocr_pretrained.pth.tar",
)
# Source directories for baseline method imports
PI_SRC = REPO_ROOT.parent / "methods" / "01_PhishIntention" / "PhishIntention-main" / "src"
PPD_ROOT = REPO_ROOT.parent / "methods" / "02_phishpedia" / "Phishpedia-main"

SKIP_PREFIXES = ("loginpage", "homepage")
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ═══════════════════════════════════════════════════════════════════════
#  Common utilities
# ═══════════════════════════════════════════════════════════════════════

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


def load_reference_logos(reference_dir: str, allowed_brands: Optional[set] = None) -> Tuple[List[str], List[str]]:
    """Return (logo_paths, brand_names) for all reference logos.

    Looks for images in ``brand/logos/`` (and falls back to ``brand/``).
    """
    logo_paths: List[str] = []
    brand_names: List[str] = []
    for brand_dir_name in sorted(os.listdir(reference_dir)):
        bd = os.path.join(reference_dir, brand_dir_name)
        if not os.path.isdir(bd):
            continue
        if allowed_brands is not None and brand_dir_name not in allowed_brands:
            continue
        logos_sub = os.path.join(bd, "logos")
        search_dir = logos_sub if os.path.isdir(logos_sub) else bd
        if not os.path.isdir(search_dir):
            continue
        for fn in sorted(os.listdir(search_dir)):
            if not fn.lower().endswith(VALID_EXTS):
                continue
            if any(fn.lower().startswith(p) for p in SKIP_PREFIXES):
                continue
            logo_paths.append(os.path.join(search_dir, fn))
            brand_names.append(brand_dir_name)
    return logo_paths, brand_names


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def pad_to_square(img: Image.Image, pad_color=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    pad = ImageOps.expand(img, ((m - w) // 2, (m - h) // 2, (m - w) // 2, (m - h) // 2), fill=pad_color)
    return pad


def compute_match_accuracy_generic(
    best_scores: np.ndarray,
    pred_brands: List[str],
    gt_brands: List[str],
    threshold: float,
) -> dict:
    n = len(gt_brands)
    if n == 0:
        return {"match_acc": 0.0, "n_total": 0, "n_benign": 0, "n_phish": 0,
                "benign_correct": 0, "phish_correct": 0, "benign_acc": 0.0,
                "phish_acc": 0.0, "threshold": threshold}

    benign_correct = 0
    phish_correct = 0
    n_benign = 0
    n_phish = 0

    for i in range(n):
        gt = gt_brands[i]
        if gt == "benign":
            n_benign += 1
            if best_scores[i] < threshold:
                benign_correct += 1
        else:
            n_phish += 1
            gt_norm = normalize_brand_key(gt)
            pred_norm = normalize_brand_key(pred_brands[i]) if pred_brands[i] else ""
            if best_scores[i] >= threshold and pred_norm == gt_norm:
                phish_correct += 1

    total = n_benign + n_phish
    total_correct = benign_correct + phish_correct
    return {
        "match_acc": float(total_correct) / float(total) if total > 0 else 0.0,
        "n_total": int(total),
        "n_benign": int(n_benign),
        "n_phish": int(n_phish),
        "benign_correct": int(benign_correct),
        "phish_correct": int(phish_correct),
        "benign_acc": float(benign_correct) / float(n_benign) if n_benign > 0 else 0.0,
        "phish_acc": float(phish_correct) / float(n_phish) if n_phish > 0 else 0.0,
        "threshold": float(threshold),
    }


# ═══════════════════════════════════════════════════════════════════════
#  VLBrandPhish evaluator (SigLIP, 0.8*img + 0.2*txt, threshold 0.62)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_vlbrandphish(
    crop_items: List[Tuple[str, str, str]],
    reference_dir: str,
    prompt_dir: str,
    normal_urls_csv: str,
    checkpoint: str,
    model_dir: str,
    device: str,
    cache_dir: str,
    image_batch_size: int = 64,
    text_batch_size: int = 256,
    max_logo_per_brand: int = -1,
    brand_text_mode: str = "dbi5",
    threshold: float = 0.62,
) -> dict:
    print("\n" + "=" * 60)
    print("VLBrandPhish (SigLIP, 0.8*img+0.2*txt, threshold=0.62)")
    print("=" * 60)

    t0 = time.perf_counter()

    # Filter brands: intersection of reference library and prompt library
    ref_brands = {d for d in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, d))}
    prompt_brands = set()
    for d in os.listdir(prompt_dir):
        dpath = os.path.join(prompt_dir, d)
        if os.path.isdir(dpath) and any(fn.lower().endswith("_logo_description.txt") for fn in os.listdir(dpath)):
            prompt_brands.add(d)
    common_brands = ref_brands & prompt_brands
    print(f"[VLP] Reference={len(ref_brands)}, Prompt={len(prompt_brands)}, Common={len(common_brands)}")

    # Create junction-linked filtered dirs
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

    # Load encoder
    encoder = CLIPEncoder(checkpoint, device=device, model_dir=model_dir)
    print(f"[VLP] Encoder loaded (backend={encoder.backend}, dim={encoder.embed_dim})")

    # Build repository
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
    repo_cache = os.path.join(cache_dir, f"vlp_repo_{repo_key}.pt")
    repo = None
    if os.path.isfile(repo_cache):
        try:
            repo = load_brand_repository_cache(repo_cache, encoder.device)
            print(f"[VLP] Repository from cache")
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
    print(f"[VLP] Repository: {len(repo.brand_names)} brands")

    # Encode crops
    crop_paths = [p for p, _, _ in crop_items]
    gt_brands = [gt for _, gt, _ in crop_items]
    crop_feats = []
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
    print(f"[VLP] Crops encoded: {tuple(crop_features.shape)}")

    # Compute similarities
    sim_img, sim_txt = score_query_features(crop_features, repo)
    scores = 0.8 * sim_img + 0.2 * sim_txt

    # Evaluate
    best_scores, best_indices = scores.max(dim=1)
    best_scores_np = best_scores.cpu().numpy()
    repo_brand_norms = [normalize_brand_key(b) for b in repo.brand_names]
    pred_brands = [repo.brand_names[int(idx)] if s >= threshold else "" for s, idx in zip(best_scores_np, best_indices.cpu().numpy())]

    result = compute_match_accuracy_generic(best_scores_np, pred_brands, gt_brands, threshold)
    print(f"[VLP] Match Acc: {result['match_acc']:.4f} "
          f"(benign={result['benign_acc']:.4f}, phish={result['phish_acc']:.4f})")
    print(f"[VLP] Time: {time.perf_counter() - t0:.1f}s")
    return {"method": "vlbrandphish", "threshold": threshold, **result}


# ═══════════════════════════════════════════════════════════════════════
#  PhishPedia evaluator (BiT-M-R50x1, image-only, threshold 0.87)
# ═══════════════════════════════════════════════════════════════════════

def _load_ppd_model(weights_path: str, device: str):
    """Load PhishPedia's BiT-M-R50x1 siamese model via importlib."""
    spec = importlib.util.spec_from_file_location("ppd_models", str(PPD_ROOT / "models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ResNetV2 = mod.KNOWN_MODELS["BiT-M-R50x1"]
    model = ResNetV2(head_size=277, zero_head=True)
    weights = torch.load(weights_path, map_location="cpu")
    weights = weights.get("model", weights)
    new_sd = OrderedDict()
    for k, v in weights.items():
        new_sd[k.split("module.")[-1] if "module." in k else k] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()
    return model


def _encode_bit_images(
    model, image_paths: List[str], device: str,
    img_size: int, batch_size: int = 64,
    ocr_model=None,
) -> torch.Tensor:
    """Encode images with a BiT model (optionally OCR-aided).

    Returns L2-normalized features [N, D].
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    all_feats: List[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start + batch_size]
        imgs_tensor = []
        ocr_embs = []
        for p in chunk:
            try:
                img = Image.open(p).convert("RGB")
            except (UnidentifiedImageError, OSError):
                img = Image.new("RGB", (img_size, img_size))
            img = pad_to_square(img)
            img = img.resize((img_size, img_size))

            if ocr_model is not None:
                # OCR embedding: resize to 32×100 for OCR encoder
                ocr_img = img.resize((100, 32))
                ocr_t = transforms.ToTensor()(ocr_img)
                ocr_t = transforms.Normalize(mean, std)(ocr_t)
                with torch.no_grad():
                    # Call encoder directly to avoid decoder device issues
                    enc_feat = ocr_model.encoder(ocr_t.unsqueeze(0).to(device))
                    enc_feat = enc_feat.contiguous()
                    ocr_emb = enc_feat.mean(dim=1).detach()  # [1, 512]
                ocr_embs.append(ocr_emb.squeeze(0))

            imgs_tensor.append(tfm(img))

        batch = torch.stack(imgs_tensor).to(device)
        with torch.no_grad():
            if ocr_model is not None:
                ocr_batch = torch.stack(ocr_embs).to(device)
                feats = model.features(batch, ocr_batch)
            else:
                feats = model.features(batch)
            feats = l2_norm(feats)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0) if all_feats else torch.empty((0, 2048))


def evaluate_phishpedia(
    crop_items: List[Tuple[str, str, str]],
    reference_dir: str,
    weights_path: str,
    device: str,
    threshold: float = 0.87,
    batch_size: int = 64,
) -> dict:
    print("\n" + "=" * 60)
    print(f"PhishPedia (BiT-M-R50x1, image-only, threshold={threshold})")
    print("=" * 60)

    t0 = time.perf_counter()
    model = _load_ppd_model(weights_path, device)
    print(f"[PPD] Model loaded")

    # Load and encode reference logos (all brands in reference library)
    ref_paths, ref_brands = load_reference_logos(reference_dir)
    print(f"[PPD] Reference logos: {len(ref_paths)} from {len(set(ref_brands))} brands")
    ref_feats = _encode_bit_images(model, ref_paths, device, img_size=128, batch_size=batch_size)
    print(f"[PPD] Reference encoded: {tuple(ref_feats.shape)}")

    # Encode crops
    crop_paths = [p for p, _, _ in crop_items]
    gt_brands = [gt for _, gt, _ in crop_items]
    crop_feats = _encode_bit_images(model, crop_paths, device, img_size=128, batch_size=batch_size)
    print(f"[PPD] Crops encoded: {tuple(crop_feats.shape)}")

    # Cosine similarity (both are L2-normalized)
    sim_matrix = crop_feats @ ref_feats.T  # [N_crops, N_refs]
    best_scores, best_indices = sim_matrix.max(dim=1)
    best_scores_np = best_scores.numpy()
    best_indices_np = best_indices.numpy()
    pred_brands = [ref_brands[int(idx)] if s >= threshold else "" for s, idx in zip(best_scores_np, best_indices_np)]

    result = compute_match_accuracy_generic(best_scores_np, pred_brands, gt_brands, threshold)
    print(f"[PPD] Match Acc: {result['match_acc']:.4f} "
          f"(benign={result['benign_acc']:.4f}, phish={result['phish_acc']:.4f})")
    print(f"[PPD] Time: {time.perf_counter() - t0:.1f}s")
    return {"method": "phishpedia", "threshold": threshold, **result}


# ═══════════════════════════════════════════════════════════════════════
#  PhishIntention evaluator (BiT-M-R50x1 + OCR, threshold 0.87)
# ═══════════════════════════════════════════════════════════════════════

def _load_pi_models(siamese_path: str, ocr_path: str, device: str):
    """Load PhishIntention's BiT+OCR siamese model and OCR model."""
    pi_src = str(PI_SRC)
    if pi_src not in sys.path:
        sys.path.insert(0, pi_src)

    from phishintention.networks.siamese_models import KNOWN_MODELS as PI_MODELS
    from phishintention.networks.ocr.models.model_builder import ModelBuilder
    from phishintention.networks.ocr.utils.labelmaps import get_vocabulary

    # Siamese model (BiT-M-R50x1 with OCR fusion head)
    model = PI_MODELS["BiT-M-R50x1"](head_size=277, zero_head=True)
    weights = torch.load(siamese_path, map_location="cpu")
    weights = weights.get("model", weights)
    new_sd = OrderedDict()
    for k, v in weights.items():
        new_sd[k.split("module.")[-1] if "module." in k else k] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()

    # OCR model (ResNet_ASTER with STN)
    voc = get_vocabulary("ALLCASES_SYMBOLS", EOS="EOS", PADDING="PADDING", UNKNOWN="UNKNOWN")
    char2id = dict(zip(voc, range(len(voc))))
    ocr_model = ModelBuilder(
        arch="ResNet_ASTER",
        rec_num_classes=len(voc),
        sDim=512, attDim=512, max_len_labels=100,
        eos=char2id["EOS"], STN_ON=True,
    )
    ocr_weights = torch.load(ocr_path, map_location="cpu")
    ocr_model.load_state_dict(ocr_weights["state_dict"])
    ocr_model.to(device)
    ocr_model.eval()

    return model, ocr_model


def evaluate_phishintention(
    crop_items: List[Tuple[str, str, str]],
    reference_dir: str,
    siamese_path: str,
    ocr_path: str,
    device: str,
    threshold: float = 0.87,
    batch_size: int = 32,
) -> dict:
    print("\n" + "=" * 60)
    print(f"PhishIntention (BiT-M-R50x1+OCR, threshold={threshold})")
    print("=" * 60)

    t0 = time.perf_counter()
    model, ocr_model = _load_pi_models(siamese_path, ocr_path, device)
    print(f"[PI] Models loaded (siamese + OCR)")

    # Load and encode reference logos
    ref_paths, ref_brands = load_reference_logos(reference_dir)
    print(f"[PI] Reference logos: {len(ref_paths)} from {len(set(ref_brands))} brands")
    ref_feats = _encode_bit_images(model, ref_paths, device, img_size=224, batch_size=batch_size, ocr_model=ocr_model)
    print(f"[PI] Reference encoded: {tuple(ref_feats.shape)}")

    # Encode crops
    crop_paths = [p for p, _, _ in crop_items]
    gt_brands = [gt for _, gt, _ in crop_items]
    crop_feats = _encode_bit_images(model, crop_paths, device, img_size=224, batch_size=batch_size, ocr_model=ocr_model)
    print(f"[PI] Crops encoded: {tuple(crop_feats.shape)}")

    # Cosine similarity
    sim_matrix = crop_feats @ ref_feats.T
    best_scores, best_indices = sim_matrix.max(dim=1)
    best_scores_np = best_scores.numpy()
    best_indices_np = best_indices.numpy()
    pred_brands = [ref_brands[int(idx)] if s >= threshold else "" for s, idx in zip(best_scores_np, best_indices_np)]

    result = compute_match_accuracy_generic(best_scores_np, pred_brands, gt_brands, threshold)
    print(f"[PI] Match Acc: {result['match_acc']:.4f} "
          f"(benign={result['benign_acc']:.4f}, phish={result['phish_acc']:.4f})")
    print(f"[PI] Time: {time.perf_counter() - t0:.1f}s")
    return {"method": "phishintention", "threshold": threshold, **result}


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    default_dataset = _first_existing(
        os.environ.get("VLBRANDPHISH_DATASET_ROOT", ""),
        REPO_ROOT / "data" / "VLBrandPhish",
    )
    default_siglip = _first_existing(
        os.environ.get("VLBRANDPHISH_SIGLIP_MODEL_DIR", ""),
        REPO_ROOT / "models" / "vlbrandphish" / "siglip-base-patch16-224",
    )
    default_ckpt = _first_existing(
        os.environ.get("VLBRANDPHISH_BRAND_CKPT", ""),
        REPO_ROOT / "models" / "vlbrandphish" / "brand_encoder_inference.pt",
        REPO_ROOT / "models" / "vlbrandphish" / "brand_identify.pt",
    )

    parser = argparse.ArgumentParser(description="Brand identification: 3-method comparison")
    parser.add_argument("--crop-dir", default=os.path.join(default_dataset, "05_logo_crops", "crops"))
    parser.add_argument("--reference-dir", default=os.path.join(default_dataset, "03_logo_reference_library", "logo_reference_library"))
    parser.add_argument("--prompt-dir", default=os.path.join(default_dataset, "04_prompt_libraries_152", "visual_all"))
    parser.add_argument("--normal-urls-csv", default=os.path.join(default_dataset, "07_normal_urls_testb.csv"))
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--model-dir", default=default_siglip)
    parser.add_argument("--ppd-weights", default=str(PPD_WEIGHTS))
    parser.add_argument("--pi-siamese-weights", default=str(PI_SIAMESS_WEIGHTS))
    parser.add_argument("--pi-ocr-weights", default=str(PI_OCR_WEIGHTS))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--pi-batch-size", type=int, default=32, help="Batch size for PhishIntention (smaller due to OCR)")
    parser.add_argument("--max-logo-per-brand", type=int, default=-1)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--brand-text-mode", default="dbi5")
    parser.add_argument("--vlp-threshold", type=float, default=0.62)
    parser.add_argument("--ppd-threshold", type=float, default=0.87)
    parser.add_argument("--pi-threshold", type=float, default=0.87)
    args = parser.parse_args(argv)

    device = args.device if torch.cuda.is_available() else "cpu"

    # Validate paths
    for label, path in [("crop-dir", args.crop_dir), ("reference-dir", args.reference_dir),
                         ("checkpoint", args.checkpoint), ("ppd-weights", args.ppd_weights),
                         ("pi-siamese-weights", args.pi_siamese_weights),
                         ("pi-ocr-weights", args.pi_ocr_weights)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Output directory
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__) or ".", "results", f"brand_id_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[brand_id] device={device}")
    print(f"[brand_id] crop_dir={args.crop_dir}")
    print(f"[brand_id] reference_dir={args.reference_dir}")
    print(f"[brand_id] out_dir={out_dir}")

    # Load crops
    crop_items = load_crops(args.crop_dir)
    n_benign = sum(1 for _, gt, _ in crop_items if gt == "benign")
    n_phish = len(crop_items) - n_benign
    unique_brands = sorted(set(gt for _, gt, _ in crop_items if gt != "benign"))
    print(f"[brand_id] Crops: {len(crop_items)} (benign={n_benign}, phish={n_phish}, brands={len(unique_brands)})")

    results = []

    # 1. VLBrandPhish
    try:
        r = evaluate_vlbrandphish(
            crop_items, args.reference_dir, args.prompt_dir, args.normal_urls_csv,
            args.checkpoint, args.model_dir, device, cache_dir,
            image_batch_size=args.image_batch_size,
            max_logo_per_brand=args.max_logo_per_brand,
            brand_text_mode=args.brand_text_mode,
            threshold=args.vlp_threshold,
        )
        results.append(r)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[VLP] FAILED: {e}")
        results.append({"method": "vlbrandphish", "error": str(e)})

    # 2. PhishPedia
    try:
        r = evaluate_phishpedia(
            crop_items, args.reference_dir, args.ppd_weights, device,
            threshold=args.ppd_threshold, batch_size=args.image_batch_size,
        )
        results.append(r)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PPD] FAILED: {e}")
        results.append({"method": "phishpedia", "error": str(e)})

    # 3. PhishIntention
    try:
        r = evaluate_phishintention(
            crop_items, args.reference_dir,
            args.pi_siamese_weights, args.pi_ocr_weights, device,
            threshold=args.pi_threshold, batch_size=args.pi_batch_size,
        )
        results.append(r)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PI] FAILED: {e}")
        results.append({"method": "phishintention", "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: Brand Identification — Match Accuracy")
    print("=" * 60)
    print(f"{'Method':<20} {'Threshold':<12} {'Match Acc':<12} {'Benign Acc':<12} {'Phish Acc':<12}")
    print("-" * 68)
    for r in results:
        if "error" in r:
            print(f"{r['method']:<20} ERROR: {r['error']}")
        else:
            print(f"{r['method']:<20} {r['threshold']:<12.2f} {r['match_acc']:<12.4f} {r['benign_acc']:<12.4f} {r['phish_acc']:<12.4f}")
    print("-" * 68)
    print(f"Total: {len(crop_items)} crops (benign={n_benign}, phish={n_phish})")

    # Save results
    summary = {
        "experiment": "brand_identification",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "crop_dir": args.crop_dir,
            "n_total": len(crop_items),
            "n_benign": n_benign,
            "n_phish": n_phish,
            "n_unique_brands": len(unique_brands),
        },
        "thresholds": {
            "vlbrandphish": args.vlp_threshold,
            "phishpedia": args.ppd_threshold,
            "phishintention": args.pi_threshold,
        },
        "results": results,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "threshold", "match_acc", "benign_acc", "phish_acc",
            "n_total", "n_benign", "n_phish", "benign_correct", "phish_correct",
        ])
        writer.writeheader()
        for r in results:
            if "error" not in r:
                writer.writerow({
                    "method": r["method"],
                    "threshold": f"{r['threshold']:.2f}",
                    "match_acc": f"{r['match_acc']:.6f}",
                    "benign_acc": f"{r['benign_acc']:.6f}",
                    "phish_acc": f"{r['phish_acc']:.6f}",
                    "n_total": r["n_total"],
                    "n_benign": r["n_benign"],
                    "n_phish": r["n_phish"],
                    "benign_correct": r["benign_correct"],
                    "phish_correct": r["phish_correct"],
                })

    print(f"\n[brand_id] Results saved to:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
