#!/usr/bin/env python
"""VLBrandPhish Reproducibility Environment Smoke Test

Checks the four prerequisites required to run the project for paper reviewers and readers:
  1. Python environment and dependencies
  2. Dataset integrity (directories and file counts)
  3. Model artifact availability
  4. Minimal inference pipeline (two-image end-to-end validation)

Usage:
    python tests/test_smoke.py
    python tests/test_smoke.py --dataset-root /path/to/datasets/02_VLBrandPhish_Official
    python tests/test_smoke.py --skip-inference          # Check environment, data, and models only
    python tests/test_smoke.py --device cpu              # Run inference on CPU
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Default path priority: environment variable > repository convention > workspace parent
# ═══════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def _first_existing(*candidates: str | Path) -> str:
    """Return the first existing path, or the first valid candidate as a hint."""
    cleaned = [Path(item) for item in candidates if str(item or "").strip()]
    for c in cleaned:
        if c.exists():
            return str(c)
    return str(cleaned[0]) if cleaned else ""


DEFAULT_DATASET_ROOT = _first_existing(
    os.environ.get("VLBRANDPHISH_DATASET_ROOT", ""),
    REPO_ROOT / "data" / "VLBrandPhish",
    WORKSPACE_ROOT / "datasets" / "02_VLBrandPhish_Official",
)

DEFAULT_BRAND_CKPT = _first_existing(
    os.environ.get("VLBRANDPHISH_BRAND_CKPT", ""),
    REPO_ROOT / "models" / "vlbrandphish" / "brand_encoder_inference.pt",
    REPO_ROOT / "models" / "vlbrandphish" / "brand_identify.pt",
)

DEFAULT_SIGLIP_DIR = _first_existing(
    os.environ.get("VLBRANDPHISH_SIGLIP_MODEL_DIR", ""),
    REPO_ROOT / "models" / "vlbrandphish" / "siglip-base-patch16-224",
    REPO_ROOT / "models" / "siglip-base-patch16-224",
)

# Minimum Python version
MIN_PYTHON = (3, 10)

# Required packages: import name -> pip package name
REQUIRED_PACKAGES = {
    "numpy":        "numpy",
    "PIL":          "Pillow",
    "torch":        "torch",
    "transformers": "transformers",
    "tldextract":   "tldextract",
    "sentencepiece":"sentencepiece",
    "yaml":         "pyyaml",
    "tqdm":         "tqdm",
}

# Dataset subdirectories relative to dataset_root
DATASET_CHECKS = {
    "benign":           "02_bengin_12K",
    "phish":            "01_phish_12K",
    "logo_library":     "03_logo_reference_library/logo_reference_library",
    "prompt_library":   "04_prompt_libraries_152/visual_all",
    "normal_urls":      "07_normal_urls_testb.csv",
}


# ═══════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def _header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")

def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}    {msg}")

def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET}  {msg}")

def _fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET}  {msg}")


# ═══════════════════════════════════════════════════════════════════════
# 1. Environment check
# ═══════════════════════════════════════════════════════════════════════

def check_environment() -> bool:
    """Check the Python version and required dependencies."""
    _header("1. Python environment")

    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver >= MIN_PYTHON:
        _ok(f"Python {ver_str}  (>= {'.'.join(map(str, MIN_PYTHON))})")
    else:
        _fail(f"Python {ver_str}  (requires >= {'.'.join(map(str, MIN_PYTHON))})")
        return False

    all_ok = True
    for import_name, pkg_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            _ok(f"{pkg_name}")
        except ImportError:
            _fail(f"{pkg_name} is not installed  ->  pip install {pkg_name}")
            all_ok = False

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            _ok(f"CUDA available (devices: {n}, {name})")
        else:
            _warn("CUDA unavailable; inference will use CPU (slower)")
    except Exception:
        pass

    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# 2. Dataset check
# ═══════════════════════════════════════════════════════════════════════

def check_datasets(dataset_root: Path) -> Dict[str, Optional[dict]]:
    """Scan dataset directories and count files and subdirectories."""
    _header("2. Dataset check")

    info: Dict[str, Optional[dict]] = {}

    if not dataset_root.exists():
        _fail(f"Dataset root does not exist: {dataset_root}")
        return info

    for name, rel in DATASET_CHECKS.items():
        path = dataset_root / rel
        if not path.exists():
            _fail(f"{name}: path does not exist ({path})")
            info[name] = None
            continue

        if path.is_dir():
            entries = list(path.iterdir())
            subdirs = [e for e in entries if e.is_dir()]
            files   = [e for e in entries if e.is_file()]
            info[name] = {"path": str(path), "dirs": len(subdirs), "files": len(files)}
            _ok(f"{name}: {len(subdirs)}  subdirectories, {len(files)}  files")
        elif path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            info[name] = {"path": str(path), "size_mb": size_mb}
            _ok(f"{name}: {size_mb:.1f} MB")
        else:
            _fail(f"{name}: unknown path type")
            info[name] = None

    # Additional brand-count checks
    phish_dir = dataset_root / "01_phish_12K"
    if phish_dir.is_dir():
        brands = [d.name for d in phish_dir.iterdir() if d.is_dir()]
        _ok(f"Phishing brand count: {len(brands)}")

    logo_lib = dataset_root / "03_logo_reference_library" / "logo_reference_library"
    if logo_lib.is_dir():
        brands = [d.name for d in logo_lib.iterdir() if d.is_dir()]
        _ok(f"Logo reference-library brand count: {len(brands)}")

    return info


# ═══════════════════════════════════════════════════════════════════════
# 3. Model check
# ═══════════════════════════════════════════════════════════════════════

def check_models(brand_ckpt: Path, siglip_dir: Path) -> Dict[str, Optional[str]]:
    """Check that required model artifacts exist."""
    _header("3. Model check")

    info: Dict[str, Optional[str]] = {}

    # Brand-recognition checkpoint
    if brand_ckpt.is_file():
        size_mb = brand_ckpt.stat().st_size / (1024 * 1024)
        _ok(f"Brand-recognition model: {brand_ckpt.name} ({size_mb:.1f} MB)")
        info["brand_ckpt"] = str(brand_ckpt)
    else:
        _fail(f"Brand-recognition model does not exist: {brand_ckpt}")
        info["brand_ckpt"] = None

    # SigLIP directory
    if siglip_dir.is_dir():
        entries = list(siglip_dir.iterdir())
        _ok(f"SigLIP model directory: {siglip_dir} ({len(entries)}  files)")
        info["siglip_dir"] = str(siglip_dir)
        for req in ("config.json", "preprocessor_config.json"):
            if (siglip_dir / req).is_file():
                _ok(f"  {req}")
            else:
                _fail(f"  missing {req}")
    else:
        _fail(f"SigLIP model directory does not exist: {siglip_dir}")
        info["siglip_dir"] = None

    return info


# ═══════════════════════════════════════════════════════════════════════
# 4. Minimal inference test
# ═══════════════════════════════════════════════════════════════════════

def _pick_sample_benign(benign_root: Path) -> Optional[Tuple[Path, str]]:
    """Select the first usable image from the benign directory and return (image_path, sample_key)."""
    for d in sorted(benign_root.iterdir()):
        if not d.is_dir():
            continue
        png = d / f"{d.name}.png"
        if png.is_file():
            return (png, d.name)
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return (f, d.name)
    return None


def _pick_sample_phish(phish_root: Path) -> Optional[Tuple[Path, str, str]]:
    """Select the first usable image from the phishing directory and return (image_path, sample_key, brand_name)."""
    for brand_dir in sorted(phish_root.iterdir()):
        if not brand_dir.is_dir():
            continue
        screens = brand_dir / "screens"
        if not screens.is_dir():
            continue
        for f in sorted(screens.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return (f, f"{brand_dir.name}/{f.stem}", brand_dir.name)
    return None


def _resolve_prompt_dir(dataset_root: Path) -> Optional[Path]:
    """Find the prompt-library directory by priority."""
    candidates = [
        dataset_root / "04_prompt_libraries_152" / "visual_all",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def run_minimal_inference(
    dataset_root: Path,
    brand_ckpt: Path,
    siglip_dir: Path,
    device: str = "cuda:0",
) -> bool:
    """Select two real sample images, create a detection CSV, and run Stage 2 inference."""
    _header("4. Minimal inference test (2 images)")

    # -- Validate paths --
    benign_root = dataset_root / "02_bengin_12K"
    phish_root  = dataset_root / "01_phish_12K"
    logo_lib    = dataset_root / "03_logo_reference_library" / "logo_reference_library"
    normal_urls = dataset_root / "07_normal_urls_testb.csv"
    prompt_dir  = _resolve_prompt_dir(dataset_root)

    for label, path in [
        ("benign directory",   benign_root),
        ("phishing directory",    phish_root),
        ("Logo reference library",   logo_lib),
        ("normal_urls",   normal_urls),
        ("Brand-recognition model",  brand_ckpt),
        ("SigLIP directory",   siglip_dir),
        ("Prompt library",     prompt_dir or Path()),
    ]:
        if not path.exists():
            _fail(f"Skipping: {label} does not exist ({path})")
            return False

    # -- Select samples --
    print("\n  Selecting sample images...")

    benign = _pick_sample_benign(benign_root)
    if not benign:
        _fail("No benign sample image found")
        return False
    _ok(f"Benign: {benign[0].name}  (key={benign[1]})")

    phish = _pick_sample_phish(phish_root)
    if not phish:
        _fail("No phishing sample image found")
        return False
    _ok(f"Phish:  {phish[0].name}  (brand={phish[2]}, key={phish[1]})")

    # -- Get image dimensions for full-image bounding boxes --
    from PIL import Image
    b_w, b_h = Image.open(benign[0]).size
    p_w, p_h = Image.open(phish[0]).size

    # -- Create temporary detection CSV --
    tmpdir = Path(tempfile.mkdtemp(prefix="vlp_smoke_"))
    det_csv = tmpdir / "detections.csv"

    with open(det_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path", "url", "sample_key", "gt_is_phishing", "gt_brand",
            "brand_split", "has_logo",
            "det_score_1", "x1_1", "y1_1", "x2_1", "y2_1",
            "det_score_2", "x1_2", "y1_2", "x2_2", "y2_2",
            "det_score_3", "x1_3", "y1_3", "x2_3", "y2_3",
        ])
        w.writerow([
            str(benign[0]),
            "https://example.com/login",
            benign[1], 0, "", "",
            1,
            0.99, 0, 0, b_w, b_h,
            "", "", "", "", "",
            "", "", "", "", "",
        ])
        w.writerow([
            str(phish[0]),
            "https://login.paypal.com.fake.example/secure",
            phish[1], 1, phish[2], "",
            1,
            0.99, 0, 0, p_w, p_h,
            "", "", "", "", "",
            "", "", "", "", "",
        ])

    _ok(f"Detection CSV: {det_csv} ({2}  records)")

    # -- Run inference --
    out_dir = tmpdir / "output"
    out_dir.mkdir(exist_ok=True)

    cli_py = REPO_ROOT / "inference" / "run.py"

    # Automatic fallback: use CPU when CUDA is unavailable
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            _warn("CUDA unavailable; automatically switching to CPU")
    except Exception:
        device = "cpu"

    cmd = [
        sys.executable, str(cli_py),
        "--det-csv",            str(det_csv),
        "--brand-ckpt",         str(brand_ckpt),
        "--model-dir",          str(siglip_dir),
        "--logo-library-dir",   str(logo_lib),
        "--dbi5-dir",           str(prompt_dir),
        "--normal-urls-csv",    str(normal_urls),
        "--benign-root",        str(benign_root),
        "--phish-root",         str(phish_root),
        "--brand-text-mode",    "dbi5",
        "--device",             device,
        "--topk",               "1",
        "--url-policy-mode",    "official_plus_overrides_no_whitelist",
        "--threshold",          "0.62",
        "--out-dir",            str(out_dir),
        "--no-progress",
    ]

    print(f"\n  Running inference ({device})...")
    print(f"  Command: python inference/run.py --det-csv {det_csv.name} ...\n")

    try:
        env = os.environ.copy()
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + prev if prev else "")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                              capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        if proc.returncode != 0:
            _fail(f"Inference failed (return code: {proc.returncode})")
            return False
    except Exception as e:
        _fail(f"Inference execution error: {e}")
        return False

    # -- Check outputs --
    pred_csv = out_dir / "predictions.csv"
    summary_json = out_dir / "summary.json"

    if not pred_csv.is_file():
        _fail("predictions.csv was not created")
        return False

    with open(pred_csv, "r", encoding="utf-8") as f:
        n_preds = sum(1 for _ in f) - 1
    _ok(f"predictions.csv: {n_preds}  prediction records")

    if summary_json.is_file():
        with open(summary_json, "r", encoding="utf-8") as f:
            s = json.load(f)
        m = s.get("metrics", {})
        acc = m.get("accuracy")
        if isinstance(acc, (int, float)):
            _ok(f"Accuracy: {acc:.4f}")
        else:
            _ok("summary.json was created")
    else:
        _warn("summary.json was not created")

    # -- Display prediction details --
    print(f"\n  Prediction details:")
    with open(pred_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt   = "phish"  if row.get("gt_is_phishing",  "0") in ("1", "True") else "benign"
            pred = "phish"  if row.get("pred_is_phishing","0") in ("1", "True") else "benign"
            score = float(row.get("fused_score", 0))
            brand = row.get("pred_brand", "")
            match = "✓" if gt == pred else "✗"
            print(f"    [{match}] GT={gt:<6}  Pred={pred:<6}  score={score:.4f}  brand={brand}")

    # -- Clean up --
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\n  Temporary directory cleaned")

    return True


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="VLBrandPhish Reproducibility Environment Smoke Test — validate environment, data, model, and inference prerequisites",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help=f"Dataset root (default: {DEFAULT_DATASET_ROOT})")
    parser.add_argument("--brand-ckpt",   default=DEFAULT_BRAND_CKPT,
                        help=f"Brand-recognition checkpoint (default: {DEFAULT_BRAND_CKPT})")
    parser.add_argument("--siglip-dir",   default=DEFAULT_SIGLIP_DIR,
                        help=f"SigLIP model directory (default: {DEFAULT_SIGLIP_DIR})")
    parser.add_argument("--device",       default="cuda:0",
                        help="Inference device (default: cuda:0)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference; check environment, data, and models only")
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    brand_ckpt   = Path(args.brand_ckpt)
    siglip_dir   = Path(args.siglip_dir)

    print("=" * 60)
    print("  VLBrandPhish  Reproducibility Environment Smoke Test")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Repository: {REPO_ROOT}")
    print(f"  Dataset: {dataset_root}")
    print("=" * 60)

    results: List[Tuple[str, bool]] = []

    # 1. Environment
    env_ok = check_environment()
    results.append(("Environment", env_ok))

    # 2. Dataset
    ds_info = check_datasets(dataset_root)
    ds_ok = all(v is not None for v in ds_info.values()) and len(ds_info) >= 5
    results.append(("Dataset", ds_ok))

    # 3. Model
    model_info = check_models(brand_ckpt, siglip_dir)
    model_ok = all(v is not None for v in model_info.values())
    results.append(("Model", model_ok))

    # 4. Inference (only when checks 1-3 all pass)
    if args.skip_inference:
        _header("4. Minimal inference test")
        _warn("Skipped (--skip-inference)")
    elif not (env_ok and ds_ok and model_ok):
        _header("4. Minimal inference test")
        _warn("Skipped (environment, dataset, or model prerequisites are missing)")
    else:
        inf_ok = run_minimal_inference(dataset_root, brand_ckpt, siglip_dir, args.device)
        results.append(("Inference", inf_ok))

    # -- Summary --
    _header("Test summary")
    all_pass = True
    for name, status in results:
        if status:
            _ok(f"{name} check passed")
        else:
            _fail(f"{name} check failed")
            all_pass = False

    print()
    if all_pass:
        print(f"  {GREEN}All checks passed; the environment is ready.{RESET}")
    else:
        print(f"  {RED}Some checks failed; use the [FAIL] messages above to resolve them.{RESET}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
