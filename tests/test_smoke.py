#!/usr/bin/env python
"""VLBrandPhish 复现环境烟雾测试

检查项目运行所需的四大条件，供复现论文的审稿人/读者使用：
  1. Python 环境与依赖包
  2. 数据集完整性（目录、文件数）
  3. 模型文件完整性
  4. 最小推理管线（2 张图片端到端推理，验证环境、数据集、模型和推理流程）

用法:
    python tests/test_smoke.py
    python tests/test_smoke.py --dataset-root /path/to/datasets/02_VLBrandPhish_Official
    python tests/test_smoke.py --skip-inference          # 仅检查环境/数据/模型
    python tests/test_smoke.py --device cpu              # CPU 推理
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
# 默认路径（优先级：环境变量 > 仓库约定 > 工作区上级）
# ═══════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def _first_existing(*candidates: str | Path) -> str:
    """返回第一个存在的路径；都不存在时返回第一个有效候选作为提示。"""
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
    REPO_ROOT / "models" / "siglip-base-patch16-224",
)

# Python 最低版本
MIN_PYTHON = (3, 10)

# 必需包: import_name → pip 包名
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

# 数据集子目录（相对 dataset_root）
DATASET_CHECKS = {
    "benign":           "02_bengin_12K",
    "phish":            "01_phish_12K",
    "logo_library":     "03_logo_reference_library/logo_reference_library",
    "prompt_library":   "04_prompt_libraries_152/visual_all",
    "normal_urls":      "07_normal_urls_testb.csv",
}


# ═══════════════════════════════════════════════════════════════════════
# 输出辅助
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
# 1. 环境检查
# ═══════════════════════════════════════════════════════════════════════

def check_environment() -> bool:
    """检查 Python 版本和依赖包。"""
    _header("1. Python 环境检查")

    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver >= MIN_PYTHON:
        _ok(f"Python {ver_str}  (>= {'.'.join(map(str, MIN_PYTHON))})")
    else:
        _fail(f"Python {ver_str}  (需要 >= {'.'.join(map(str, MIN_PYTHON))})")
        return False

    all_ok = True
    for import_name, pkg_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            _ok(f"{pkg_name}")
        except ImportError:
            _fail(f"{pkg_name} 未安装  →  pip install {pkg_name}")
            all_ok = False

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            _ok(f"CUDA 可用 (设备数: {n}, {name})")
        else:
            _warn("CUDA 不可用，推理将使用 CPU（速度较慢）")
    except Exception:
        pass

    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# 2. 数据集检查
# ═══════════════════════════════════════════════════════════════════════

def check_datasets(dataset_root: Path) -> Dict[str, Optional[dict]]:
    """扫描数据集目录，统计文件数和子目录数。"""
    _header("2. 数据集检查")

    info: Dict[str, Optional[dict]] = {}

    if not dataset_root.exists():
        _fail(f"数据集根目录不存在: {dataset_root}")
        return info

    for name, rel in DATASET_CHECKS.items():
        path = dataset_root / rel
        if not path.exists():
            _fail(f"{name}: 路径不存在 ({path})")
            info[name] = None
            continue

        if path.is_dir():
            entries = list(path.iterdir())
            subdirs = [e for e in entries if e.is_dir()]
            files   = [e for e in entries if e.is_file()]
            info[name] = {"path": str(path), "dirs": len(subdirs), "files": len(files)}
            _ok(f"{name}: {len(subdirs)} 个子目录, {len(files)} 个文件")
        elif path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            info[name] = {"path": str(path), "size_mb": size_mb}
            _ok(f"{name}: {size_mb:.1f} MB")
        else:
            _fail(f"{name}: 路径类型未知")
            info[name] = None

    # 额外：brand 数量统计
    phish_dir = dataset_root / "01_phish_12K"
    if phish_dir.is_dir():
        brands = [d.name for d in phish_dir.iterdir() if d.is_dir()]
        _ok(f"钓鱼品牌数: {len(brands)}")

    logo_lib = dataset_root / "03_logo_reference_library" / "logo_reference_library"
    if logo_lib.is_dir():
        brands = [d.name for d in logo_lib.iterdir() if d.is_dir()]
        _ok(f"Logo 参考库品牌数: {len(brands)}")

    return info


# ═══════════════════════════════════════════════════════════════════════
# 3. 模型检查
# ═══════════════════════════════════════════════════════════════════════

def check_models(brand_ckpt: Path, siglip_dir: Path) -> Dict[str, Optional[str]]:
    """检查模型文件是否存在。"""
    _header("3. 模型文件检查")

    info: Dict[str, Optional[str]] = {}

    # 品牌识别 checkpoint
    if brand_ckpt.is_file():
        size_mb = brand_ckpt.stat().st_size / (1024 * 1024)
        _ok(f"品牌识别模型: {brand_ckpt.name} ({size_mb:.1f} MB)")
        info["brand_ckpt"] = str(brand_ckpt)
    else:
        _fail(f"品牌识别模型不存在: {brand_ckpt}")
        info["brand_ckpt"] = None

    # SigLIP 目录
    if siglip_dir.is_dir():
        entries = list(siglip_dir.iterdir())
        _ok(f"SigLIP 模型目录: {siglip_dir} ({len(entries)} 个文件)")
        info["siglip_dir"] = str(siglip_dir)
        for req in ("config.json", "preprocessor_config.json"):
            if (siglip_dir / req).is_file():
                _ok(f"  {req}")
            else:
                _fail(f"  缺少 {req}")
    else:
        _fail(f"SigLIP 模型目录不存在: {siglip_dir}")
        info["siglip_dir"] = None

    return info


# ═══════════════════════════════════════════════════════════════════════
# 4. 最小推理测试
# ═══════════════════════════════════════════════════════════════════════

def _pick_sample_benign(benign_root: Path) -> Optional[Tuple[Path, str]]:
    """从 benign 目录选取第一张可用图片，返回 (image_path, sample_key)。"""
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
    """从 phishing 目录选取第一张可用图片，返回 (image_path, sample_key, brand_name)。"""
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
    """按优先级查找 prompt library 目录。"""
    candidates = [
        dataset_root / "04_prompt_libraries_152" / "visual_all",
        dataset_root / "06_generated_prompt_libraries" / "visual_all",
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
    """选取 2 张真实样本图片，构造检测 CSV，运行 Stage 2 推理。"""
    _header("4. 最小推理测试 (2 张图片)")

    # ── 验证路径 ──
    benign_root = dataset_root / "02_bengin_12K"
    phish_root  = dataset_root / "01_phish_12K"
    logo_lib    = dataset_root / "03_logo_reference_library" / "logo_reference_library"
    normal_urls = dataset_root / "07_normal_urls_testb.csv"
    prompt_dir  = _resolve_prompt_dir(dataset_root)

    for label, path in [
        ("benign 目录",   benign_root),
        ("phish 目录",    phish_root),
        ("Logo 参考库",   logo_lib),
        ("normal_urls",   normal_urls),
        ("品牌识别模型",  brand_ckpt),
        ("SigLIP 目录",   siglip_dir),
        ("Prompt 库",     prompt_dir or Path()),
    ]:
        if not path.exists():
            _fail(f"跳过: {label} 不存在 ({path})")
            return False

    # ── 选取样本 ──
    print("\n  选取样本图片...")

    benign = _pick_sample_benign(benign_root)
    if not benign:
        _fail("未找到 benign 样本图片")
        return False
    _ok(f"Benign: {benign[0].name}  (key={benign[1]})")

    phish = _pick_sample_phish(phish_root)
    if not phish:
        _fail("未找到 phishing 样本图片")
        return False
    _ok(f"Phish:  {phish[0].name}  (brand={phish[2]}, key={phish[1]})")

    # ── 获取图片尺寸（用于整图 bbox）──
    from PIL import Image
    b_w, b_h = Image.open(benign[0]).size
    p_w, p_h = Image.open(phish[0]).size

    # ── 创建临时检测 CSV ──
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

    _ok(f"检测 CSV: {det_csv} ({2} 条记录)")

    # ── 运行推理 ──
    out_dir = tmpdir / "output"
    out_dir.mkdir(exist_ok=True)

    cli_py = REPO_ROOT / "inference" / "run.py"

    # 自动 fallback: 如果 CUDA 不可用则用 CPU
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            _warn("CUDA 不可用，自动切换为 CPU")
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

    print(f"\n  执行推理 ({device})...")
    print(f"  命令: python inference/run.py --det-csv {det_csv.name} ...\n")

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
            _fail(f"推理失败 (返回码: {proc.returncode})")
            return False
    except Exception as e:
        _fail(f"推理执行异常: {e}")
        return False

    # ── 检查输出 ──
    pred_csv = out_dir / "predictions.csv"
    summary_json = out_dir / "summary.json"

    if not pred_csv.is_file():
        _fail("predictions.csv 未生成")
        return False

    with open(pred_csv, "r", encoding="utf-8") as f:
        n_preds = sum(1 for _ in f) - 1
    _ok(f"predictions.csv: {n_preds} 条预测结果")

    if summary_json.is_file():
        with open(summary_json, "r", encoding="utf-8") as f:
            s = json.load(f)
        m = s.get("metrics", {})
        acc = m.get("accuracy")
        if isinstance(acc, (int, float)):
            _ok(f"准确率: {acc:.4f}")
        else:
            _ok("summary.json 已生成")
    else:
        _warn("summary.json 未生成")

    # ── 显示预测详情 ──
    print(f"\n  预测详情:")
    with open(pred_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt   = "phish"  if row.get("gt_is_phishing",  "0") in ("1", "True") else "benign"
            pred = "phish"  if row.get("pred_is_phishing","0") in ("1", "True") else "benign"
            score = float(row.get("fused_score", 0))
            brand = row.get("pred_brand", "")
            match = "✓" if gt == pred else "✗"
            print(f"    [{match}] GT={gt:<6}  Pred={pred:<6}  score={score:.4f}  brand={brand}")

    # ── 清理 ──
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\n  已清理临时目录")

    return True


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="VLBrandPhish 复现环境烟雾测试 — 校验环境/数据/模型/推理四项条件",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help=f"数据集根目录 (默认: {DEFAULT_DATASET_ROOT})")
    parser.add_argument("--brand-ckpt",   default=DEFAULT_BRAND_CKPT,
                        help=f"品牌识别 checkpoint (默认: {DEFAULT_BRAND_CKPT})")
    parser.add_argument("--siglip-dir",   default=DEFAULT_SIGLIP_DIR,
                        help=f"SigLIP 模型目录 (默认: {DEFAULT_SIGLIP_DIR})")
    parser.add_argument("--device",       default="cuda:0",
                        help="推理设备 (默认: cuda:0)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过推理测试，仅检查环境/数据/模型")
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    brand_ckpt   = Path(args.brand_ckpt)
    siglip_dir   = Path(args.siglip_dir)

    print("=" * 60)
    print("  VLBrandPhish  复现环境烟雾测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  仓库: {REPO_ROOT}")
    print(f"  数据集: {dataset_root}")
    print("=" * 60)

    results: List[Tuple[str, bool]] = []

    # 1. 环境
    env_ok = check_environment()
    results.append(("环境", env_ok))

    # 2. 数据集
    ds_info = check_datasets(dataset_root)
    ds_ok = all(v is not None for v in ds_info.values()) and len(ds_info) >= 5
    results.append(("数据集", ds_ok))

    # 3. 模型
    model_info = check_models(brand_ckpt, siglip_dir)
    model_ok = all(v is not None for v in model_info.values())
    results.append(("模型", model_ok))

    # 4. 推理（仅在 1/2/3 全部通过时执行）
    if args.skip_inference:
        _header("4. 最小推理测试")
        _warn("跳过 (--skip-inference)")
    elif not (env_ok and ds_ok and model_ok):
        _header("4. 最小推理测试")
        _warn("跳过（前置条件未满足：环境/数据集/模型有缺失）")
    else:
        inf_ok = run_minimal_inference(dataset_root, brand_ckpt, siglip_dir, args.device)
        results.append(("推理", inf_ok))

    # ── 总结 ──
    _header("测试总结")
    all_pass = True
    for name, status in results:
        if status:
            _ok(f"{name}检查 通过")
        else:
            _fail(f"{name}检查 未通过")
            all_pass = False

    print()
    if all_pass:
        print(f"  {GREEN}全部检查通过，环境已准备就绪！{RESET}")
    else:
        print(f"  {RED}部分检查未通过，请根据上述 [FAIL] 提示修复。{RESET}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
