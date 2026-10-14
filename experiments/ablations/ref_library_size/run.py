"""Reference-logo count ablation: full cross of train-N x infer-M.

Pipeline
--------
1. ``manifest``  : deterministically sample 5,000 phishing + 5,000 benign samples
                   from PhishIntention (same filtering chain as
                   ``experiments/overall_phishintention/run.py``).
2. ``stage1``    : run the DINO logo detector once -> frozen detection CSV.
3. ``infer``     : for every (train-N checkpoint, infer-M reference library) run
                   Stage-2 inference with the fixed image 0.4/text 0.6 fusion and 0.4 threshold.
4. ``aggregate`` : collect point metrics into ``ablation_table.{csv,md}``.

Text prompts, decision rule and the evaluation set are held constant across all
combinations; only the number of reference logo images per brand changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "inference").is_dir())
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results" / "pending_run"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.overall_phishintention.run import _load_phishintention_samples  # noqa: E402
from experiments._shared.e2e import write_manifest  # noqa: E402


DEFAULT_PYTHON = sys.executable
PI_ROOT = REPO_ROOT / "data" / "PhishIntention"
DEFAULT_PHISH_ROOT = PI_ROOT / "phish_sample_30k"
DEFAULT_BENIGN_ROOT = PI_ROOT / "benign_25k"
DEFAULT_NORMAL_URLS_CSV = PI_ROOT / "normal_urls.csv"
DEFAULT_NONCRP_TXT = PI_ROOT / "phish30k_nonCRP.txt"
SOURCE_LIB = PI_ROOT / "277_brand_pages"
DBI5_DIR = REPO_ROOT / "data" / "VLBrandPhish" / "06_prompt_library_277"
SIGLIP_DIR = REPO_ROOT / "models" / "vlbrandphish" / "siglip-base-patch16-224"
DET_CONFIG = REPO_ROOT / "models" / "vlbrandphish" / "patched_config.py"
DET_CKPT = REPO_ROOT / "models" / "vlbrandphish" / "detector.pth"
TRAIN_DIR = REPO_ROOT / "model_train" / "Brand" / "outputs"
PDOS_ROOT = REPO_ROOT / "baselines" / "PDOS"
PDOS_WEIGHTS = PDOS_ROOT / "results" / "training" / "pdos_yolov5s_se_logo" / "weights" / "best.pt"
PDOS_RESNET_WEIGHTS = PDOS_ROOT / "models" / "resnet101-cd907fc2.pth"
PDOS_THRESHOLD = 0.60


def _ckpt_for(train_n: int, run_id: str) -> Path:
    return TRAIN_DIR / f"refcount_{run_id}" / f"train{train_n}" / "best.pt"


def _parse_counts(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def build_manifest(n_phish: int, n_benign: int) -> Path:
    records = _load_phishintention_samples(
        phish_root=str(DEFAULT_PHISH_ROOT),
        benign_root=str(DEFAULT_BENIGN_ROOT),
        normal_urls_csv=str(DEFAULT_NORMAL_URLS_CSV),
        noncrp_txt=str(DEFAULT_NONCRP_TXT),
    )
    selected: list[dict[str, str]] = []
    seen_phish = seen_benign = 0
    for row in records:
        is_phish = str(row.get("gt_is_phishing", "0")) == "1"
        if is_phish:
            if seen_phish >= int(n_phish):
                continue
            seen_phish += 1
        else:
            if seen_benign >= int(n_benign):
                continue
            seen_benign += 1
        selected.append(row)
    manifest_path = RESULTS_DIR / "manifest.csv"
    write_manifest(selected, manifest_path)
    print(f"[manifest] phish={seen_phish} benign={seen_benign} total={len(selected)} -> {manifest_path}")
    return manifest_path


def run_stage1(python: str, manifest_path: Path, det_csv: Path, device: str) -> None:
    det_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-m", "inference.detection.manifest_command",
        "--manifest", str(manifest_path),
        "--out-csv", str(det_csv),
        "--det-config", str(DET_CONFIG),
        "--det-checkpoint", str(DET_CKPT),
        "--device", device,
        "--det-score-thr", "0.05",
        "--topk", "3",
        "--progress",
    ]
    print("[stage1] " + " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def run_inference(python: str, train_n: int, infer_m: int, det_csv: Path, device: str, run_id: str) -> Path:
    ckpt = _ckpt_for(train_n, run_id)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Trained checkpoint missing for train-N={train_n}: {ckpt}")
    ref_lib = RESULTS_DIR / f"ref_library_N{infer_m}"
    if not ref_lib.is_dir():
        raise FileNotFoundError(f"Reference library missing for infer-M={infer_m}: {ref_lib}")
    out_dir = RESULTS_DIR / f"infer_train{train_n}_ref{infer_m}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / "cache" / f"train{train_n}_ref{infer_m}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(REPO_ROOT / "inference" / "run.py"),
        "--det-csv", str(det_csv),
        "--brand-ckpt", str(ckpt),
        "--model-dir", str(SIGLIP_DIR),
        "--logo-library-dir", str(ref_lib),
        "--dbi5-dir", str(DBI5_DIR),
        "--normal-urls-csv", str(DEFAULT_NORMAL_URLS_CSV),
        "--benign-root", str(DEFAULT_BENIGN_ROOT),
        "--phish-root", str(DEFAULT_PHISH_ROOT),
        "--brand-text-mode", "ckpt_text_proto_mean",
        "--image-weight", "0.4",
        "--topk", "3",
        "--max-logo-per-brand", "-1",
        "--url-policy-mode", "phishintention",
        "--threshold", "0.4",
        "--device", device,
        "--out-dir", str(out_dir),
        "--cache-dir", str(cache_dir),
        "--no-progress",
    ]
    print(f"[infer] train-N={train_n} infer-M={infer_m} -> {out_dir.name}")
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return out_dir


def run_pdos_reference_cap(
    python: str,
    infer_m: int,
    manifest_path: Path,
    det_csv: Path,
    device: str,
    pdos_weights: Path,
    pdos_resnet_weights: Path,
) -> Path:
    """Evaluate PDOS with the shared frozen Top-3 detector CSV and a capped gallery."""
    if not pdos_weights.is_file():
        raise FileNotFoundError(f"PDOS detector weights not found: {pdos_weights}")
    if not pdos_resnet_weights.is_file():
        raise FileNotFoundError(f"PDOS ResNet weights not found: {pdos_resnet_weights}")
    ref_lib = RESULTS_DIR / f"ref_library_N{infer_m}"
    if not ref_lib.is_dir():
        raise FileNotFoundError(f"Reference library missing for infer-M={infer_m}: {ref_lib}")

    run_dir = RESULTS_DIR / "pdos" / f"ref{infer_m}"
    out_root = run_dir / "outputs"
    stage2_csv = run_dir / "stage2" / "brand_similarity.csv"
    runner = PDOS_ROOT / "scripts" / "run_pdos_staged_experiment.py"
    common = [
        python, str(runner),
        "--dataset", "phishintention",
        "--experiment", "overall_phishintention",
        "--dataset-root", str(PI_ROOT),
        "--output-root", str(out_root),
        "--manifest", str(manifest_path),
        "--stage1-csv", str(det_csv),
        "--stage2-csv", str(stage2_csv),
        "--logo-library-root", str(ref_lib),
        "--weights", str(pdos_weights),
        "--resnet-weights", str(pdos_resnet_weights),
        "--device", device,
        "--topk", "3",
        "--threshold", str(PDOS_THRESHOLD),
        "--max-reference-per-brand", "-1",
    ]
    for stage in ("recognize", "decide"):
        command = [*common, "--stage", stage]
        print(f"[PDOS] infer-M={infer_m} stage={stage}", flush=True)
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return out_root / "overall_phishintention" / "stage3" / "predictions.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _metrics_from_predictions(path: Path) -> dict[str, float]:
    rows = _read_rows(path)
    y = np.asarray([1 if str(r.get("gt_is_phishing", "0")) == "1" else 0 for r in rows], dtype=np.uint8)
    pred = np.asarray([1 if str(r.get("pred_is_phishing", "0")) == "1" else 0 for r in rows], dtype=np.uint8)
    score = np.asarray([_f(r, "score") for r in rows], dtype=np.float32)
    has_logo = np.asarray([1 if str(r.get("has_logo", "0")) == "1" else 0 for r in rows], dtype=np.uint8).astype(bool)
    url_bad = np.asarray([1 if str(r.get("url_would_be_phish", "0")) == "1" else 0 for r in rows], dtype=np.uint8).astype(bool)
    correct = np.asarray([1 if str(r.get("brand_correct_phish_only", "0")) == "1" else 0 for r in rows], dtype=np.uint8).astype(bool)
    score[~has_logo | ~url_bad] = 0.0

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-15, precision + recall)
    n_pos = max(1, int(np.sum(y == 1)))

    auroc = ap = tpr1 = float("nan")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

        auroc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
        fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
        ok = fpr <= 0.010000000001
        tpr1 = float(np.max(tpr[ok])) if np.any(ok) else 0.0
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[aggregate] curve metrics skipped: {type(exc).__name__}: {exc}")

    return {
        "n": int(len(rows)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, len(rows)),
        "benign_alert_rate": fp / max(1, tn + fp),
        "auroc": auroc,
        "ap": ap,
        "tpr_at_fpr_le_1pct": tpr1,
        "match_acc": float(np.sum(correct & (y == 1)) / n_pos),
    }


def _metrics_from_pdos_predictions(path: Path) -> dict[str, float]:
    rows = _read_rows(path)
    y = np.asarray([1 if str(r.get("gt_is_phishing", "0")) == "1" else 0 for r in rows], dtype=np.uint8)
    pred = np.asarray([1 if str(r.get("final_is_phishing", "0")) == "1" else 0 for r in rows], dtype=np.uint8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-15, precision + recall)
    phishing = [row for row in rows if str(row.get("gt_is_phishing", "0")) == "1"]
    brand_correct = sum(
        str(row.get("gt_brand", "")).strip().casefold()
        == str(row.get("predicted_brand", "")).strip().casefold()
        and bool(str(row.get("predicted_brand", "")).strip())
        for row in phishing
    )
    return {
        "n": int(len(rows)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / max(1, len(rows)),
        "benign_alert_rate": fp / max(1, tn + fp),
        "auroc": float("nan"), "ap": float("nan"),
        "tpr_at_fpr_le_1pct": float("nan"),
        "match_acc": brand_correct / max(1, len(phishing)),
    }


def aggregate(train_counts: list[int], infer_counts: list[int], baseline_f1_csv: str = "") -> None:
    rows_out: list[dict[str, object]] = []
    for train_n in train_counts:
        for infer_m in infer_counts:
            pred_path = RESULTS_DIR / f"infer_train{train_n}_ref{infer_m}" / "predictions.csv"
            if not pred_path.is_file():
                print(f"[aggregate] missing {pred_path}; skipped")
                continue
            metric = _metrics_from_predictions(pred_path)
            rows_out.append({"method": "VLBrandPhish", "train_ref": train_n, "infer_ref": infer_m, **metric})
    for infer_m in infer_counts:
        pdos_path = RESULTS_DIR / "pdos" / f"ref{infer_m}" / "outputs" / "overall_phishintention" / "stage3" / "predictions.csv"
        if pdos_path.is_file():
            rows_out.append({"method": "PDOS", "train_ref": "", "infer_ref": infer_m, **_metrics_from_pdos_predictions(pdos_path)})
        else:
            print(f"[aggregate] PDOS predictions missing; skipped: {pdos_path}")
    if baseline_f1_csv:
        baseline_path = Path(baseline_f1_csv)
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline F1 CSV not found: {baseline_path}")
        for baseline in _read_rows(baseline_path):
            method = str(baseline.get("method", "")).strip()
            if method not in {"PhishIntention", "Phishpedia"}:
                raise ValueError(f"Unsupported baseline method in {baseline_path}: {method!r}")
            try:
                infer_m = int(baseline["infer_ref"])
                f1 = float(baseline["f1"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Baseline CSV needs method,infer_ref,f1 columns.") from exc
            rows_out.append({"method": method, "train_ref": "", "infer_ref": infer_m, "f1": f1})
    if not rows_out:
        raise RuntimeError("No inference results found to aggregate.")

    fields = ["method", "train_ref", "infer_ref", "n", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy", "benign_alert_rate", "auroc", "ap", "tpr_at_fpr_le_1pct", "match_acc"]
    csv_path = RESULTS_DIR / "ablation_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    cols = ["method", "train_ref", "infer_ref", "n", "precision", "recall", "f1", "accuracy",
            "benign_alert_rate", "auroc", "ap", "tpr_at_fpr_le_1pct", "match_acc"]
    md_path = RESULTS_DIR / "ablation_table.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for row in rows_out:
            cells = []
            for c in cols:
                v = row.get(c, "")
                cells.append(f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v))
            handle.write("| " + " | ".join(cells) + " |\n")
    print(f"[aggregate] wrote {csv_path}")
    print(f"[aggregate] wrote {md_path}")


def main() -> None:
    global RESULTS_DIR, PI_ROOT, DEFAULT_PHISH_ROOT, DEFAULT_BENIGN_ROOT
    global DEFAULT_NORMAL_URLS_CSV, DEFAULT_NONCRP_TXT, SOURCE_LIB, DBI5_DIR
    global SIGLIP_DIR, DET_CONFIG, DET_CKPT, TRAIN_DIR, PDOS_THRESHOLD

    parser = argparse.ArgumentParser(description="Reference-logo count ablation (full cross).")
    parser.add_argument("--train-counts", default="1,3,5,10")
    parser.add_argument("--infer-counts", default="1,3,5,10")
    parser.add_argument("--n-phish", type=int, default=5000)
    parser.add_argument("--n-benign", type=int, default=5000)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", required=True, help="Unique output directory name.")
    parser.add_argument("--train-run-id", required=True, help="Run ID used by train_variants.py for checkpoint paths.")
    parser.add_argument("--phishintention-root", default=str(PI_ROOT))
    parser.add_argument("--source-library", default=str(SOURCE_LIB))
    parser.add_argument("--prompt-dir", default=str(DBI5_DIR))
    parser.add_argument("--model-dir", default=str(SIGLIP_DIR))
    parser.add_argument("--det-config", default=str(DET_CONFIG))
    parser.add_argument("--det-checkpoint", default=str(DET_CKPT))
    parser.add_argument("--pdos-weights", default=str(PDOS_WEIGHTS))
    parser.add_argument("--pdos-resnet-weights", default=str(PDOS_RESNET_WEIGHTS))
    parser.add_argument("--pdos-threshold", type=float, default=PDOS_THRESHOLD,
                        help="Fixed PDOS cosine threshold (0.60 matches the existing PDOS evaluation protocol).")
    parser.add_argument("--baseline-f1-csv", default="",
                        help="Optional CSV with method,infer_ref,f1 columns for PhishIntention/Phishpedia.")
    parser.add_argument(
        "--steps",
        default="manifest,stage1,infer,pdos,aggregate",
        help="Comma list of steps: manifest,stage1,infer,pdos,aggregate",
    )
    args = parser.parse_args()
    RESULTS_DIR = EXPERIMENT_DIR / "results" / args.run_id
    PI_ROOT = Path(args.phishintention_root).resolve()
    # Some releases place the split under ``test/``; the server copy stores it
    # directly under the PhishIntention root. Accept either layout.
    pi_test_root = PI_ROOT / "test" if (PI_ROOT / "test").is_dir() else PI_ROOT
    DEFAULT_PHISH_ROOT = pi_test_root / "phish_sample_30k"
    DEFAULT_BENIGN_ROOT = pi_test_root / "benign_25k"
    DEFAULT_NORMAL_URLS_CSV = PI_ROOT / "normal_urls.csv"
    DEFAULT_NONCRP_TXT = PI_ROOT / "phish30k_nonCRP.txt"
    SOURCE_LIB = Path(args.source_library).resolve()
    DBI5_DIR = Path(args.prompt_dir).resolve()
    SIGLIP_DIR = Path(args.model_dir).resolve()
    DET_CONFIG = Path(args.det_config).resolve()
    DET_CKPT = Path(args.det_checkpoint).resolve()
    TRAIN_DIR = REPO_ROOT / "model_train" / "Brand" / "outputs"
    PDOS_THRESHOLD = float(args.pdos_threshold)

    steps = {s.strip() for s in str(args.steps).split(",") if s.strip()}
    unknown_steps = steps - {"manifest", "stage1", "infer", "pdos", "aggregate"}
    if unknown_steps:
        parser.error(f"Unknown --steps value(s): {sorted(unknown_steps)}")
    train_counts = _parse_counts(args.train_counts)
    infer_counts = _parse_counts(args.infer_counts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "manifest" in steps:
        build_manifest(int(args.n_phish), int(args.n_benign))

    manifest_path = RESULTS_DIR / "manifest.csv"
    det_csv = RESULTS_DIR / "stage1" / "logo_detection_top3.csv"

    if "stage1" in steps:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest missing (run the 'manifest' step first): {manifest_path}")
        run_stage1(args.python, manifest_path, det_csv, args.device)

    if "infer" in steps or "pdos" in steps:
        if not det_csv.is_file():
            raise FileNotFoundError(f"Detection CSV missing (run the 'stage1' step first): {det_csv}")
        subprocess.run(
            [args.python, str(EXPERIMENT_DIR / "build_ref_libraries.py"), "--source-lib", str(SOURCE_LIB), "--out-root", str(RESULTS_DIR), "--counts", args.infer_counts],
            cwd=str(REPO_ROOT),
            check=True,
        )

    if "infer" in steps:
        for train_n in train_counts:
            for infer_m in infer_counts:
                run_inference(args.python, train_n, infer_m, det_csv, args.device, args.train_run_id)

    if "pdos" in steps:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest missing (run the 'manifest' step first): {manifest_path}")
        for infer_m in infer_counts:
            run_pdos_reference_cap(
                args.python,
                infer_m,
                manifest_path,
                det_csv,
                args.device,
                Path(args.pdos_weights).resolve(),
                Path(args.pdos_resnet_weights).resolve(),
            )

    if "aggregate" in steps:
        aggregate(train_counts, infer_counts, args.baseline_f1_csv)


if __name__ == "__main__":
    main()
