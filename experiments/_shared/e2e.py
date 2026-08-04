"""Shared, manifest-driven execution helpers for VLBrandPhish experiments.

The four paper experiments use the same two-stage method.  This module owns the
boring but important parts once: frozen asset paths, manifest serialization,
subprocess execution in the two required environments, and result aggregation.
Individual experiment ``run.py`` files only define which samples belong to a
protocol.  This keeps every reported result on the identical fixed-fusion path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from inference.data.detection_csv import (
    DETECTION_TOP3_HEADER,
    DetectionSample,
    detection_to_csv_row,
    read_detection_top3_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent


def _first_existing(*candidates: str | Path) -> str:
    """Return the first available candidate, preserving a useful fallback path."""
    cleaned = [Path(item) for item in candidates if str(item or "").strip()]
    for candidate in cleaned:
        if candidate.exists():
            return str(candidate)
    return str(cleaned[0]) if cleaned else ""


DEFAULT_DATASET_ROOT = _first_existing(
    os.environ.get("VLBRANDPHISH_DATASET_ROOT", ""),
    REPO_ROOT / "data" / "VLBrandPhish",
    WORKSPACE_ROOT / "datasets" / "02_VLBrandPhish_Official",
    REPO_ROOT / "data" / "vlbp_set",
)
DEFAULT_DETECTOR_PYTHON = _first_existing(
    os.environ.get("VLBRANDPHISH_DETECTOR_PYTHON", ""),
    sys.executable,
)
DEFAULT_INFERENCE_PYTHON = _first_existing(
    os.environ.get("VLBRANDPHISH_INFERENCE_PYTHON", ""),
    sys.executable,
)
DEFAULT_SIGLIP_DIR = _first_existing(
    os.environ.get("VLBRANDPHISH_SIGLIP_MODEL_DIR", ""),
    REPO_ROOT / "models" / "siglip-base-patch16-224",
)
DEFAULT_DBI5_DIR = _first_existing(
    os.environ.get("VLBRANDPHISH_DBI5_DIR", ""),
    REPO_ROOT / "data" / "VLBrandPhish" / "06_prompt_library_277",
)

MANIFEST_BASE_FIELDS = (
    "sample_key",
    "image_path",
    "url",
    "gt_is_phishing",
    "gt_brand",
)


@dataclass(frozen=True)
class ExperimentAssets:
    """Frozen assets needed to execute the two stages of one experiment."""

    detector_python: str
    inference_python: str
    detector_config: str
    detector_checkpoint: str
    brand_checkpoint: str
    model_dir: str
    logo_library_dir: str
    dbi5_dir: str
    normal_urls_csv: str
    benign_root: str
    phish_root: str
    device: str
    brand_text_mode: str
    brands_overlap_csv: str = ""
    restrict_brands_csv: str = ""


def add_e2e_arguments(parser, *, default_dataset_root: str = DEFAULT_DATASET_ROOT) -> None:
    """Add the common CLI contract shared by all four experiment runners."""
    models_dir = REPO_ROOT / "models" / "vlbrandphish"
    parser.add_argument("--dataset-root", default=default_dataset_root, help="Official VLBP dataset root.")
    parser.add_argument("--out-dir", default="", help="Run directory; defaults to this experiment's results folder.")
    parser.add_argument("--run-name", default="", help="Readable suffix for a timestamped default output directory.")
    parser.add_argument("--prepare-only", action="store_true", help="Write and validate the input manifest without inference.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a non-empty run directory only after its frozen manifest is verified.",
    )
    parser.add_argument("--detector-python", default=DEFAULT_DETECTOR_PYTHON, help="Python executable with MMDetection.")
    parser.add_argument("--inference-python", default=DEFAULT_INFERENCE_PYTHON, help="Python executable with SigLIP dependencies.")
    parser.add_argument("--det-config", default=str(models_dir / "patched_config.py"), help="DINO detector config.")
    parser.add_argument("--det-checkpoint", default=_first_existing(models_dir / "detect_6250.pth", models_dir / "detector.pth"), help="DINO detector checkpoint.")
    parser.add_argument(
        "--brand-checkpoint",
        default=_first_existing(models_dir / "brand_encoder_inference.pt", models_dir / "brand_identify.pt"),
        help="Inference-only SigLIP checkpoint.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_SIGLIP_DIR, help="Local Hugging Face SigLIP directory.")
    parser.add_argument(
        "--logo-library-dir",
        default=str(Path(default_dataset_root) / "03_logo_reference_library" / "logo_reference_library"),
        help="Reference-logo directory.",
    )
    parser.add_argument("--dbi5-dir", default=DEFAULT_DBI5_DIR, help="DBI text-prompt directory for dbi5 mode.")
    parser.add_argument(
        "--normal-urls-csv",
        default=str(Path(default_dataset_root) / "07_normal_urls_testb.csv"),
        help="Official brand-domain mapping.",
    )
    parser.add_argument(
        "--benign-root",
        default="",
        help="Override benign dataset root (default derives from --dataset-root).",
    )
    parser.add_argument(
        "--phish-root",
        default="",
        help="Override phishing dataset root (default derives from --dataset-root).",
    )
    parser.add_argument("--device", default="cuda:0", help="MMDetection and SigLIP device.")
    parser.add_argument(
        "--brand-text-mode",
        default="dbi5",
        choices=["dbi5", "train_templates_mean", "train12_mean", "train_templates",
                 "ckpt_text_proto", "ckpt_text_proto_mean", "ckpt_proto", "ckpt_proto_mean"],
        help="Fixed text-evidence construction mode; ckpt_* modes reuse pre-encoded "
        "prototypes and avoid the Windows sentencepiece tokenizer crash.",
    )
    parser.add_argument("--det-score-thr", type=float, default=0.05, help="Logo detector score floor.")
    parser.add_argument("--topk", type=int, default=3, help="Retain this many logo detections per screenshot.")
    parser.add_argument("--max-logo-per-brand", type=int, default=-1, help="Optional deterministic reference-logo cap.")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--no-progress", action="store_true", help="Suppress Stage-1 progress messages.")
    parser.add_argument(
        "--brands-overlap-csv",
        default="",
        help="Optional overlap CSV for canonical→277 alias lookup (used for repository brand name mapping).",
    )
    parser.add_argument(
        "--restrict-brands-csv",
        default="",
        help="Optional overlap CSV (in_277=1 or 0 filter) that restricts the logo/token repository brands for cross-time or cross-brand experiments.",
    )


def assets_from_args(args) -> ExperimentAssets:
    """Resolve assets from parsed arguments without silently changing a protocol."""
    dataset_root = Path(args.dataset_root).resolve()
    bo = str(getattr(args, "brands_overlap_csv", "") or "").strip()
    rb = str(getattr(args, "restrict_brands_csv", "") or "").strip()
    br = str(getattr(args, "benign_root", "") or "").strip()
    pr = str(getattr(args, "phish_root", "") or "").strip()
    if br:
        benign_root = str(Path(br).resolve())
    else:
        benign_root = str((dataset_root / "02_bengin_12K").resolve())
    if pr:
        phish_root = str(Path(pr).resolve())
    else:
        phish_root = str((dataset_root / "01_phish_12K").resolve())
    return ExperimentAssets(
        detector_python=str(Path(args.detector_python).resolve()),
        inference_python=str(Path(args.inference_python).resolve()),
        detector_config=str(Path(args.det_config).resolve()),
        detector_checkpoint=str(Path(args.det_checkpoint).resolve()),
        brand_checkpoint=str(Path(args.brand_checkpoint).resolve()),
        model_dir=str(Path(args.model_dir).resolve()),
        logo_library_dir=str(Path(args.logo_library_dir).resolve()),
        dbi5_dir=str(Path(args.dbi5_dir).resolve()),
        normal_urls_csv=str(Path(args.normal_urls_csv).resolve()),
        benign_root=benign_root,
        phish_root=phish_root,
        device=str(args.device),
        brand_text_mode=str(args.brand_text_mode),
        brands_overlap_csv=str(Path(bo).resolve()) if bo else "",
        restrict_brands_csv=str(Path(rb).resolve()) if rb else "",
    )


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return dict(json.load(handle))


def _metadata_timestamp(image_path: str, *, is_phishing: bool) -> str:
    """Recover a collection timestamp when the official Test-B layout exposes one."""
    image = Path(image_path)
    try:
        if is_phishing:
            meta = image.parent.parent / "jsons" / f"{image.stem}.json"
        else:
            meta = image.with_suffix(".json")
        return str(_read_json(meta).get("timestamp", "") or "") if meta.is_file() else ""
    except (OSError, ValueError, json.JSONDecodeError):
        return ""


def official_records(
    dataset_root: str | Path,
    *,
    selected_phish_brands: set[str] | None = None,
    phish_split: str = "full",
    max_benign: int = -1,
    max_phish: int = -1,
) -> list[dict[str, str]]:
    """Materialize a deterministic manifest from the official full-dataset layout."""
    from inference.data.testb_dataset import load_testb_samples

    root = Path(dataset_root).resolve()
    samples = load_testb_samples(
        benign_root=str(root / "02_bengin_12K"),
        phish_root=str(root / "01_phish_12K"),
        max_benign=-1,
        max_phish=-1,
    )
    records: list[dict[str, str]] = []
    n_benign = 0
    n_phish = 0
    allowed = set(selected_phish_brands or [])
    for sample in samples:
        is_phish = bool(sample.gt_is_phishing)
        if is_phish and allowed and str(sample.gt_brand or "") not in allowed:
            continue
        if is_phish:
            if max_phish >= 0 and n_phish >= int(max_phish):
                continue
            n_phish += 1
        else:
            if max_benign >= 0 and n_benign >= int(max_benign):
                continue
            n_benign += 1
        records.append(
            {
                "sample_key": str(sample.sample_key),
                "image_path": str(Path(sample.image_path).resolve()),
                "url": str(sample.url),
                "gt_is_phishing": "1" if is_phish else "0",
                "gt_brand": str(sample.gt_brand or ""),
                "dataset_split": phish_split if is_phish else "benign",
                "collection_timestamp": _metadata_timestamp(sample.image_path, is_phishing=is_phish),
                "source_dataset": "VLBrandPhish_Official",
            }
        )
    if not records:
        raise RuntimeError("The selected official-data split has zero usable samples.")
    return records


def limit_by_class(
    records: Sequence[Mapping[str, object]], *, max_benign: int, max_phish: int
) -> list[dict[str, str]]:
    """Apply deterministic per-class limits after an experiment-specific split."""
    selected: list[dict[str, str]] = []
    seen_benign = 0
    seen_phish = 0
    for record in records:
        row = {str(k): str(v) for k, v in record.items()}
        is_phish = str(row.get("gt_is_phishing", "0")).strip().lower() in {"1", "true", "yes"}
        if is_phish:
            if max_phish >= 0 and seen_phish >= int(max_phish):
                continue
            seen_phish += 1
        else:
            if max_benign >= 0 and seen_benign >= int(max_benign):
                continue
            seen_benign += 1
        selected.append(row)
    return selected


def write_manifest(records: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    """Write a self-contained, stable sample manifest and validate its contract."""
    if not records:
        raise ValueError("Cannot write an empty experiment manifest.")
    normalized = [{str(k): str(v) for k, v in item.items()} for item in records]
    for field in MANIFEST_BASE_FIELDS:
        if any(not row.get(field, "") and field not in {"gt_brand"} for row in normalized):
            raise ValueError(f"Manifest field '{field}' is required for every sample.")
    keys = [row["sample_key"] for row in normalized]
    duplicate_keys = [key for key, count in Counter(keys).items() if count > 1]
    if duplicate_keys:
        raise ValueError(f"Manifest sample_key values must be unique; duplicates include {duplicate_keys[:5]}")
    extras = sorted({key for row in normalized for key in row} - set(MANIFEST_BASE_FIELDS))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*MANIFEST_BASE_FIELDS, *extras],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(normalized)
    return target


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read the experiment manifest without accepting missing core fields."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = [field for field in MANIFEST_BASE_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return rows


def _core_manifest_values(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return the immutable fields used to prove a resume targets the same data."""
    return tuple(str(row.get(field, "") or "") for field in MANIFEST_BASE_FIELDS)


def _validate_resume_manifest(
    *, manifest_rows: Sequence[Mapping[str, str]], requested_records: Sequence[Mapping[str, object]]
) -> None:
    """Reject resume attempts whose current input differs from the frozen manifest."""
    if len(manifest_rows) != len(requested_records):
        raise ValueError(
            "Cannot resume: existing manifest row count differs from current protocol selection "
            f"({len(manifest_rows)} != {len(requested_records)})."
        )
    for index, (existing, requested) in enumerate(zip(manifest_rows, requested_records), start=1):
        if _core_manifest_values(existing) != _core_manifest_values(requested):
            raise ValueError(
                "Cannot resume: existing manifest differs from current protocol selection at "
                f"row {index}; use a new --run-name instead."
            )


def _prediction_csv_is_complete(path: Path, manifest_rows: Sequence[Mapping[str, str]]) -> bool:
    """Return whether an existing Stage-2 CSV is a one-to-one result for the manifest."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    expected = [str(row["sample_key"]) for row in manifest_rows]
    observed = [str(row.get("sample_key", "")) for row in rows]
    if len(observed) != len(expected) or len(set(observed)) != len(observed):
        raise RuntimeError(
            "Stage-2 predictions.csv exists but is incomplete or has duplicate sample keys; "
            "move it aside before resuming."
        )
    if set(observed) != set(expected):
        raise RuntimeError(
            "Stage-2 predictions.csv does not correspond to the frozen manifest; "
            "move it aside before resuming."
        )
    return True

def _file_hash(path: str | Path) -> dict[str, object]:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def _validate_assets(assets: ExperimentAssets) -> None:
    required_files = {
        "detector Python": assets.detector_python,
        "inference Python": assets.inference_python,
        "detector config": assets.detector_config,
        "detector checkpoint": assets.detector_checkpoint,
        "brand checkpoint": assets.brand_checkpoint,
        "normal URL mapping": assets.normal_urls_csv,
    }
    required_dirs = {
        "SigLIP model directory": assets.model_dir,
        "reference logo library": assets.logo_library_dir,
    }
    if assets.brand_text_mode == "dbi5":
        required_dirs["DBI-5 prompt directory"] = assets.dbi5_dir
    missing = [f"{name}: {path}" for name, path in required_files.items() if not Path(path).is_file()]
    missing.extend(f"{name}: {path}" for name, path in required_dirs.items() if not Path(path).is_dir())
    if missing:
        raise FileNotFoundError("Required experiment assets are unavailable:\n- " + "\n- ".join(missing))


def _stream_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    """Run a stage, mirror its output to console, and retain a complete stage log."""
    environment = os.environ.copy()
    previous = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + previous if previous else "")
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _write_unique_image_manifest(full_manifest: Path, run_dir: Path) -> Path:
    """Create a deduplicated manifest with one record per unique image_path.

    Returns the path to the unique-image manifest.  When every record already
    has a distinct image, the original manifest is returned unchanged.
    """
    with full_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        img = row.get("image_path", "")
        if img in seen:
            continue
        seen.add(img)
        unique_rows.append(row)
    if len(unique_rows) == len(rows):
        return full_manifest
    unique_path = run_dir / "stage1" / "unique_images_manifest.csv"
    unique_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with unique_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    print(
        f"[stage1] deduplicated manifest: {len(rows)} records -> {len(unique_rows)} unique images "
        f"({len(rows) - len(unique_rows)} shared-image records skipped for detection)",
        flush=True,
    )
    return unique_path


def _expand_detection_csv(unique_csv: Path, full_manifest: Path, target_csv: Path) -> None:
    """Expand a unique-image detection CSV to cover every record in the full manifest.

    Reads detections keyed by ``image_path`` from *unique_csv*, then writes one
    row per manifest record to *target_csv*, copying the detection boxes/scores
    from the shared image.
    """
    samples = read_detection_top3_csv(str(unique_csv))
    det_by_image: dict[str, tuple[list[tuple[float, float, float, float]], list[float]]] = {}
    for s in samples:
        det_by_image[s.image_path] = (s.bboxes, s.det_scores)
    with full_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with target_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_TOP3_HEADER)
        writer.writeheader()
        for row in manifest_rows:
            img = row.get("image_path", "")
            bboxes, scores = det_by_image.get(img, ([], []))
            sample = DetectionSample(
                image_path=img,
                url=row.get("url", ""),
                sample_key=row.get("sample_key", ""),
                gt_is_phishing=_is_true(row.get("gt_is_phishing", "0")),
                gt_brand=row.get("gt_brand", "") or None,
                bboxes=bboxes,
                det_scores=scores,
            )
            writer.writerow(detection_to_csv_row(sample))
    print(
        f"[stage1] expanded detection CSV: {len(samples)} unique images -> {len(manifest_rows)} records "
        f"at {target_csv}",
        flush=True,
    )


def _is_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) else 0.0


def _evaluate_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """Calculate the same fixed-threshold outcome metrics for a result subgroup."""
    tp = tn = fp = fn = 0
    brand_total = brand_correct = 0
    for row in rows:
        truth = _is_true(row.get("gt_is_phishing"))
        predicted = _is_true(row.get("pred_is_phishing"))
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
        if truth and str(row.get("gt_brand", "")).strip():
            brand_total += 1
            if predicted and _is_true(row.get("brand_correct_phish_only")):
                brand_correct += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "n_samples": int(len(rows)),
        "n_phish": int(tp + fn),
        "n_benign": int(tn + fp),
        "n_pred_phish": int(tp + fp),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "benign_alert_rate": _safe_div(fp, tn + fp),
        "brand_acc_on_phish": _safe_div(brand_correct, brand_total),
    }


def _write_aggregates(
    *, run_dir: Path, manifest_rows: Sequence[Mapping[str, str]], group_fields: Sequence[str]
) -> dict[str, object]:
    """Join inference predictions back to protocol metadata and write grouped metrics."""
    prediction_path = run_dir / "stage2" / "predictions.csv"
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        predictions = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    manifest_by_key = {str(row["sample_key"]): dict(row) for row in manifest_rows}
    joined: list[dict[str, str]] = []
    for prediction in predictions:
        key = str(prediction.get("sample_key", ""))
        if key not in manifest_by_key:
            raise RuntimeError(f"Prediction key absent from frozen manifest: {key}")
        joined.append({**manifest_by_key[key], **prediction})
    if len(joined) != len(manifest_rows):
        raise RuntimeError(f"Prediction count {len(joined)} does not match manifest count {len(manifest_rows)}")

    prediction_fields = [*MANIFEST_BASE_FIELDS]
    all_fields = {key for row in joined for key in row}
    prediction_fields.extend(sorted(all_fields - set(prediction_fields)))
    merged_path = run_dir / "predictions_with_manifest.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(joined)

    metric_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {("all",): joined}
    if group_fields:
        groups = {}
        for row in joined:
            key = tuple(str(row.get(field, "") or "") for field in group_fields)
            groups.setdefault(key, []).append(row)
        groups[("all",) * len(group_fields)] = joined
    for key in sorted(groups):
        metric = _evaluate_rows(groups[key])
        if group_fields:
            metric.update({field: value for field, value in zip(group_fields, key)})
        metric_rows.append(metric)
    metric_fields = [*group_fields, "n_samples", "n_phish", "n_benign", "n_pred_phish", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "benign_alert_rate", "brand_acc_on_phish"]
    metrics_path = run_dir / "metrics_by_group.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (run_dir / "metrics_by_group.json").open("w", encoding="utf-8") as handle:
        json.dump(metric_rows, handle, ensure_ascii=False, indent=2)
    return {"predictions_with_manifest": str(merged_path), "metrics_by_group": str(metrics_path), "overall": _evaluate_rows(joined)}


def execute_experiment(
    *,
    experiment_name: str,
    experiment_dir: str | Path,
    records: Sequence[Mapping[str, object]],
    args,
    group_fields: Sequence[str] = (),
    protocol: Mapping[str, object] | None = None,
) -> Path:
    """Freeze selected samples, execute both stages, and save reproducible outputs."""
    experiment_dir = Path(experiment_dir).resolve()
    if args.out_dir:
        run_dir = Path(args.out_dir).resolve()
    else:
        suffix = str(args.run_name or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = experiment_dir / "results" / f"{experiment_name}_{suffix}"
    run_preexisted = run_dir.exists() and any(run_dir.iterdir())
    if run_preexisted and not bool(args.resume):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty experiment directory: {run_dir}. "
            "Pass --resume only for the same frozen protocol."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "input_manifest.csv"
    if run_preexisted:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without input_manifest.csv: {run_dir}")
        manifest_rows = read_manifest(manifest_path)
        _validate_resume_manifest(manifest_rows=manifest_rows, requested_records=records)
        print(f"[experiment] verified resume manifest: {manifest_path}")
    else:
        manifest_path = write_manifest(records, manifest_path)
        manifest_rows = read_manifest(manifest_path)
    selected_counts = {
        "n_samples": len(manifest_rows),
        "n_phish": sum(_is_true(row.get("gt_is_phishing")) for row in manifest_rows),
        "n_benign": sum(not _is_true(row.get("gt_is_phishing")) for row in manifest_rows),
    }
    base_summary: dict[str, object] = {
        "experiment": experiment_name,
        "status": "prepared",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_manifest": str(manifest_path),
        "counts": selected_counts,
        "protocol": dict(protocol or {}),
        "fixed_decision": {"image_weight": 0.8, "text_weight": 0.2, "threshold": 0.62},
    }
    if args.prepare_only:
        with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(base_summary, handle, ensure_ascii=False, indent=2)
        print(f"[experiment] prepared manifest only: {run_dir}")
        return run_dir

    assets = assets_from_args(args)
    _validate_assets(assets)
    base_summary["assets"] = {
        "detector_config": _file_hash(assets.detector_config),
        "detector_checkpoint": _file_hash(assets.detector_checkpoint),
        "brand_checkpoint": _file_hash(assets.brand_checkpoint),
        "normal_urls_csv": _file_hash(assets.normal_urls_csv),
        "logo_library_dir": assets.logo_library_dir,
        "dbi5_dir": assets.dbi5_dir if assets.brand_text_mode == "dbi5" else "",
        "model_dir": assets.model_dir,
        "detector_python": assets.detector_python,
        "inference_python": assets.inference_python,
        "brand_text_mode": assets.brand_text_mode,
        "device": assets.device,
    }

    base_summary["status"] = "running"
    base_summary["resumed_from_existing_run"] = bool(run_preexisted)
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(base_summary, handle, ensure_ascii=False, indent=2)

    detection_csv = run_dir / "stage1" / "logo_detection_top3.csv"
    detection_csv.parent.mkdir(parents=True, exist_ok=True)

    # When multiple manifest records share the same image (e.g. PhishEval
    # squatted-URL variants), deduplicate to one record per image so Stage-1
    # only runs the detector on unique screenshots, then expand the detection
    # CSV to cover every record for Stage-2.
    stage1_manifest = _write_unique_image_manifest(manifest_path, run_dir)
    stage1_out = detection_csv if stage1_manifest is manifest_path else run_dir / "stage1" / "unique_detections.csv"
    stage1_command = [
        assets.detector_python,
        str(REPO_ROOT / "experiments" / "_shared" / "stage1.py"),
        "--manifest",
        str(stage1_manifest),
        "--out-csv",
        str(stage1_out),
        "--det-config",
        assets.detector_config,
        "--det-checkpoint",
        assets.detector_checkpoint,
        "--device",
        assets.device,
        "--det-score-thr",
        str(float(args.det_score_thr)),
        "--topk",
        str(int(args.topk)),
    ]
    if not args.no_progress:
        stage1_command.append("--progress")
    _stream_command(stage1_command, cwd=REPO_ROOT, log_path=run_dir / "stage1.log")
    if not stage1_out.is_file():
        raise RuntimeError("Stage 1 reported success but did not write its detection CSV.")
    if stage1_out is not detection_csv:
        _expand_detection_csv(stage1_out, manifest_path, detection_csv)

    stage2_dir = run_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage2_command = [
        assets.inference_python,
        str(REPO_ROOT / "inference" / "run.py"),
        "--det-csv",
        str(detection_csv),
        "--brand-ckpt",
        assets.brand_checkpoint,
        "--model-dir",
        assets.model_dir,
        "--logo-library-dir",
        assets.logo_library_dir,
        "--normal-urls-csv",
        assets.normal_urls_csv,
        "--benign-root",
        assets.benign_root,
        "--phish-root",
        assets.phish_root,
        "--brand-text-mode",
        assets.brand_text_mode,
        "--max-logo-per-brand",
        str(int(args.max_logo_per_brand)),
        "--topk",
        str(int(args.topk)),
        "--image-batch-size",
        str(int(args.image_batch_size)),
        "--text-batch-size",
        str(int(args.text_batch_size)),
        "--device",
        assets.device,
        "--url-policy-mode",
        "official_plus_overrides_no_whitelist",
        "--threshold",
        "0.62",
        "--out-dir",
        str(stage2_dir),
        "--cache-dir",
        str(stage2_dir / "cache"),
        "--no-progress",
    ]
    prediction_path = stage2_dir / "predictions.csv"
    if _prediction_csv_is_complete(prediction_path, manifest_rows):
        print(f"[experiment] verified existing Stage-2 predictions: {prediction_path}")
    else:
        if assets.brand_text_mode == "dbi5":
            stage2_command.extend(["--dbi5-dir", assets.dbi5_dir])
        if assets.brands_overlap_csv:
            stage2_command.extend(["--brands-overlap-csv", assets.brands_overlap_csv])
        if assets.restrict_brands_csv:
            stage2_command.extend(["--restrict-brands-csv", assets.restrict_brands_csv])
        _stream_command(stage2_command, cwd=REPO_ROOT, log_path=run_dir / "stage2.log")
        if not _prediction_csv_is_complete(prediction_path, manifest_rows):
            raise RuntimeError("Stage 2 reported success but did not write a complete predictions.csv.")

    aggregates = _write_aggregates(run_dir=run_dir, manifest_rows=manifest_rows, group_fields=group_fields)
    base_summary["status"] = "complete"
    base_summary["outputs"] = {
        "detection_csv": str(detection_csv),
        "stage2_summary": str(stage2_dir / "summary.json"),
        **aggregates,
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(base_summary, handle, ensure_ascii=False, indent=2)
    print(f"[experiment] complete: {run_dir}")
    return run_dir

def execute_from_existing_predictions(
    *,
    experiment_name: str,
    experiment_dir: str | Path,
    records: Sequence[Mapping[str, object]],
    args,
    source_overall_run: str | Path,
    group_fields: Sequence[str] = (),
    protocol: Mapping[str, object] | None = None,
) -> Path:
    """Materialize an exact subset result from a completed full-dataset run.

    Scores and decisions do not depend on an experiment label.  Reusing predictions
    from a verified full run is therefore mathematically equivalent to rerunning the
    same fixed model on that subset, while preventing duplicate GPU work.
    """
    source_dir = Path(source_overall_run).resolve()
    source_summary_path = source_dir / "experiment_summary.json"
    source_predictions_path = source_dir / "stage2" / "predictions.csv"
    if not source_summary_path.is_file() or not source_predictions_path.is_file():
        raise FileNotFoundError(
            "A reusable full run needs experiment_summary.json and stage2/predictions.csv: "
            f"{source_dir}"
        )
    source_summary = _read_json(source_summary_path)
    source_decision = dict(source_summary.get("fixed_decision") or {})
    expected_decision = {"image_weight": 0.8, "text_weight": 0.2, "threshold": 0.62}
    if source_decision != expected_decision:
        raise ValueError(
            "The supplied full run does not use the frozen VLBrandPhish decision rule: "
            f"{source_decision}"
        )

    experiment_dir = Path(experiment_dir).resolve()
    if args.out_dir:
        run_dir = Path(args.out_dir).resolve()
    else:
        suffix = str(args.run_name or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = experiment_dir / "results" / f"{experiment_name}_{suffix}"
    run_preexisted = run_dir.exists() and any(run_dir.iterdir())
    if run_preexisted and not bool(args.resume):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty experiment directory: {run_dir}. "
            "Pass --resume only for the same frozen protocol."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "input_manifest.csv"
    if run_preexisted:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without input_manifest.csv: {run_dir}")
        manifest_rows = read_manifest(manifest_path)
        _validate_resume_manifest(manifest_rows=manifest_rows, requested_records=records)
        print(f"[experiment] verified resume manifest: {manifest_path}")
    else:
        manifest_path = write_manifest(records, manifest_path)
        manifest_rows = read_manifest(manifest_path)
    wanted = {str(row["sample_key"]) for row in manifest_rows}
    with source_predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    subset_rows = [row for row in source_rows if str(row.get("sample_key", "")) in wanted]
    found = {str(row.get("sample_key", "")) for row in subset_rows}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(
            "The source full run does not contain every selected subset sample; "
            f"first missing keys: {missing[:5]}"
        )
    subset_rows.sort(key=lambda row: str(row.get("sample_key", "")))

    stage2_dir = run_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = stage2_dir / "predictions.csv"
    fields = list(subset_rows[0]) if subset_rows else []
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subset_rows)
    reused_stage2_summary = {
        "status": "reused_full_prediction",
        "source_overall_run": str(source_dir),
        "source_predictions": str(source_predictions_path),
        "n_samples": len(subset_rows),
        "decision": expected_decision,
    }
    with (stage2_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(reused_stage2_summary, handle, ensure_ascii=False, indent=2)

    aggregates = _write_aggregates(run_dir=run_dir, manifest_rows=manifest_rows, group_fields=group_fields)
    summary = {
        "experiment": experiment_name,
        "status": "complete_reused_full_prediction",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_manifest": str(manifest_path),
        "counts": {
            "n_samples": len(manifest_rows),
            "n_phish": sum(_is_true(row.get("gt_is_phishing")) for row in manifest_rows),
            "n_benign": sum(not _is_true(row.get("gt_is_phishing")) for row in manifest_rows),
        },
        "protocol": dict(protocol or {}),
        "fixed_decision": expected_decision,
        "reused_from": {
            "overall_run": str(source_dir),
            "stage1_detection_csv": str(source_dir / "stage1" / "logo_detection_top3.csv"),
            "stage2_predictions": str(source_predictions_path),
            "source_assets": source_summary.get("assets", {}),
        },
        "outputs": {"stage2_summary": str(stage2_dir / "summary.json"), **aggregates},
    }
    with (run_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[experiment] complete via exact full-run subset: {run_dir}")
    return run_dir
