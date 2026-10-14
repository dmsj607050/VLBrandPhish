"""Run Logo Detection on an explicit experiment manifest.

This small worker is intentionally separate from the orchestration module: it runs
inside the MMDetection environment, while Stage 2 runs inside the SigLIP
environment.  The CSV it writes is the canonical boundary between those stages.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.data.detection_csv import (
    DETECTION_TOP3_HEADER,
    DetectionSample,
    detection_to_csv_row,
    read_detection_top3_csv,
)
from inference.detection.detector import MMDetTopKDetector


def _is_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    required = {"sample_key", "image_path", "url", "gt_is_phishing", "gt_brand"}
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest has no required columns: {sorted(missing)}")
    if any(not Path(row["image_path"]).is_file() for row in rows):
        missing_paths = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
        raise FileNotFoundError(f"Manifest image paths are missing; first entries: {missing_paths[:5]}")
    return rows


def _resume_prefix(manifest: list[dict[str, str]], target: Path) -> int:
    """Return the completed prefix in an existing CSV, rejecting mixed manifests.

    Stage 1 is slow on the 8 GiB evaluation GPU.  Persisting each screenshot lets
    a later invocation continue safely after an interruption instead of losing
    hours of completed DINO detections.
    """
    if not target.is_file() or target.stat().st_size == 0:
        return 0
    previous = read_detection_top3_csv(str(target))
    if len(previous) > len(manifest):
        raise ValueError(f"Existing detection CSV has more rows than manifest: {target}")
    expected = [row["sample_key"] for row in manifest[: len(previous)]]
    observed = [row.sample_key for row in previous]
    if observed != expected:
        raise ValueError(
            "Existing detection CSV is not the completed prefix of this manifest; "
            f"refusing unsafe resume: {target}"
        )
    return len(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DINO logo detection on a frozen experiment manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-score-thr", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    manifest = _read_manifest(Path(args.manifest).resolve())
    target = Path(args.out_csv).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = _resume_prefix(manifest, target)
    previous = read_detection_top3_csv(str(target)) if completed else []
    cached_detections = {
        sample.image_path: list(zip(sample.bboxes, sample.det_scores)) for sample in previous
    }
    has_logo = sum(bool(sample.bboxes) for sample in previous)
    if completed == len(manifest):
        print(
            f"[stage1] existing CSV already complete: {target} "
            f"has_logo={has_logo}/{len(manifest)}",
            flush=True,
        )
        return

    # Multiple URL policies can refer to one screenshot (notably PhishEval).  Logo
    # localization is image-only, so one detector call is sufficient; the stored
    # boxes are copied into every URL-specific Stage-2 row.
    pending_images = {row["image_path"] for row in manifest[completed:]} - set(cached_detections)
    detector = None
    if pending_images:
        detector = MMDetTopKDetector(args.det_config, args.det_checkpoint, device=args.device)
    start = time.perf_counter()
    detector_calls = 0
    mode = "a" if completed else "w"
    with target.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_TOP3_HEADER)
        if not completed:
            writer.writeheader()
            handle.flush()
        else:
            print(f"[stage1] resuming at {completed}/{len(manifest)}", flush=True)
        for index, row in enumerate(manifest[completed:], start=completed + 1):
            image_path = row["image_path"]
            detections = cached_detections.get(image_path)
            if detections is None:
                assert detector is not None
                detections = detector.detect_topk(
                    image_path, score_thr=float(args.det_score_thr), topk=max(1, int(args.topk))
                )
                cached_detections[image_path] = detections
                detector_calls += 1
            sample = DetectionSample(
                image_path=image_path,
                url=row["url"],
                sample_key=row["sample_key"],
                gt_is_phishing=_is_true(row["gt_is_phishing"]),
                gt_brand=row["gt_brand"] or None,
                bboxes=[box for box, _ in detections],
                det_scores=[float(score) for _, score in detections],
            )
            writer.writerow(detection_to_csv_row(sample))
            handle.flush()
            has_logo += int(bool(sample.bboxes))
            if index % 25 == 0 or index == len(manifest):
                os.fsync(handle.fileno())
            if args.progress and (index == 1 or index == len(manifest) or index % 25 == 0):
                print(f"[stage1] progress {index}/{len(manifest)}", flush=True)
    print(
        f"[stage1] wrote={target} has_logo={has_logo}/{len(manifest)} "
        f"detector_calls={detector_calls} unique_images={len(cached_detections)} "
        f"elapsed_sec={time.perf_counter() - start:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
