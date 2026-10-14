"""Provide the Stage-1 logo-detection command without importing MMDetection at module import time."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.runtime.paths import default_run_dir, resolve_default_paths
from inference.data.detection_csv import DetectionSample, write_detection_top3_csv
from inference.data.testb_dataset import TestBSample, load_testb_samples
from inference.runtime.progress import iter_with_progress



def _to_detection_row(
    sample: TestBSample,
    dets: List[Tuple[Tuple[float, float, float, float], float]],
) -> DetectionSample:
    bboxes = [bbox for bbox, _ in dets]
    scores = [float(score) for _, score in dets]
    return DetectionSample(
        image_path=sample.image_path,
        url=sample.url,
        sample_key=sample.sample_key,
        gt_is_phishing=sample.gt_is_phishing,
        gt_brand=sample.gt_brand,
        bboxes=bboxes,
        det_scores=scores,
    )


def main() -> None:
    """Run Stage 1 and write a detector-independent Top-3 CSV for Stage 2."""
    defaults = resolve_default_paths(__file__)
    parser = argparse.ArgumentParser(description="Generate top-k logo detections for Test-B.")
    parser.add_argument("--det-config", type=str, required=True, help="MMDetection config path.")
    parser.add_argument("--det-checkpoint", type=str, default=defaults.detector_ckpt)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det-score-thr", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=1)

    parser.add_argument("--benign-root", type=str, default=defaults.benign_root)
    parser.add_argument("--phish-root", type=str, default=defaults.phish_root)
    parser.add_argument("--max-benign", type=int, default=-1)
    parser.add_argument("--max-phish", type=int, default=-1)

    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--out-base-dir", type=str, default=os.path.join(defaults.phishkiller_root, "outputs"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    out_csv = str(args.out_csv).strip()
    if not out_csv:
        run_dir = default_run_dir(str(args.out_base_dir), "detection")
        os.makedirs(run_dir, exist_ok=True)
        out_csv = os.path.join(run_dir, "logo_detection_top3.csv")
    out_csv = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    samples = load_testb_samples(
        benign_root=os.path.abspath(args.benign_root),
        phish_root=os.path.abspath(args.phish_root),
        max_benign=int(args.max_benign),
        max_phish=int(args.max_phish),
    )
    print(f"[detection] loaded samples={len(samples)}")

    from .detector import MMDetTopKDetector

    detector = MMDetTopKDetector(
        config_path=os.path.abspath(args.det_config),
        checkpoint_path=os.path.abspath(args.det_checkpoint),
        device=str(args.device),
    )

    t0 = time.perf_counter()
    det_rows: List[DetectionSample] = []
    for sample in iter_with_progress(samples, description="logo_detection", enabled=bool(args.progress)):
        dets = detector.detect_topk(
            image_path=sample.image_path,
            score_thr=float(args.det_score_thr),
            topk=int(args.topk),
        )
        det_rows.append(_to_detection_row(sample, dets))

    write_detection_top3_csv(det_rows, out_csv)
    dt = time.perf_counter() - t0
    n_has_logo = sum(1 for row in det_rows if row.bboxes)
    print(f"[detection] done: out_csv={out_csv}")
    print(f"[detection] has_logo={n_has_logo}/{len(det_rows)} elapsed_sec={dt:.2f}")


if __name__ == "__main__":
    main()
