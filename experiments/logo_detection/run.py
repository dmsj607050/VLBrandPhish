"""VLBrandPhish logo detection evaluation (COCO bbox mAP, logo class only).

Research question: RQ2
Datasets: PhishIntention val (901) / PhishPedia test (1,579)
Metrics: COCO bbox mAP@[.50:.95], mAP@.50, mAP@.75
Paper table(s): cn-logo-det
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


# ----- Dataset defaults (relative to repo root) -----
DEFAULT_DATASETS = {
    "phishintention": {
        "coco_json": str(REPO_ROOT / "data" / "phishintention" / "val_coco.json"),
        "images_root": str(REPO_ROOT / "data" / "phishintention" / "val_imgs"),
    },
    "phishpedia": {
        "coco_json": str(REPO_ROOT / "data" / "phishpedia" / "coco_test.json"),
        "images_root": str(REPO_ROOT / "data" / "phishpedia"),
    },
}


def _load_coco_gt(coco_json: str) -> dict:
    with open(coco_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_logo_annotations(gt: dict) -> Tuple[dict, List[int]]:
    logo_cat_ids = [c["id"] for c in gt["categories"] if c["name"] == "logo"]
    if not logo_cat_ids:
        raise RuntimeError("No 'logo' category found in COCO ground truth.")
    logo_anns = [a for a in gt["annotations"] if a["category_id"] in logo_cat_ids]
    filtered = {
        "images": gt["images"],
        "annotations": logo_anns,
        "categories": [c for c in gt["categories"] if c["id"] in logo_cat_ids],
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
    }
    return filtered, logo_cat_ids


def _run_detection(
    det_config: str,
    det_checkpoint: str,
    image_paths: List[str],
    device: str,
    score_thr: float,
) -> List[List[Tuple[float, float, float, float, float]]]:
    from mmdet.apis import init_detector, inference_detector

    model = init_detector(det_config, det_checkpoint, device=device)
    results: List[List[Tuple[float, float, float, float, float]]] = []
    for idx, img_path in enumerate(image_paths, start=1):
        if idx % 50 == 0:
            print(f"  [detect] {idx}/{len(image_paths)}")
        try:
            result = inference_detector(model, img_path)
        except Exception as exc:
            print(f"  [detect] WARN failed on {img_path}: {exc}")
            results.append([])
            continue
        instances = getattr(result, "pred_instances", None)
        if instances is None:
            results.append([])
            continue
        bboxes = getattr(instances, "bboxes", None)
        scores = getattr(instances, "scores", None)
        if bboxes is None or scores is None:
            results.append([])
            continue
        boxes_np = bboxes.detach().cpu().numpy().astype(np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        keep = scores_np >= float(score_thr)
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        if boxes_np.size == 0:
            results.append([])
            continue
        results.append(
            [
                (float(boxes_np[i, 0]), float(boxes_np[i, 1]),
                 float(boxes_np[i, 2]), float(boxes_np[i, 3]),
                 float(scores_np[i]))
                for i in range(len(boxes_np))
            ]
        )
    return results


def _evaluate_coco_map(gt_filtered: dict, coco_results: List[dict]) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(gt_filtered, f)
        gt_path = f.name
    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(coco_results) if coco_results else COCO()
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return {
            "mAP@[.50:.95]": float(ev.stats[0]),
            "mAP@.50": float(ev.stats[1]),
            "mAP@.75": float(ev.stats[2]),
            "mAP@.50:.95_small": float(ev.stats[3]),
            "mAP@.50:.95_medium": float(ev.stats[4]),
            "mAP@.50:.95_large": float(ev.stats[5]),
            "n_gt_logos": len(gt_filtered["annotations"]),
            "n_pred_logos": len(coco_results),
        }
    finally:
        os.unlink(gt_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VLBrandPhish logo detection evaluation (COCO bbox mAP)")
    parser.add_argument("--det-config", required=True, help="MMDetection DINO config (.py).")
    parser.add_argument("--det-checkpoint", required=True, help="DINO checkpoint (.pth).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument(
        "--dataset", choices=["phishintention", "phishpedia", "both"],
        default="phishintention",
    )
    parser.add_argument("--pi-coco-json", default=DEFAULT_DATASETS["phishintention"]["coco_json"])
    parser.add_argument("--pi-images-root", default=DEFAULT_DATASETS["phishintention"]["images_root"])
    parser.add_argument("--pp-coco-json", default=DEFAULT_DATASETS["phishpedia"]["coco_json"])
    parser.add_argument("--pp-images-root", default=DEFAULT_DATASETS["phishpedia"]["images_root"])
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, dict] = {}

    for ds_name, ds_key in [("phishintention", "pi"), ("phishpedia", "pp")]:
        if args.dataset not in (ds_name, "both"):
            continue
        coco_json = getattr(args, f"{ds_key}_coco_json")
        images_root = getattr(args, f"{ds_key}_images_root")

        print(f"\n{'='*60}")
        print(f"[{ds_name}] coco_json={coco_json}")
        print(f"[{ds_name}] images_root={images_root}")
        print(f"{'='*60}")

        gt = _load_coco_gt(coco_json)
        gt_filtered, logo_cat_ids = _filter_logo_annotations(gt)
        logo_cat_id = logo_cat_ids[0]
        print(f"[{ds_name}] images={len(gt['images'])} logo_anns={len(gt_filtered['annotations'])}")

        image_ids = [img["id"] for img in gt["images"]]
        if int(args.max_images) > 0:
            image_ids = image_ids[: int(args.max_images)]
        image_paths = [
            os.path.join(images_root, img["file_name"])
            for img in gt["images"]
            if img["id"] in image_ids
        ]

        missing = [p for p in image_paths if not os.path.isfile(p)]
        if missing:
            print(f"[{ds_name}] WARN {len(missing)} images not found (e.g. {missing[:2]})")

        t0 = time.perf_counter()
        detections = _run_detection(
            args.det_config, args.det_checkpoint, image_paths,
            args.device, float(args.score_thr),
        )
        detect_sec = time.perf_counter() - t0
        print(f"[{ds_name}] detection done in {detect_sec:.1f}s")

        coco_results: List[dict] = []
        for img_id, dets in zip(image_ids, detections):
            for (x1, y1, x2, y2, score) in dets:
                coco_results.append({
                    "image_id": int(img_id),
                    "category_id": int(logo_cat_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

        metrics = _evaluate_coco_map(gt_filtered, coco_results)
        metrics["n_images"] = len(image_ids)
        metrics["detect_sec"] = detect_sec
        all_metrics[ds_name] = metrics

    # Summary
    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("LOGO DETECTION RESULTS (COCO bbox mAP, logo class only)")
    print(f"{'='*60}")
    header = f"{'Dataset':<18} {'Images':>8} {'GT logos':>10} {'mAP@[.50:.95]':>16} {'mAP@.50':>10} {'mAP@.75':>10}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(f"{name:<18} {m['n_images']:>8} {m['n_gt_logos']:>10} "
              f"{m['mAP@[.50:.95]']:>16.4f} {m['mAP@.50']:>10.4f} {m['mAP@.75']:>10.4f}")
    print(f"\nResults: {summary_path}")


if __name__ == "__main__":
    main()
