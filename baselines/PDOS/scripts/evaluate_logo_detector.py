"""Evaluate a trained one-class Logo detector on Default/Merged/New."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdos.detector import YOLOv5LogoDetector


IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    detector = YOLOv5LogoDetector(
        ROOT, args.weights, args.device, args.confidence, args.nms_iou,
        config=ROOT / "configs" / "pdos_yolov5s_se_logo.yaml", class_names=["logo"]
    )
    rows = []
    prediction_dump = {}
    expected_counts = {"default": 901, "merged": 2480, "new": 12801}
    for split in args.splits:
        records = manifest["splits"][split]
        if args.max_images:
            records = records[:args.max_images]
        stats = evaluate_split(detector, records)
        metrics = summarize(stats)
        metrics.update({"dataset": split, "images": len(records), "expected_images": expected_counts[split]})
        rows.append(metrics)
        if args.save_predictions:
            prediction_dump[split] = stats["predictions"]
        print(json.dumps(metrics, indent=2))

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if args.save_predictions:
        args.output.with_suffix(".predictions.json").write_text(json.dumps(prediction_dump, indent=2), encoding="utf-8")


def evaluate_split(detector, records):
    confidences = []
    true_positive = [[] for _ in IOU_THRESHOLDS]
    total_gt = 0
    total_pred = 0
    predictions = []
    total_time = 0.0
    for record in records:
        gt = read_labels(Path(record["label"]), record["width"], record["height"])
        start = time.perf_counter()
        detections = detector.predict(record["image"])
        total_time += time.perf_counter() - start
        detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        total_gt += len(gt)
        total_pred += len(detections)
        confidences.extend(item["confidence"] for item in detections)
        for index, threshold in enumerate(IOU_THRESHOLDS):
            matched = set()
            for detection in detections:
                best_iou, best_index = 0.0, -1
                for gt_index, target in enumerate(gt):
                    if gt_index not in matched:
                        overlap = iou(detection["bbox"], target)
                        if overlap > best_iou:
                            best_iou, best_index = overlap, gt_index
                is_tp = best_index >= 0 and best_iou >= threshold
                true_positive[index].append(1 if is_tp else 0)
                if is_tp:
                    matched.add(best_index)
        predictions.append({"image": record["image"], "ground_truth": gt, "detections": detections})
    return {"confidences": confidences, "true_positive": true_positive, "total_gt": total_gt, "total_pred": total_pred, "predictions": predictions, "time": total_time}


def summarize(stats):
    aps = [average_precision(tp, stats["confidences"], stats["total_gt"]) for tp in stats["true_positive"]]
    tp50 = sum(stats["true_positive"][0])
    fp50 = stats["total_pred"] - tp50
    fn50 = stats["total_gt"] - tp50
    precision = tp50 / (tp50 + fp50) if tp50 + fp50 else 0.0
    recall = tp50 / (tp50 + fn50) if tp50 + fn50 else 0.0
    return {
        "mAP50": float(aps[0]),
        "mAP50_95": float(np.mean(aps)),
        "precision@0.50IoU": precision,
        "recall@0.50IoU": recall,
        "f1@0.50IoU": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ground_truth_boxes": stats["total_gt"],
        "predicted_boxes": stats["total_pred"],
        "avg_inference_seconds": stats["time"] / max(len(stats["predictions"]), 1),
    }


def average_precision(tp, confidence, total_gt):
    if total_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(confidence))
    values = np.asarray(tp, dtype=np.float64)[order]
    recall = np.cumsum(values) / total_gt
    precision = np.cumsum(values) / np.maximum(np.arange(len(values)) + 1, 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def read_labels(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, w, h = (float(value) for value in line.split())
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-12)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "prepared_logo_detection")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=0, help="Smoke-test limit; 0 evaluates all images")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tables" / "table_logo_detection.csv")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=["default", "merged", "new"], default=["default", "merged", "new"], help="Evaluation splits")
    return parser.parse_args()


if __name__ == "__main__":
    main()
