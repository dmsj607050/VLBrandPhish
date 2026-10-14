"""Run the four PDOS experiments with reusable Stage-1/Stage-2 artifacts.

Stages:
  manifest  dataset discovery and protocol filtering (CPU)
  detect    YOLOv5s+SE Logo localization (GPU)
  recognize ResNet-101 crop-to-reference cosine matching (GPU)
  decide    dataset URL policy + final phishing label (CPU)

The CSV files are intentional interchange artifacts: copy them between the
server and local machine to avoid repeating completed GPU stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdos.detector import YOLOv5LogoDetector
from pdos.experiment_data import _first_existing_child, _load_normal_url_csv, build_records, limit_records, read_manifest, write_manifest
from pdos.feature_extractor import ResNet101FeatureExtractor
from pdos.legal_logo_db import LegalLogoDB
from pdos.preprocessing import crop_box, load_rgb_image
from pdos.url_policy import decide_url


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged YOLO+ResNet PDOS evaluation")
    parser.add_argument("--dataset", choices=["phishintention", "ours"], required=True)
    parser.add_argument("--experiment", choices=["overall_phishintention", "overall_ours", "cross_time", "cross_brand"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "experiments")
    parser.add_argument("--stage", choices=["manifest", "detect", "recognize", "decide", "all"], default="all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stage1-csv", type=Path)
    parser.add_argument("--stage2-csv", type=Path)
    parser.add_argument("--source-stage1", type=Path, help="Reuse a larger Stage-1 CSV and select rows by manifest sample_key.")
    parser.add_argument("--max-phish", type=int, default=-1)
    parser.add_argument("--max-benign", type=int, default=-1)
    parser.add_argument("--weights", type=Path, default=ROOT / "results" / "training" / "pdos_yolov5s_se_logo" / "weights" / "best.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pdos_yolov5s_se_logo.yaml")
    parser.add_argument("--resnet-weights", type=Path, default=ROOT / "models" / "resnet101-cd907fc2.pth")
    parser.add_argument("--logo-library-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--topk", type=int, default=1, help="Logo detections used for recognition (paper comparison protocol: top-1).")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--max-reference-per-brand", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--recognize-batch-pages", type=int, default=16, help="Pages per batched ResNet/reference-scoring step.")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    return (args.output_root / args.experiment).resolve()


def _manifest_path(args: argparse.Namespace) -> Path:
    return (args.manifest or (_run_dir(args) / "manifest.csv")).resolve()


def _stage1_path(args: argparse.Namespace) -> Path:
    return (args.stage1_csv or (_run_dir(args) / "stage1" / "logo_detection.csv")).resolve()


def _stage2_path(args: argparse.Namespace) -> Path:
    return (args.stage2_csv or (_run_dir(args) / "stage2" / "brand_similarity.csv")).resolve()


def _make_manifest(args: argparse.Namespace) -> Path:
    records, _ = build_records(args.dataset, args.dataset_root, args.experiment)
    records = limit_records(records, args.max_phish, args.max_benign)
    path = write_manifest(records, _manifest_path(args))
    counts = {"phish": sum(row["gt_is_phishing"] == "1" for row in records), "benign": sum(row["gt_is_phishing"] == "0" for row in records)}
    print(json.dumps({"manifest": str(path), "samples": len(records), **counts}, ensure_ascii=False))
    return path


def _read_stage1_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def _write_stage1_row(writer: csv.DictWriter, manifest: dict[str, str], detections: list[dict[str, object]], topk: int) -> None:
    row = dict(manifest)
    row["has_logo"] = "1" if detections else "0"
    for index, detection in enumerate(detections[:topk], 1):
        x1, y1, x2, y2 = detection["bbox"]
        row[f"det_score_{index}"] = detection["confidence"]
        row[f"x1_{index}"] = x1
        row[f"y1_{index}"] = y1
        row[f"x2_{index}"] = x2
        row[f"y2_{index}"] = y2
    writer.writerow(row)


def _run_detect(args: argparse.Namespace, manifest_path: Path) -> Path:
    manifests = {row["sample_key"]: row for row in read_manifest(manifest_path)}
    source_rows = _read_stage1_rows(args.source_stage1) if args.source_stage1 else None
    if source_rows is not None:
        selected = [row for row in source_rows if row.get("sample_key") in manifests]
        if len(selected) != len(manifests):
            raise ValueError(f"Source Stage-1 CSV is missing {len(manifests) - len(selected)} manifest rows.")
        path = _stage1_path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(selected[0]) if selected else []
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(selected, key=lambda row: row["sample_key"]))
        print(f"Reused Stage-1 rows: {len(selected)} -> {path}")
        return path

    detector = YOLOv5LogoDetector(ROOT, args.weights, args.device, args.confidence, args.iou, config=args.config, class_names=["logo"])
    path = _stage1_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = ["has_logo"] + [field for index in range(1, args.topk + 1) for field in (f"det_score_{index}", f"x1_{index}", f"y1_{index}", f"x2_{index}", f"y2_{index}")]
    fields = list(next(iter(manifests.values()))) + extra_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        started = time.time()
        for index, row in enumerate(manifests.values(), 1):
            detections = detector.predict(row["image_path"])
            _write_stage1_row(writer, row, detections, args.topk)
            if index == 1 or index % max(1, args.progress_every) == 0 or index == len(manifests):
                print(f"Stage 1 {index}/{len(manifests)} ({time.time() - started:.1f}s)", flush=True)
    print(f"Stage 1 saved: {path}")
    return path


def _reference_brands(args: argparse.Namespace, root: Path) -> tuple[LegalLogoDB, list[str]]:
    if args.logo_library_root:
        library = args.logo_library_root
    elif args.dataset == "ours":
        candidates = (
            root / "03_logo_reference_library" / "logo_reference_library",
            root / "logo_reference_library",
            root / "logo_reference_library_all",
        )
        library = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
    else:
        library = root / "277_brand_pages"
    db = LegalLogoDB(library)
    brands = db.brands()
    if not brands:
        raise FileNotFoundError(f"No reference-brand directories found under {library}")
    if args.experiment in {"cross_time", "cross_brand"} and args.dataset == "ours":
        overlap = _first_existing_child(root, ("08_brands_overlap.csv", "brands_overlap.csv", "08_brand_overlap.csv"))
        allowed: set[str] = set()
        with overlap.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                flag = str(row.get("in_277", "") or "").strip()
                if (args.experiment == "cross_time" and flag == "1") or (args.experiment == "cross_brand" and flag == "0"):
                    allowed.add(str(row.get("brand_dir_7299", "") or "").strip())
        brands = [brand for brand in brands if brand in allowed]
    if not brands:
        raise RuntimeError("The selected experiment has no usable reference brands.")
    return db, brands


def _run_recognize(args: argparse.Namespace, manifest_path: Path, stage1_path: Path) -> Path:
    rows = _read_stage1_rows(stage1_path)
    expected = {row["sample_key"] for row in read_manifest(manifest_path)}
    rows = [row for row in rows if row.get("sample_key") in expected]
    if len(rows) != len(expected):
        raise ValueError("Stage-1 and manifest sample sets do not match.")
    db, brands = _reference_brands(args, args.dataset_root.resolve())
    extractor = ResNet101FeatureExtractor(args.device, pretrained=True, weights_path=args.resnet_weights)
    reference_items: list[tuple[str, Path, torch.Tensor]] = []
    print(f"Encoding {len(brands)} reference brands...")
    skipped_reference_images = 0
    for brand in brands:
        paths = db.paths_for(brand)
        if args.max_reference_per_brand >= 0:
            paths = paths[:args.max_reference_per_brand]
        for path in paths:
            try:
                reference_items.append((brand, path, extractor.extract_path(path)))
            except (OSError, ValueError) as error:
                skipped_reference_images += 1
                print(f"Skipping unreadable reference image: {path} ({error})", flush=True)
    if skipped_reference_images:
        print(f"Skipped {skipped_reference_images} unreadable reference image(s).", flush=True)
    if not reference_items:
        raise RuntimeError("No readable reference images found in the selected logo library.")

    # The original implementation normalized and compared each reference image
    # separately for every crop. Flattening the same sorted (brand, path) order
    # and taking one matrix maximum is mathematically equivalent, with much less
    # Python/torch-call overhead. Ties still select the first sorted reference.
    reference_matrix = F.normalize(torch.stack([item[2] for item in reference_items]), p=2, dim=1)
    print(f"Prepared {len(reference_items)} reference images across {len({item[0] for item in reference_items})} brands.", flush=True)

    path = _stage2_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["predicted_brand", "cosine_similarity", "similarity_threshold", "logo_is_real", "reference_logo", "best_det_score", "best_x1", "best_y1", "best_x2", "best_y2"]
    batch_pages = max(1, int(args.recognize_batch_pages))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        started = time.time()
        for start in range(0, len(rows), batch_pages):
            page_rows = rows[start:start + batch_pages]
            crops: list[object] = []
            crop_metadata: list[tuple[int, float, list[float]]] = []
            for page_index, row in enumerate(page_rows):
                has_candidate = False
                for det_index in range(1, args.topk + 1):
                    score_raw = row.get(f"det_score_{det_index}", "")
                    if not score_raw or not row.get(f"x1_{det_index}", ""):
                        continue
                    if not has_candidate:
                        image = load_rgb_image(row["image_path"])
                        has_candidate = True
                    box = [float(row[f"{axis}_{det_index}"]) for axis in ("x1", "y1", "x2", "y2")]
                    crops.append(crop_box(image, box, padding=0.05))
                    crop_metadata.append((page_index, float(score_raw), box))

            best_by_page: list[dict[str, object] | None] = [None] * len(page_rows)
            if crops:
                query_features = extractor.extract_batch(crops)
                query_matrix = F.normalize(query_features, p=2, dim=1)
                similarities = query_matrix @ reference_matrix.T
                best_scores, best_reference_indices = similarities.max(dim=1)
                scores = best_scores.tolist()
                reference_indices = best_reference_indices.tolist()
                for crop_index, (page_index, det_score, box) in enumerate(crop_metadata):
                    score = float(scores[crop_index])
                    reference_brand, reference_path, _ = reference_items[reference_indices[crop_index]]
                    candidate = {
                        "brand": reference_brand,
                        "similarity": score,
                        "reference_logo": str(reference_path),
                        "det_score": det_score,
                        "box": box,
                    }
                    current = best_by_page[page_index]
                    if current is None or score > float(current["similarity"]):
                        best_by_page[page_index] = candidate

            for row, best in zip(page_rows, best_by_page):
                output = dict(row)
                output.update({
                    "predicted_brand": best["brand"] if best else "",
                    "cosine_similarity": best["similarity"] if best else "",
                    "similarity_threshold": args.threshold,
                    "logo_is_real": int(bool(best and float(best["similarity"]) >= args.threshold)),
                    "reference_logo": best["reference_logo"] if best else "",
                    "best_det_score": best["det_score"] if best else "",
                    "best_x1": best["box"][0] if best else "",
                    "best_y1": best["box"][1] if best else "",
                    "best_x2": best["box"][2] if best else "",
                    "best_y2": best["box"][3] if best else "",
                })
                writer.writerow(output)
            completed = min(start + len(page_rows), len(rows))
            if completed == batch_pages or completed % max(1, args.progress_every) < batch_pages or completed == len(rows):
                print(f"Stage 2 {completed}/{len(rows)} ({time.time() - started:.1f}s)", flush=True)
    print(f"Stage 2 saved: {path}")
    return path


def _run_decide(args: argparse.Namespace, stage2_path: Path) -> Path:
    rows = _read_stage1_rows(stage2_path)
    # Stage 3 is CPU-only.  Do not rescan tens of thousands of screenshots just
    # to recover the same URL map that is already a small CSV asset.
    if args.dataset == "ours":
        normal_csv = _first_existing_child(args.dataset_root, ("07_normal_urls_testb.csv", "normal_urls_testb.csv", "normal_urls.csv"))
    else:
        normal_csv = _first_existing_local(args.dataset_root / "normal_urls.csv", args.dataset_root.parent / "normal_urls.csv")
    if not normal_csv.is_file():
        raise FileNotFoundError(f"Official URL mapping not found: {normal_csv}")
    _, official_urls = _load_normal_url_csv(normal_csv)
    output_path = _run_dir(args) / "stage3" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["url_policy_mode", "url_accept_phishing", "url_reason", "registered_domain", "final_is_phishing", "final_label", "final_reason"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            brand_key = str(row.get("predicted_brand", "") or "").strip().lower()
            hosts = official_urls.get(brand_key, [])
            mode = row.get("url_policy_mode", "phishintention" if args.dataset == "phishintention" else "official_plus_overrides_no_whitelist")
            url_result = decide_url(row.get("url", ""), hosts, mode)
            logo_is_real = row.get("logo_is_real", "0") == "1"
            final_is_phishing = not (logo_is_real and not url_result.accept)
            if final_is_phishing:
                reason = "logo_similarity_failed" if not logo_is_real else "url_brand_inconsistent"
            else:
                reason = "verified_logo_and_url"
            output = dict(row)
            output.update({
                "url_policy_mode": mode,
                "url_accept_phishing": int(url_result.accept),
                "url_reason": url_result.reason,
                "registered_domain": url_result.registered_domain,
                "final_is_phishing": int(final_is_phishing),
                "final_label": "phishing" if final_is_phishing else "legitimate",
                "final_reason": reason,
            })
            writer.writerow(output)
    _write_summary(output_path)
    print(f"Stage 3 saved: {output_path}")
    return output_path


def _first_existing_local(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def _write_summary(predictions_path: Path) -> None:
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    def positive(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}
    tp = sum(positive(row["gt_is_phishing"]) and positive(row["final_is_phishing"]) for row in rows)
    tn = sum(not positive(row["gt_is_phishing"]) and not positive(row["final_is_phishing"]) for row in rows)
    fp = sum(not positive(row["gt_is_phishing"]) and positive(row["final_is_phishing"]) for row in rows)
    fn = sum(positive(row["gt_is_phishing"]) and not positive(row["final_is_phishing"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    summary = {"images": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "accuracy": accuracy}
    (predictions_path.parent.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = _args()
    manifest = _manifest_path(args)
    if args.stage in {"manifest", "all"} or not manifest.is_file():
        manifest = _make_manifest(args)
    if args.stage == "manifest":
        return
    stage1 = _stage1_path(args)
    if args.stage in {"detect", "all"}:
        stage1 = _run_detect(args, manifest)
    elif not stage1.is_file():
        raise FileNotFoundError(f"Stage-1 CSV not found: {stage1}")
    if args.stage == "detect":
        return
    stage2 = _stage2_path(args)
    if args.stage in {"recognize", "all"}:
        stage2 = _run_recognize(args, manifest, stage1)
    elif not stage2.is_file():
        raise FileNotFoundError(f"Stage-2 CSV not found: {stage2}")
    if args.stage == "recognize":
        return
    _run_decide(args, stage2)


if __name__ == "__main__":
    main()
