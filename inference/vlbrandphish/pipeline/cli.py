from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import warnings
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..runtime.paths import default_run_dir, resolve_default_paths
from ..components.brand.naming import normalize_brand_key
from ..data.contracts import DetectionSample, read_detection_top3_csv
from ..decision.domain import phishintention_domain_accept, registered_domain
from ..components.brand.encoders import CLIPEncoder
from ..components.brand.repository import BrandRepository, build_brand_repository, score_query_features
from ..runtime.types import AlphaOutput, Metrics, PrecomputedData, QueryFeatureData
from ..recognition.features import (
    _compute_alpha_output,
    _compute_similarity_from_query_features,
    _precompute_query_features,
)
from ..evaluation.metrics import (
    _build_brand_level_reports,
    _build_prediction_rows,
    _evaluate_threshold,
    _objective_value,
    _parse_grid_values,
)
from ..data.image_paths import _inspect_detection_csv_format, _iter_items
from ..recognition.resources import load_brand_resources

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# Cache serialization and signatures are maintained in their own module. The
# aliases retain the existing internal call surface while downstream scripts
# migrate to ``inference.cache`` directly.
from ..runtime.cache import (
    _build_feature_cache_key,
    _build_query_cache_key,
    _build_repo_cache_key,
    _find_compatible_query_cache,
    _hash_dir_tree_signature,
    _hash_file_signature,
    _load_brand_repo_cache,
    _load_precomputed_cache,
    _load_query_cache,
    _save_brand_repo_cache,
    _save_precomputed_cache,
    _save_query_cache,
    _slice_query_data_topk,
    _truncate_detections_topk,
)


def _print_runtime_format_info(
    *,
    det_csv: str,
    csv_info: Dict[str, object],
    detections: List[DetectionSample],
    repo: BrandRepository,
    args: argparse.Namespace,
    alpha_grid: List[float],
    gate_k_grid: List[float],
    gate_b_grid: List[float],
    n_fusion_candidates: int,
    threshold_grid: List[float],
) -> None:
    n = len(detections)
    n_phish = sum(1 for d in detections if bool(d.gt_is_phishing))
    n_benign = int(n - n_phish)
    n_has_logo = sum(1 for d in detections if len(d.bboxes) > 0)
    n_box1 = sum(1 for d in detections if len(d.bboxes) >= 1)
    n_box2 = sum(1 for d in detections if len(d.bboxes) >= 2)
    n_box3 = sum(1 for d in detections if len(d.bboxes) >= 3)
    n_empty_url = sum(1 for d in detections if not str(d.url or "").strip())

    logo_count_per_brand = repo.logo_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().tolist()
    text_count_per_brand = repo.text_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().tolist()
    avg_logo_per_brand = float(sum(int(x) for x in logo_count_per_brand) / max(1, len(logo_count_per_brand)))
    avg_text_per_brand = float(sum(int(x) for x in text_count_per_brand) / max(1, len(text_count_per_brand)))

    print("[format] ===== Runtime Format Info =====")
    print(f"[format] det_csv={det_csv}")
    print(
        "[format] csv_structure: "
        f"rows={int(csv_info.get('row_count', 0))} "
        f"has_header={int(bool(csv_info.get('has_header', False)))} "
        f"first_row_ncols={int(csv_info.get('first_row_ncols', 0))} "
        f"min_cols={int(csv_info.get('min_cols', 0))} "
        f"max_cols={int(csv_info.get('max_cols', 0))}"
    )
    print(f"[format] csv_first_row_preview={csv_info.get('first_row_preview', [])}")
    print(
        "[format] samples_after_max_samples: "
        f"n={n} benign={n_benign} phish={n_phish} "
        f"has_logo={n_has_logo} box1={n_box1} box2={n_box2} box3={n_box3} "
        f"empty_url={n_empty_url}"
    )
    print(
        "[format] repository: "
        f"brands={len(repo.brand_names)} "
        f"logo_emb_shape={tuple(repo.logo_emb.shape)} "
        f"text_emb_shape={tuple(repo.text_emb.shape)} "
        f"avg_logo_per_brand={avg_logo_per_brand:.2f} "
        f"avg_text_per_brand={avg_text_per_brand:.2f}"
    )
    print(
        "[format] run_config: "
        f"fusion_method={str(args.fusion_method)} "
        f"alpha={float(args.alpha):.4f} "
        f"gate_k={float(args.gate_k):.4f} gate_b={float(args.gate_b):.4f} "
        f"threshold={float(args.threshold):.4f} "
        f"max_logo_per_brand={int(args.max_logo_per_brand)} "
        f"image_batch_size={int(args.image_batch_size)} "
        f"text_batch_size={int(args.text_batch_size)} "
        f"device={str(args.device)} progress={int(bool(args.progress))}"
    )
    print(
        "[format] search_config: "
        f"grid_search={int(bool(args.grid_search))} objective={str(args.grid_objective)} "
        f"n_alpha={len(alpha_grid)} n_gate_k={len(gate_k_grid)} n_gate_b={len(gate_b_grid)} "
        f"n_fusion_candidates={int(n_fusion_candidates)} n_threshold={len(threshold_grid)}"
    )
    print("[format] =================================")


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*Torch was not compiled with flash attention.*",
        category=UserWarning,
    )

    defaults = resolve_default_paths(__file__)

    parser = argparse.ArgumentParser(description="End-to-end brand matching and phishing decision.")
    parser.add_argument("--det-csv", type=str, default=defaults.det_csv)
    parser.add_argument("--brand-ckpt", type=str, default=defaults.brand_ckpt)
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Local Hugging Face SigLIP directory; overrides a stale checkpoint model_name path.",
    )
    parser.add_argument("--logo-library-dir", type=str, default=defaults.logo_library_dir)
    parser.add_argument("--dbi5-dir", type=str, default=defaults.dbi5_dir)
    parser.add_argument("--normal-urls-csv", type=str, default=defaults.normal_urls_csv)
    parser.add_argument("--benign-root", type=str, default=defaults.benign_root)
    parser.add_argument("--phish-root", type=str, default=defaults.phish_root)
    parser.add_argument(
        "--brand-text-mode",
        type=str,
        default="dbi5",
        choices=["dbi5", "train_templates_mean", "train12_mean", "train_templates"],
        help="How the recognizer builds text prompts for brand prototypes.",
    )

    parser.add_argument(
        "--fusion-method",
        type=str,
        default="dynamic_margin_gate",
        choices=["dynamic_margin_gate", "linear_alpha", "router_gate"],
        help="Fusion strategy for combining image/text brand similarities.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gate-k", type=float, default=2.0, help="Dynamic gate slope for dynamic_margin_gate.")
    parser.add_argument("--gate-b", type=float, default=0.0, help="Dynamic gate bias for dynamic_margin_gate.")
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--topk", type=int, default=3, help="How many detector boxes to keep per sample.")
    parser.add_argument("--max-logo-per-brand", type=int, default=-1)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--alpha-grid", type=str, default="", help="e.g. '0:1:0.1' or '0.2,0.4,0.6'")
    parser.add_argument("--gate-k-grid", type=str, default="", help="e.g. '0,1,2,5' for dynamic_margin_gate.")
    parser.add_argument("--gate-b-grid", type=str, default="", help="e.g. '-2,-1,0,1' for dynamic_margin_gate.")
    parser.add_argument("--threshold-grid", type=str, default="", help="e.g. '0.4:0.9:0.01' or '0.55,0.58,0.60'")
    parser.add_argument(
        "--grid-objective",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "precision", "recall", "brand_acc_on_phish"],
    )

    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--out-base-dir", type=str, default=os.path.join(defaults.phishkiller_root, "outputs"))
    parser.add_argument("--cache-dir", type=str, default=os.path.join(defaults.phishkiller_root, "outputs", "cache"))
    parser.add_argument("--no-repo-cache", action="store_true")
    parser.add_argument("--rebuild-repo-cache", action="store_true")
    parser.add_argument("--no-query-cache", action="store_true")
    parser.add_argument("--rebuild-query-cache", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", help="Enable progress bars.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars.")
    parser.set_defaults(progress=True)
    args = parser.parse_args()

    out_dir = str(args.out_dir).strip()
    if not out_dir:
        out_dir = default_run_dir(str(args.out_base_dir), "inference")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.abspath(str(args.cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    det_csv = os.path.abspath(args.det_csv)
    csv_info = _inspect_detection_csv_format(det_csv)
    detections = read_detection_top3_csv(det_csv)
    if int(args.max_samples) > 0:
        detections = detections[: int(args.max_samples)]
    detections = _truncate_detections_topk(detections, int(args.topk))
    print(f"[inference] loaded detections={len(detections)} from {det_csv}")

    brand_ckpt_abs = os.path.abspath(args.brand_ckpt)
    logo_library_dir_abs = os.path.abspath(args.logo_library_dir)
    dbi5_dir_abs = os.path.abspath(args.dbi5_dir)
    normal_urls_csv_abs = os.path.abspath(args.normal_urls_csv)
    benign_root_abs = os.path.abspath(args.benign_root)
    phish_root_abs = os.path.abspath(args.phish_root)
    encoder, repo, repo_cache_key, repo_cache_hit, repo_cache_path = load_brand_resources(
        brand_checkpoint=brand_ckpt_abs,
        logo_library_dir=logo_library_dir_abs,
        dbi5_dir=dbi5_dir_abs,
        normal_urls_csv=normal_urls_csv_abs,
        cache_dir=cache_dir,
        device=str(args.device),
        model_dir=str(args.model_dir or ""),
        max_logo_per_brand=int(args.max_logo_per_brand),
        image_batch_size=int(args.image_batch_size),
        text_batch_size=int(args.text_batch_size),
        brand_text_mode=str(args.brand_text_mode),
        use_repository_cache=not bool(args.no_repo_cache),
        rebuild_repository_cache=bool(args.rebuild_repo_cache),
    )
    if bool(args.grid_search):
        alpha_grid = _parse_grid_values(str(args.alpha_grid), fallback=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        gate_k_grid = _parse_grid_values(str(args.gate_k_grid), fallback=[0.0, 1.0, 2.0, 5.0])
        gate_b_grid = _parse_grid_values(str(args.gate_b_grid), fallback=[-2.0, -1.0, 0.0, 1.0])
        threshold_grid = _parse_grid_values(str(args.threshold_grid), fallback=[round(0.40 + 0.01 * i, 2) for i in range(51)])
    else:
        alpha_grid = [float(args.alpha)]
        gate_k_grid = [float(args.gate_k)]
        gate_b_grid = [float(args.gate_b)]
        threshold_grid = [float(args.threshold)]

    if str(args.fusion_method) == "dynamic_margin_gate":
        fusion_candidates = [
            {
                "fusion_method": "dynamic_margin_gate",
                "alpha": float(args.alpha),
                "gate_k": float(k),
                "gate_b": float(b),
            }
            for k in gate_k_grid
            for b in gate_b_grid
        ]
    elif str(args.fusion_method) == "router_gate":
        fusion_candidates = [
            {
                "fusion_method": "router_gate",
                "alpha": float(args.alpha),
                "gate_k": float(args.gate_k),
                "gate_b": float(args.gate_b),
            }
        ]
    else:
        fusion_candidates = [
            {
                "fusion_method": "linear_alpha",
                "alpha": float(a),
                "gate_k": float(args.gate_k),
                "gate_b": float(args.gate_b),
            }
            for a in alpha_grid
        ]

    if not fusion_candidates:
        raise RuntimeError("No valid fusion candidates built from current search configuration.")

    _print_runtime_format_info(
        det_csv=det_csv,
        csv_info=csv_info,
        detections=detections,
        repo=repo,
        args=args,
        alpha_grid=alpha_grid,
        gate_k_grid=gate_k_grid,
        gate_b_grid=gate_b_grid,
        n_fusion_candidates=int(len(fusion_candidates)),
        threshold_grid=threshold_grid,
    )

    t2 = time.perf_counter()
    query_cache_key = _build_query_cache_key(
        brand_ckpt=brand_ckpt_abs,
        det_csv=det_csv,
        max_samples=int(args.max_samples),
        topk=int(args.topk),
        benign_root=benign_root_abs,
        phish_root=phish_root_abs,
    )
    query_cache_path = os.path.join(cache_dir, f"stage2_query_{query_cache_key}.npz")
    query_cache_hit = False
    query_time = 0.0
    sim_from_query_time = 0.0

    feature_cache_key = _build_feature_cache_key(
        det_csv=det_csv,
        max_samples=int(args.max_samples),
        topk=int(args.topk),
        benign_root=benign_root_abs,
        phish_root=phish_root_abs,
        repo_cache_key=repo_cache_key,
    )
    feature_cache_path = os.path.join(cache_dir, f"stage2_precompute_{feature_cache_key}.npz")
    feature_cache_hit = False
    pre: Optional[PrecomputedData] = None
    if (not bool(args.no_feature_cache)) and (not bool(args.rebuild_feature_cache)) and os.path.isfile(feature_cache_path):
        try:
            pre = _load_precomputed_cache(feature_cache_path)
            feature_cache_hit = True
            print(f"[cache] stage2_feature hit: {feature_cache_path}")
        except Exception as e:
            print(f"[cache] stage2_feature load failed, recomputing: {type(e).__name__}: {e}")

    if pre is None:
        query_data: Optional[QueryFeatureData] = None
        tq = time.perf_counter()
        if (not bool(args.no_query_cache)) and (not bool(args.rebuild_query_cache)) and os.path.isfile(query_cache_path):
            try:
                query_data = _load_query_cache(query_cache_path)
                query_cache_hit = True
                print(f"[cache] stage2_query hit: {query_cache_path}")
            except Exception as e:
                print(f"[cache] stage2_query load failed, recomputing: {type(e).__name__}: {e}")

        if query_data is None and (not bool(args.no_query_cache)) and (not bool(args.rebuild_query_cache)):
            compat_path, compat_query = _find_compatible_query_cache(
                cache_dir=cache_dir,
                n_samples=len(detections),
                topk=max(1, int(args.topk)),
            )
            if compat_query is not None:
                query_data = _slice_query_data_topk(compat_query, int(args.topk))
                query_cache_hit = True
                print(f"[cache] stage2_query compatible hit: {compat_path}")
                try:
                    _save_query_cache(query_cache_path, query_data)
                    print(f"[cache] stage2_query promoted: {query_cache_path}")
                except Exception as e:
                    print(f"[cache] stage2_query promote failed: {type(e).__name__}: {e}")

        if query_data is None:
            query_data = _precompute_query_features(
                detections=detections,
                encoder=encoder,
                benign_root=benign_root_abs,
                phish_root=phish_root_abs,
                progress=bool(args.progress),
            )
            if not bool(args.no_query_cache):
                try:
                    _save_query_cache(query_cache_path, query_data)
                    print(f"[cache] stage2_query saved: {query_cache_path}")
                except Exception as e:
                    print(f"[cache] stage2_query save failed: {type(e).__name__}: {e}")
        query_time = time.perf_counter() - tq

        ts = time.perf_counter()
        pre = _compute_similarity_from_query_features(
            query_data=query_data,
            repo=repo,
            progress=bool(args.progress),
        )
        sim_from_query_time = time.perf_counter() - ts

        if not bool(args.no_feature_cache):
            try:
                _save_precomputed_cache(feature_cache_path, pre)
                print(f"[cache] stage2_feature saved: {feature_cache_path}")
            except Exception as e:
                print(f"[cache] stage2_feature save failed: {type(e).__name__}: {e}")

    feature_time = time.perf_counter() - t2
    print(
        "[inference] feature_precompute: "
        f"elapsed_sec={feature_time:.2f} "
        f"has_logo={int(pre.has_logo.sum())}/{len(detections)} "
        f"feature_cache_hit={int(feature_cache_hit)} "
        f"query_cache_hit={int(query_cache_hit)} "
        f"query_sec={float(query_time):.2f} "
        f"sim_sec={float(sim_from_query_time):.2f}"
    )

    search_records: List[Dict[str, object]] = []
    best_obj = -1.0
    best_alpha = float(fusion_candidates[0]["alpha"])
    best_fusion_method = str(fusion_candidates[0]["fusion_method"])
    best_gate_k = float(fusion_candidates[0]["gate_k"])
    best_gate_b = float(fusion_candidates[0]["gate_b"])
    best_threshold = float(threshold_grid[0])
    best_metrics = Metrics()
    best_alpha_out: Optional[AlphaOutput] = None
    best_pred_vec: Optional[np.ndarray] = None
    screenshot_urls = [str(d.url or "") for d in detections]

    fusion_iter = _iter_items(fusion_candidates, desc="grid_fusion", enabled=bool(args.progress))
    search_t0 = time.perf_counter()
    for cand in fusion_iter:
        alpha_out = _compute_alpha_output(
            fusion_method=str(cand["fusion_method"]),
            alpha=float(cand["alpha"]),
            gate_k=float(cand["gate_k"]),
            gate_b=float(cand["gate_b"]),
            pre=pre,
            repo=repo,
            screenshot_urls=screenshot_urls,
        )
        for threshold in threshold_grid:
            metrics, pred_vec = _evaluate_threshold(pre=pre, alpha_out=alpha_out, threshold=float(threshold))
            obj = _objective_value(metrics, objective=str(args.grid_objective))
            m = metrics.to_dict()
            search_records.append(
                {
                    "fusion_method": str(cand["fusion_method"]),
                    "alpha": float(cand["alpha"]),
                    "alpha_mean": float(alpha_out.alpha),
                    "gate_k": float(cand["gate_k"]),
                    "gate_b": float(cand["gate_b"]),
                    "threshold": float(threshold),
                    "objective": str(args.grid_objective),
                    "objective_value": float(obj),
                    "metrics": m,
                }
            )

            if obj > best_obj:
                best_obj = float(obj)
                best_alpha = float(alpha_out.alpha)
                best_fusion_method = str(cand["fusion_method"])
                best_gate_k = float(cand["gate_k"])
                best_gate_b = float(cand["gate_b"])
                best_threshold = float(threshold)
                best_metrics = metrics
                best_alpha_out = alpha_out
                best_pred_vec = pred_vec
    search_time = time.perf_counter() - search_t0

    if best_alpha_out is None or best_pred_vec is None:
        raise RuntimeError("Search produced no valid result")

    rows = _build_prediction_rows(
        detections=detections,
        pre=pre,
        alpha_out=best_alpha_out,
        threshold=float(best_threshold),
        pred_vec=best_pred_vec,
    )

    pred_csv_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    brand_level_rows, brand_level_overview = _build_brand_level_reports(
        rows=rows,
        threshold=float(best_threshold),
    )
    brand_level_csv_path = os.path.join(out_dir, "brand_level_diagnosis.csv")
    brand_level_json_path = os.path.join(out_dir, "brand_level_diagnosis.json")
    if brand_level_rows:
        with open(brand_level_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(brand_level_rows[0].keys()))
            writer.writeheader()
            for r in brand_level_rows:
                writer.writerow(r)
    else:
        with open(brand_level_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
    with open(brand_level_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overview": brand_level_overview,
                "brands": brand_level_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics_dict = best_metrics.to_dict()
    summary = {
        "config": {
            "det_csv": det_csv,
            "brand_ckpt": os.path.abspath(args.brand_ckpt),
            "model_dir": os.path.abspath(args.model_dir) if args.model_dir else "",
            "logo_library_dir": logo_library_dir_abs,
            "dbi5_dir": dbi5_dir_abs,
            "normal_urls_csv": normal_urls_csv_abs,
            "brand_text_mode": str(args.brand_text_mode),
            "benign_root": benign_root_abs,
            "phish_root": phish_root_abs,
            "max_samples": int(args.max_samples),
            "topk": int(args.topk),
            "max_logo_per_brand": int(args.max_logo_per_brand),
            "image_batch_size": int(args.image_batch_size),
            "text_batch_size": int(args.text_batch_size),
            "device": str(args.device),
        },
        "selection": {
            "grid_search": bool(args.grid_search),
            "grid_objective": str(args.grid_objective),
            "fusion_method": str(best_fusion_method),
            "best_alpha": float(best_alpha),
            "best_gate_k": float(best_gate_k),
            "best_gate_b": float(best_gate_b),
            "best_threshold": float(best_threshold),
            "best_objective_value": float(best_obj),
            "n_alpha": int(len(alpha_grid)),
            "n_gate_k": int(len(gate_k_grid)),
            "n_gate_b": int(len(gate_b_grid)),
            "n_fusion_candidates": int(len(fusion_candidates)),
            "n_threshold": int(len(threshold_grid)),
            "n_combinations": int(len(fusion_candidates) * len(threshold_grid)),
        },
        "runtime": {
            "feature_precompute_sec": float(feature_time),
            "query_precompute_sec": float(query_time),
            "similarity_from_query_sec": float(sim_from_query_time),
            "grid_eval_sec": float(search_time),
            "avg_sample_feature_sec": float(feature_time / max(1, len(rows))),
            "query_cache_hit": int(query_cache_hit),
            "feature_cache_hit": int(feature_cache_hit),
        },
        "cache": {
            "cache_dir": cache_dir,
            "repo_cache_path": repo_cache_path,
            "query_cache_path": query_cache_path,
            "feature_cache_path": feature_cache_path,
            "repo_cache_hit": int(repo_cache_hit),
            "query_cache_hit": int(query_cache_hit),
            "feature_cache_hit": int(feature_cache_hit),
        },
        "counts": {
            "n_samples": int(len(rows)),
            "n_with_logo": int(pre.has_logo.sum()),
            "n_pred_phish": int(np.sum(best_pred_vec.astype(np.uint8))),
        },
        "metrics": metrics_dict,
        "brand_level_overview": brand_level_overview,
        "path_resolution": dict(pre.resolve_stats),
        "outputs": {
            "predictions_csv": pred_csv_path,
            "brand_level_diagnosis_csv": brand_level_csv_path,
            "brand_level_diagnosis_json": brand_level_json_path,
        },
    }

    if bool(args.grid_search):
        grid_path = os.path.join(out_dir, "grid_results.json")
        with open(grid_path, "w", encoding="utf-8") as f:
            json.dump(search_records, f, ensure_ascii=False, indent=2)
        summary["outputs"]["grid_results_json"] = grid_path

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics_raw.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(best_metrics), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    repo_meta = {
        "n_brands": int(len(repo.brand_names)),
        "brand_names": list(repo.brand_names),
        "brand_domains": dict(repo.brand_domains),
    }
    with open(os.path.join(out_dir, "brand_repository_meta.json"), "w", encoding="utf-8") as f:
        json.dump(repo_meta, f, ensure_ascii=False, indent=2)

    print(f"[inference] done: out_dir={out_dir}")
    print(f"[inference] predictions={pred_csv_path}")
    print(f"[inference] brand_level_diagnosis_csv={brand_level_csv_path}")
    print(f"[inference] brand_level_diagnosis_json={brand_level_json_path}")
    print(f"[inference] summary={summary_path}")
    print(
        "[inference] best: "
        f"fusion_method={best_fusion_method} "
        f"alpha_mean={best_alpha:.4f} "
        f"gate_k={best_gate_k:.4f} gate_b={best_gate_b:.4f} "
        f"threshold={best_threshold:.4f} "
        f"{args.grid_objective}={best_obj:.4f}"
    )
    print(
        "[inference] path_resolution: "
        f"direct={int(pre.resolve_stats.get('direct_image_path_exists', 0))} "
        f"by_sample_key={int(pre.resolve_stats.get('resolved_from_sample_key', 0))}"
    )
    print(
        "[inference] metrics: "
        f"acc={metrics_dict['accuracy']:.4f} "
        f"p={metrics_dict['precision']:.4f} "
        f"r={metrics_dict['recall']:.4f} "
        f"f1={metrics_dict['f1']:.4f} "
        f"brand_acc_on_phish={metrics_dict['brand_acc_on_phish']:.4f}"
    )


if __name__ == "__main__":
    main()
