from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ...config.paths import resolve_default_paths
from ...core.brands import normalize_brand_key


def _discover_latest_stage2_summary(outputs_dir: str) -> str:
    outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")

    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(outputs_dir):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir) or not name.startswith(("inference_", "stage2_")):
            continue
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            mtime = os.path.getmtime(summary_path)
        except OSError:
            mtime = 0.0
        candidates.append((mtime, summary_path))

    if not candidates:
        raise FileNotFoundError(f"no stage2 summary found under: {outputs_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_grid_values(raw: str, fallback: Sequence[float]) -> List[float]:
    text = str(raw or "").strip()
    if not text:
        return [float(x) for x in fallback]

    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"invalid range grid: {text}")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
        if step <= 0:
            raise ValueError(f"grid step must be positive: {text}")

        out: List[float] = []
        value = start
        while value <= stop + 1e-12:
            out.append(round(float(value), 10))
            value += step
        return out

    vals = [p.strip() for p in text.split(",") if p.strip()]
    if not vals:
        raise ValueError(f"empty grid values: {text}")
    return [float(v) for v in vals]


def _load_summary(summary_json: str) -> Dict[str, object]:
    summary_json = os.path.abspath(summary_json)
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"summary.json not found: {summary_json}")
    with open(summary_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_repo_brand_keys(repo_meta_json: str) -> List[str]:
    repo_meta_json = os.path.abspath(repo_meta_json)
    if not os.path.isfile(repo_meta_json):
        raise FileNotFoundError(f"brand_repository_meta.json not found: {repo_meta_json}")
    with open(repo_meta_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    brand_names = [str(x) for x in list(payload.get("brand_names", []))]
    if not brand_names:
        raise ValueError(f"empty brand_names in {repo_meta_json}")
    return [normalize_brand_key(x) for x in brand_names]


def _load_split_map(brands_overlap_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(os.path.abspath(brands_overlap_csv), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand_key = normalize_brand_key(row.get("brand_dir_7299", ""))
            if not brand_key:
                continue
            in_277 = str(row.get("in_277", "0")).strip()
            out[brand_key] = "Overlap" if in_277 == "1" else "Novel"
    if not out:
        raise ValueError(f"empty split map from: {brands_overlap_csv}")
    return out


def _load_feature_cache(feature_cache_path: str) -> Dict[str, object]:
    feature_cache_path = os.path.abspath(feature_cache_path)
    if not os.path.isfile(feature_cache_path):
        raise FileNotFoundError(f"feature cache not found: {feature_cache_path}")

    z = np.load(feature_cache_path, allow_pickle=True)
    return {
        "sim_img": np.asarray(z["sim_img"], dtype=np.float32),
        "sim_txt": np.asarray(z["sim_txt"], dtype=np.float32),
        "has_logo": np.asarray(z["has_logo"], dtype=np.uint8).reshape(-1),
        "y_true": np.asarray(z["y_true"], dtype=np.uint8).reshape(-1),
        "gt_brand_keys": [str(x) for x in np.asarray(z["gt_brand_keys"], dtype=object).tolist()],
    }


def _safe_div(numer: int, denom: int) -> float:
    if int(denom) <= 0:
        return 0.0
    return float(numer) / float(denom)


def _predict_subset(
    *,
    sim_img: np.ndarray,
    sim_txt: np.ndarray,
    alpha: float,
    subset_idx: np.ndarray,
    has_logo: np.ndarray,
) -> np.ndarray:
    fused = float(alpha) * sim_img + (1.0 - float(alpha)) * sim_txt
    subset_scores = fused[:, subset_idx]
    best_local_idx = np.asarray(np.argmax(subset_scores, axis=1), dtype=np.int64)
    pred_global_idx = np.asarray(subset_idx[best_local_idx], dtype=np.int64)
    pred_global_idx[has_logo.astype(bool) == 0] = -1
    return pred_global_idx


def _calc_brand_metrics(
    *,
    gt_brand_keys: Sequence[str],
    has_logo: np.ndarray,
    y_true: np.ndarray,
    pred_idx_global: np.ndarray,
    repo_brand_keys: Sequence[str],
    eval_gt_key_set: set[str],
) -> Dict[str, object]:
    total = 0
    total_has_logo = 0
    correct = 0
    correct_has_logo = 0

    for i in range(len(gt_brand_keys)):
        if int(y_true[i]) != 1:
            continue

        gt_key = str(gt_brand_keys[i] or "")
        if not gt_key or gt_key not in eval_gt_key_set:
            continue

        total += 1
        has_logo_i = int(has_logo[i]) == 1
        if has_logo_i:
            total_has_logo += 1

        pred_idx = int(pred_idx_global[i])
        if pred_idx < 0 or pred_idx >= len(repo_brand_keys):
            continue

        pred_key = str(repo_brand_keys[pred_idx] or "")
        if has_logo_i and pred_key and pred_key == gt_key:
            correct += 1
            correct_has_logo += 1

    return {
        "total_phish": int(total),
        "total_phish_has_logo": int(total_has_logo),
        "correct_top1": int(correct),
        "acc_top1_on_all_phish": float(_safe_div(correct, total)),
        "acc_top1_on_has_logo_only": float(_safe_div(correct_has_logo, total_has_logo)),
    }


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    defaults = resolve_default_paths(__file__)
    default_outputs_dir = os.path.join(defaults.phishkiller_root, "outputs")
    default_overlap_csv = os.path.join(defaults.testb_root, "brands_overlap.csv")

    parser = argparse.ArgumentParser(
        description=(
            "Independent brand classification evaluation from cached stage2 similarities. "
            "Scans best alpha only on full protocol, then reports alpha=0, alpha=1, and best-alpha "
            "under Overlap/Novel/full protocols."
        )
    )
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--outputs-dir", type=str, default=default_outputs_dir)
    parser.add_argument("--feature-cache-path", type=str, default="")
    parser.add_argument("--repo-meta-json", type=str, default="")
    parser.add_argument("--brands-overlap-csv", type=str, default=default_overlap_csv)
    parser.add_argument("--alpha-grid", type=str, default="0:1:0.1")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    summary_json = str(args.summary_json or "").strip()
    if not summary_json:
        summary_json = _discover_latest_stage2_summary(str(args.outputs_dir))
    summary_json = os.path.abspath(summary_json)
    summary = _load_summary(summary_json)
    run_dir = os.path.dirname(summary_json)

    feature_cache_path = str(args.feature_cache_path or "").strip()
    if not feature_cache_path:
        feature_cache_path = str(summary.get("cache", {}).get("feature_cache_path", "") or "").strip()
    if not feature_cache_path:
        raise ValueError("feature_cache_path is missing; pass --feature-cache-path or provide a summary.json with cache info")

    repo_meta_json = str(args.repo_meta_json or "").strip()
    if not repo_meta_json:
        repo_meta_json = os.path.join(run_dir, "brand_repository_meta.json")

    cache = _load_feature_cache(feature_cache_path)
    sim_img = np.asarray(cache["sim_img"], dtype=np.float32)
    sim_txt = np.asarray(cache["sim_txt"], dtype=np.float32)
    has_logo = np.asarray(cache["has_logo"], dtype=np.uint8).reshape(-1)
    y_true = np.asarray(cache["y_true"], dtype=np.uint8).reshape(-1)
    gt_brand_keys = [str(x) for x in cache["gt_brand_keys"]]
    repo_brand_keys = _load_repo_brand_keys(repo_meta_json)

    if sim_img.ndim != 2 or sim_txt.ndim != 2:
        raise RuntimeError(f"invalid similarity shapes: sim_img={tuple(sim_img.shape)} sim_txt={tuple(sim_txt.shape)}")
    if sim_img.shape != sim_txt.shape:
        raise RuntimeError(f"sim_img/sim_txt shape mismatch: {tuple(sim_img.shape)} vs {tuple(sim_txt.shape)}")
    if sim_img.shape[1] != len(repo_brand_keys):
        raise RuntimeError(
            f"brand dim mismatch: feature B={sim_img.shape[1]} but repo_brand_keys={len(repo_brand_keys)}"
        )

    split_map = _load_split_map(str(args.brands_overlap_csv))
    overlap_key_set = {k for k, v in split_map.items() if v == "Overlap"}
    novel_key_set = {k for k, v in split_map.items() if v == "Novel"}
    full_key_set = set(split_map.keys())

    protocol_key_sets = {
        "Overlap": overlap_key_set,
        "Novel": novel_key_set,
        "full": full_key_set,
    }
    protocol_subset_idx: Dict[str, np.ndarray] = {}
    for protocol_name, key_set in protocol_key_sets.items():
        subset_idx = np.asarray([i for i, k in enumerate(repo_brand_keys) if k in key_set], dtype=np.int64)
        if subset_idx.size <= 0:
            raise RuntimeError(f"empty candidate library for protocol={protocol_name}")
        protocol_subset_idx[protocol_name] = subset_idx

    alpha_grid = _parse_grid_values(str(args.alpha_grid), fallback=[round(0.1 * i, 2) for i in range(11)])

    full_scan_rows: List[Dict[str, object]] = []
    best_alpha = float(alpha_grid[0])
    best_correct = -1
    best_acc = -1.0
    for alpha in alpha_grid:
        pred_idx_full = _predict_subset(
            sim_img=sim_img,
            sim_txt=sim_txt,
            alpha=float(alpha),
            subset_idx=protocol_subset_idx["full"],
            has_logo=has_logo,
        )
        metrics_full = _calc_brand_metrics(
            gt_brand_keys=gt_brand_keys,
            has_logo=has_logo,
            y_true=y_true,
            pred_idx_global=pred_idx_full,
            repo_brand_keys=repo_brand_keys,
            eval_gt_key_set=full_key_set,
        )
        scan_row = {
            "alpha": float(alpha),
            **metrics_full,
        }
        full_scan_rows.append(scan_row)

        correct = int(metrics_full["correct_top1"])
        acc = float(metrics_full["acc_top1_on_all_phish"])
        if correct > best_correct or (correct == best_correct and acc > best_acc):
            best_alpha = float(alpha)
            best_correct = int(correct)
            best_acc = float(acc)

    selected_alpha_rows = [
        ("alpha_0_image_text_only", 0.0),
        ("alpha_1_image_image_only", 1.0),
        ("best_alpha_full_scan", float(best_alpha)),
    ]

    result_rows: List[Dict[str, object]] = []
    for setting_name, alpha in selected_alpha_rows:
        for protocol_name in ["Overlap", "Novel", "full"]:
            pred_idx = _predict_subset(
                sim_img=sim_img,
                sim_txt=sim_txt,
                alpha=float(alpha),
                subset_idx=protocol_subset_idx[protocol_name],
                has_logo=has_logo,
            )
            metrics = _calc_brand_metrics(
                gt_brand_keys=gt_brand_keys,
                has_logo=has_logo,
                y_true=y_true,
                pred_idx_global=pred_idx,
                repo_brand_keys=repo_brand_keys,
                eval_gt_key_set=protocol_key_sets[protocol_name],
            )
            result_rows.append(
                {
                    "setting": setting_name,
                    "alpha": float(alpha),
                    "protocol": protocol_name,
                    "n_candidate_brands": int(protocol_subset_idx[protocol_name].size),
                    "n_eval_brands": int(len(protocol_key_sets[protocol_name])),
                    **metrics,
                }
            )

    print("[brand_protocol_cache] ===== Brand Classification From Cache =====")
    print(f"[brand_protocol_cache] summary_json={summary_json}")
    print(f"[brand_protocol_cache] feature_cache={os.path.abspath(feature_cache_path)}")
    print(f"[brand_protocol_cache] repo_meta_json={os.path.abspath(repo_meta_json)}")
    print(
        "[brand_protocol_cache] protocol_brand_counts: "
        f"Overlap={len(overlap_key_set)} Novel={len(novel_key_set)} full={len(full_key_set)}"
    )
    print(
        "[brand_protocol_cache] best_alpha_full_scan: "
        f"alpha={best_alpha:.4f} correct={best_correct}/{len([k for k in gt_brand_keys if k in full_key_set])} "
        f"acc={_fmt_pct(best_acc)}"
    )
    for row in result_rows:
        print(
            "[brand_protocol_cache] "
            f"{row['setting']} | protocol={row['protocol']} | brands={row['n_candidate_brands']} "
            f"| correct={row['correct_top1']}/{row['total_phish']} ({_fmt_pct(float(row['acc_top1_on_all_phish']))}) "
            f"| has_logo_acc={_fmt_pct(float(row['acc_top1_on_has_logo_only']))}"
        )
    print("[brand_protocol_cache] ===========================================")

    out_payload = {
        "summary_json": summary_json,
        "feature_cache_path": os.path.abspath(feature_cache_path),
        "repo_meta_json": os.path.abspath(repo_meta_json),
        "alpha_grid": [float(x) for x in alpha_grid],
        "best_alpha_full_scan": float(best_alpha),
        "protocol_brand_counts": {
            "Overlap": int(len(overlap_key_set)),
            "Novel": int(len(novel_key_set)),
            "full": int(len(full_key_set)),
        },
        "full_alpha_scan": full_scan_rows,
        "results": result_rows,
    }

    out_json = str(args.out_json or "").strip()
    if not out_json:
        out_json = os.path.join(run_dir, "brand_protocol_eval_from_cache.json")
    out_json = os.path.abspath(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    out_csv = str(args.out_csv or "").strip()
    if not out_csv:
        out_csv = os.path.join(run_dir, "brand_protocol_eval_from_cache.csv")
    out_csv = os.path.abspath(out_csv)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(result_rows[0].keys()) if result_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    print(f"[brand_protocol_cache] saved_json={out_json}")
    print(f"[brand_protocol_cache] saved_csv={out_csv}")


if __name__ == "__main__":
    main()
