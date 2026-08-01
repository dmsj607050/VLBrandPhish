from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from inference.vlbrandphish.runtime.paths import resolve_default_paths
from inference.vlbrandphish.components.brand.naming import normalize_brand_key


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


def _load_split_map(brands_overlap_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(os.path.abspath(brands_overlap_csv), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand_key = normalize_brand_key(row.get("brand_dir_7299", ""))
            if not brand_key:
                continue
            in_277 = str(row.get("in_277", "0")).strip()
            out[brand_key] = "overlap" if in_277 == "1" else "novel"
    if not out:
        raise ValueError(f"empty split map from: {brands_overlap_csv}")
    return out


def _load_summary(summary_json: str) -> Dict[str, object]:
    summary_json = os.path.abspath(summary_json)
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"summary.json not found: {summary_json}")
    with open(summary_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_repo_brand_keys(run_dir: str) -> List[str]:
    repo_meta_path = os.path.join(run_dir, "brand_repository_meta.json")
    if not os.path.isfile(repo_meta_path):
        raise FileNotFoundError(f"brand_repository_meta.json not found: {repo_meta_path}")
    with open(repo_meta_path, "r", encoding="utf-8") as f:
        repo_meta = json.load(f)
    brand_names = [str(x) for x in list(repo_meta.get("brand_names", []))]
    if not brand_names:
        raise ValueError(f"empty brand_names in: {repo_meta_path}")
    return [normalize_brand_key(x) for x in brand_names]


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


def _top2_margin(sim: np.ndarray) -> np.ndarray:
    if sim.ndim != 2:
        return np.zeros((int(sim.shape[0]) if sim.ndim > 0 else 0,), dtype=np.float32)
    n = int(sim.shape[0])
    b = int(sim.shape[1]) if sim.ndim == 2 else 0
    if n <= 0 or b <= 0:
        return np.zeros((max(n, 0),), dtype=np.float32)
    if b == 1:
        return np.asarray(sim[:, 0], dtype=np.float32)
    top2 = np.partition(sim, kth=b - 2, axis=1)[:, -2:]
    top1 = np.max(top2, axis=1)
    top2nd = np.min(top2, axis=1)
    return np.asarray(top1 - top2nd, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=np.float32), -30.0, 30.0)
    return np.asarray(1.0 / (1.0 + np.exp(-x_clip)), dtype=np.float32)


def _fuse_scores(
    *,
    sim_img: np.ndarray,
    sim_txt: np.ndarray,
    fusion_method: str,
    alpha: float,
    gate_k: float,
    gate_b: float,
) -> np.ndarray:
    method = str(fusion_method or "").strip().lower()
    if method == "dynamic_margin_gate":
        delta_margin = np.asarray(_top2_margin(sim_img) - _top2_margin(sim_txt), dtype=np.float32)
        alpha_vec = _sigmoid(float(gate_k) * delta_margin + float(gate_b))
    else:
        alpha_vec = np.full((sim_img.shape[0],), float(alpha), dtype=np.float32)
    alpha_col = alpha_vec.reshape(-1, 1)
    return np.asarray(alpha_col * sim_img + (1.0 - alpha_col) * sim_txt, dtype=np.float32)


def _predict_for_subset(
    *,
    sim_img: np.ndarray,
    sim_txt: np.ndarray,
    fusion_method: str,
    alpha: float,
    gate_k: float,
    gate_b: float,
    sub_idx: np.ndarray,
    has_logo: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    fused = _fuse_scores(
        sim_img=sim_img,
        sim_txt=sim_txt,
        fusion_method=str(fusion_method),
        alpha=float(alpha),
        gate_k=float(gate_k),
        gate_b=float(gate_b),
    )
    sub_scores = fused[:, sub_idx]
    local_idx = np.asarray(np.argmax(sub_scores, axis=1), dtype=np.int64)
    pred_idx_global = np.asarray(sub_idx[local_idx], dtype=np.int64)
    pred_score = np.asarray(sub_scores[np.arange(sub_scores.shape[0]), local_idx], dtype=np.float32)

    no_logo = has_logo.astype(bool) == 0
    pred_idx_global[no_logo] = -1
    pred_score[no_logo] = 0.0
    return pred_idx_global, pred_score


def _calc_protocol_metrics(
    *,
    gt_brand_keys: Sequence[str],
    has_logo: np.ndarray,
    y_true: np.ndarray,
    pred_idx_global: np.ndarray,
    pred_score: np.ndarray,
    repo_brand_keys: Sequence[str],
    eval_gt_key_set: set[str],
    threshold: float,
) -> Dict[str, object]:
    total = 0
    total_has_logo = 0
    correct_top1 = 0
    correct_top1_has_logo = 0
    correct_top1_above_thr = 0
    correct_top1_above_thr_has_logo = 0

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
        pred_key = ""
        if 0 <= pred_idx < len(repo_brand_keys):
            pred_key = str(repo_brand_keys[pred_idx] or "")
        correct = has_logo_i and bool(pred_key) and pred_key == gt_key
        if correct:
            correct_top1 += 1
            correct_top1_has_logo += 1

        if correct and float(pred_score[i]) >= float(threshold):
            correct_top1_above_thr += 1
            correct_top1_above_thr_has_logo += 1

    return {
        "total_phish": int(total),
        "total_phish_has_logo": int(total_has_logo),
        "correct_top1": int(correct_top1),
        "acc_top1_on_all_phish": float(_safe_div(correct_top1, total)),
        "acc_top1_on_has_logo_only": float(_safe_div(correct_top1_has_logo, total_has_logo)),
        "correct_top1_above_best_threshold": int(correct_top1_above_thr),
        "acc_top1_above_best_threshold_on_all_phish": float(_safe_div(correct_top1_above_thr, total)),
        "acc_top1_above_best_threshold_on_has_logo_only": float(
            _safe_div(correct_top1_above_thr_has_logo, total_has_logo)
        ),
    }


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    defaults = resolve_default_paths(__file__)
    default_outputs_dir = os.path.join(defaults.phishkiller_root, "outputs")
    default_overlap_csv = os.path.join(defaults.testb_root, "brands_overlap.csv")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate brand classification ability under Overlap/Novel/full protocols "
            "for alpha=0, alpha=1, and best-fusion mode using cached stage2 similarities."
        )
    )
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--outputs-dir", type=str, default=default_outputs_dir)
    parser.add_argument("--brands-overlap-csv", type=str, default=default_overlap_csv)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    summary_json = str(args.summary_json or "").strip()
    if not summary_json:
        summary_json = _discover_latest_stage2_summary(str(args.outputs_dir))
    summary_json = os.path.abspath(summary_json)

    summary = _load_summary(summary_json)
    run_dir = os.path.dirname(summary_json)
    feature_cache_path = str(summary.get("cache", {}).get("feature_cache_path", "") or "").strip()
    if not feature_cache_path:
        raise ValueError("feature_cache_path missing in summary.json")

    cache = _load_feature_cache(feature_cache_path)
    sim_img = np.asarray(cache["sim_img"], dtype=np.float32)
    sim_txt = np.asarray(cache["sim_txt"], dtype=np.float32)
    has_logo = np.asarray(cache["has_logo"], dtype=np.uint8).reshape(-1)
    y_true = np.asarray(cache["y_true"], dtype=np.uint8).reshape(-1)
    gt_brand_keys = [str(x) for x in cache["gt_brand_keys"]]

    repo_brand_keys = _load_repo_brand_keys(run_dir)
    if sim_img.ndim != 2 or sim_txt.ndim != 2:
        raise RuntimeError(f"invalid similarity shapes: sim_img={sim_img.shape}, sim_txt={sim_txt.shape}")
    if sim_img.shape != sim_txt.shape:
        raise RuntimeError(f"sim_img/sim_txt shape mismatch: {sim_img.shape} vs {sim_txt.shape}")
    if sim_img.shape[1] != len(repo_brand_keys):
        raise RuntimeError(
            f"brand dim mismatch: feature B={sim_img.shape[1]} but repo_brand_keys={len(repo_brand_keys)}"
        )

    split_map = _load_split_map(str(args.brands_overlap_csv))
    overlap_key_set = {k for k, v in split_map.items() if v == "overlap"}
    novel_key_set = {k for k, v in split_map.items() if v == "novel"}
    full_key_set = set(split_map.keys())

    protocol_sets = {
        "Overlap": overlap_key_set,
        "Novel": novel_key_set,
        "full": full_key_set,
    }
    protocol_indices: Dict[str, np.ndarray] = {}
    for protocol_name, key_set in protocol_sets.items():
        idx = np.asarray([i for i, k in enumerate(repo_brand_keys) if k in key_set], dtype=np.int64)
        if idx.size <= 0:
            raise RuntimeError(f"candidate index empty for protocol={protocol_name}")
        protocol_indices[protocol_name] = idx

    selection = summary.get("selection", {})
    best_fusion_method = str(selection.get("fusion_method", "linear_alpha") or "linear_alpha").strip().lower()
    if best_fusion_method not in {"linear_alpha", "dynamic_margin_gate"}:
        best_fusion_method = "linear_alpha"
    best_alpha = float(selection.get("best_alpha", 0.5))
    best_gate_k = float(selection.get("best_gate_k", 0.0))
    best_gate_b = float(selection.get("best_gate_b", 0.0))
    best_threshold = float(selection.get("best_threshold", 0.0))
    fusion_specs = [
        ("alpha_0_image_text_only", "linear_alpha", 0.0, 0.0, 0.0),
        ("alpha_1_image_image_only", "linear_alpha", 1.0, 0.0, 0.0),
    ]
    if best_fusion_method == "dynamic_margin_gate":
        fusion_specs.append(("best_fusion_dynamic_margin_gate", "dynamic_margin_gate", best_alpha, best_gate_k, best_gate_b))
    else:
        fusion_specs.append(("best_fusion", "linear_alpha", best_alpha, best_gate_k, best_gate_b))

    result_rows: List[Dict[str, object]] = []
    for score_mode, fusion_method, alpha, gate_k, gate_b in fusion_specs:
        for protocol_name in ["Overlap", "Novel", "full"]:
            key_set = protocol_sets[protocol_name]
            sub_idx = protocol_indices[protocol_name]
            pred_idx_global, pred_score = _predict_for_subset(
                sim_img=sim_img,
                sim_txt=sim_txt,
                fusion_method=str(fusion_method),
                alpha=float(alpha),
                gate_k=float(gate_k),
                gate_b=float(gate_b),
                sub_idx=sub_idx,
                has_logo=has_logo,
            )
            metrics = _calc_protocol_metrics(
                gt_brand_keys=gt_brand_keys,
                has_logo=has_logo,
                y_true=y_true,
                pred_idx_global=pred_idx_global,
                pred_score=pred_score,
                repo_brand_keys=repo_brand_keys,
                eval_gt_key_set=key_set,
                threshold=best_threshold,
            )
            result_rows.append(
                {
                    "score_mode": score_mode,
                    "fusion_method": str(fusion_method),
                    "alpha": float(alpha),
                    "gate_k": float(gate_k),
                    "gate_b": float(gate_b),
                    "protocol": protocol_name,
                    "n_candidate_brands": int(len(sub_idx)),
                    "n_eval_brands": int(len(key_set)),
                    **metrics,
                }
            )

    print("[brand_protocol] ===== Brand Protocol Evaluation =====")
    print(f"[brand_protocol] summary_json={summary_json}")
    print(f"[brand_protocol] feature_cache={os.path.abspath(feature_cache_path)}")
    print(
        "[brand_protocol] best_selection: "
        f"fusion_method={best_fusion_method} alpha={best_alpha:.4f} "
        f"gate_k={best_gate_k:.4f} gate_b={best_gate_b:.4f} threshold={best_threshold:.4f}"
    )
    for row in result_rows:
        print(
            "[brand_protocol] "
            f"{row['score_mode']} | protocol={row['protocol']} "
            f"| fusion={row['fusion_method']} alpha={float(row['alpha']):.4f} "
            f"gate_k={float(row['gate_k']):.2f} gate_b={float(row['gate_b']):.2f} "
            f"| brands={row['n_candidate_brands']} "
            f"| top1={row['correct_top1']}/{row['total_phish']} ({_fmt_pct(float(row['acc_top1_on_all_phish']))}) "
            f"| top1_has_logo={_fmt_pct(float(row['acc_top1_on_has_logo_only']))} "
            f"| top1_thr={row['correct_top1_above_best_threshold']}/{row['total_phish']} "
            f"({_fmt_pct(float(row['acc_top1_above_best_threshold_on_all_phish']))})"
        )
    print("[brand_protocol] =====================================")

    out_payload = {
        "summary_json": summary_json,
        "feature_cache_path": os.path.abspath(feature_cache_path),
        "best_fusion_method": str(best_fusion_method),
        "best_alpha": float(best_alpha),
        "best_gate_k": float(best_gate_k),
        "best_gate_b": float(best_gate_b),
        "best_threshold": float(best_threshold),
        "protocol_brand_counts": {
            "Overlap": int(len(overlap_key_set)),
            "Novel": int(len(novel_key_set)),
            "full": int(len(full_key_set)),
        },
        "results": result_rows,
    }

    out_json = str(args.out_json or "").strip()
    if not out_json:
        out_json = os.path.join(run_dir, "brand_protocol_eval.json")
    out_json = os.path.abspath(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    out_csv = str(args.out_csv or "").strip()
    if not out_csv:
        out_csv = os.path.join(run_dir, "brand_protocol_eval.csv")
    out_csv = os.path.abspath(out_csv)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(result_rows[0].keys()) if result_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    print(f"[brand_protocol] saved_json={out_json}")
    print(f"[brand_protocol] saved_csv={out_csv}")


if __name__ == "__main__":
    main()
