from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from common.paths import resolve_default_paths
from inference.vlbrandphish.components.brand.naming import normalize_brand_key


def _discover_latest_stage2_summary(outputs_dir: str) -> str:
    outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")
    cands: List[Tuple[float, str]] = []
    for name in os.listdir(outputs_dir):
        d = os.path.join(outputs_dir, name)
        if not os.path.isdir(d):
            continue
        if not name.startswith(("inference_", "stage2_")):
            continue
        p = os.path.join(d, "summary.json")
        if not os.path.isfile(p):
            continue
        try:
            ts = os.path.getmtime(p)
        except OSError:
            ts = 0.0
        cands.append((ts, p))
    if not cands:
        raise FileNotFoundError(f"no stage2 summary found under: {outputs_dir}")
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _load_split_map(brands_overlap_csv: str) -> Dict[str, str]:
    import csv

    out: Dict[str, str] = {}
    with open(os.path.abspath(brands_overlap_csv), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = normalize_brand_key(row.get("brand_dir_7299", ""))
            if not b:
                continue
            in_277 = str(row.get("in_277", "0")).strip()
            out[b] = "old48" if in_277 == "1" else "new104"
    if not out:
        raise ValueError(f"empty split map from: {brands_overlap_csv}")
    return out


def _calc_mode_metrics(
    *,
    mode_name: str,
    gt_brand_keys: List[str],
    has_logo: np.ndarray,
    y_true: np.ndarray,
    pred_idx_global: np.ndarray,
    repo_brand_keys: List[str],
    eval_gt_key_set: set,
) -> Dict[str, object]:
    total = 0
    total_has_logo = 0
    correct = 0
    correct_has_logo = 0

    for i in range(len(gt_brand_keys)):
        if int(y_true[i]) != 1:
            continue
        gt = str(gt_brand_keys[i] or "")
        if not gt or gt not in eval_gt_key_set:
            continue
        total += 1

        if int(has_logo[i]) == 1:
            total_has_logo += 1

        pidx = int(pred_idx_global[i])
        if pidx < 0 or pidx >= len(repo_brand_keys):
            continue
        pred_key = str(repo_brand_keys[pidx] or "")
        ok = (int(has_logo[i]) == 1) and bool(pred_key) and (pred_key == gt)
        if ok:
            correct += 1
            correct_has_logo += 1

    acc_all = float(correct / total) if total > 0 else 0.0
    acc_has_logo = float(correct_has_logo / total_has_logo) if total_has_logo > 0 else 0.0
    return {
        "mode": mode_name,
        "total_phish": int(total),
        "total_phish_has_logo": int(total_has_logo),
        "correct": int(correct),
        "acc_on_all_phish": float(acc_all),
        "acc_on_has_logo_only": float(acc_has_logo),
    }


def main() -> None:
    defaults = resolve_default_paths(__file__)
    default_outputs_dir = os.path.join(defaults.phishkiller_root, "outputs")
    default_overlap_csv = os.path.join(defaults.testb_root, "brands_overlap.csv")

    parser = argparse.ArgumentParser(
        description="Independent brand-mode accuracy: old48/new104/all152 with separate reference candidate sets."
    )
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--outputs-dir", type=str, default=default_outputs_dir)
    parser.add_argument("--brands-overlap-csv", type=str, default=default_overlap_csv)
    parser.add_argument("--alpha", type=float, default=-1.0, help="If <0, use best_alpha from summary.")
    args = parser.parse_args()

    summary_json = str(args.summary_json or "").strip()
    if not summary_json:
        summary_json = _discover_latest_stage2_summary(str(args.outputs_dir))
    summary_json = os.path.abspath(summary_json)
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"summary.json not found: {summary_json}")

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    feature_cache_path = str(summary.get("cache", {}).get("feature_cache_path", "") or "").strip()
    if not feature_cache_path:
        raise ValueError("feature_cache_path missing in summary.json")
    feature_cache_path = os.path.abspath(feature_cache_path)
    if not os.path.isfile(feature_cache_path):
        raise FileNotFoundError(f"feature cache not found: {feature_cache_path}")

    run_dir = os.path.dirname(summary_json)
    repo_meta_path = os.path.join(run_dir, "brand_repository_meta.json")
    if not os.path.isfile(repo_meta_path):
        raise FileNotFoundError(f"brand_repository_meta.json not found in run dir: {repo_meta_path}")

    with open(repo_meta_path, "r", encoding="utf-8") as f:
        repo_meta = json.load(f)
    brand_names = [str(x) for x in list(repo_meta.get("brand_names", []))]
    if not brand_names:
        raise ValueError("empty brand_names in brand_repository_meta.json")
    repo_brand_keys = [normalize_brand_key(x) for x in brand_names]

    z = np.load(feature_cache_path, allow_pickle=True)
    sim_img = np.asarray(z["sim_img"], dtype=np.float32)
    sim_txt = np.asarray(z["sim_txt"], dtype=np.float32)
    has_logo = np.asarray(z["has_logo"], dtype=np.uint8).reshape(-1)
    y_true = np.asarray(z["y_true"], dtype=np.uint8).reshape(-1)
    gt_brand_keys = [str(x) for x in np.asarray(z["gt_brand_keys"], dtype=object).tolist()]

    n = int(sim_img.shape[0])
    b = int(sim_img.shape[1]) if sim_img.ndim == 2 else 0
    if n <= 0 or b <= 0:
        raise RuntimeError(f"invalid feature cache shape: sim_img={tuple(sim_img.shape)}")
    if b != len(repo_brand_keys):
        raise RuntimeError(
            f"brand dim mismatch: feature B={b} but repo_brand_keys={len(repo_brand_keys)} (run dir: {run_dir})"
        )

    alpha = float(args.alpha)
    if alpha < 0:
        alpha = float(summary.get("selection", {}).get("best_alpha", 0.5))

    fused = float(alpha) * sim_img + (1.0 - float(alpha)) * sim_txt

    split_map = _load_split_map(os.path.abspath(args.brands_overlap_csv))
    old_key_set = {k for k, v in split_map.items() if str(v) == "old48"}
    new_key_set = {k for k, v in split_map.items() if str(v) == "new104"}
    all_key_set = set(split_map.keys())

    old_idx = np.asarray([i for i, k in enumerate(repo_brand_keys) if k in old_key_set], dtype=np.int64)
    new_idx = np.asarray([i for i, k in enumerate(repo_brand_keys) if k in new_key_set], dtype=np.int64)
    all_idx = np.asarray([i for i, _ in enumerate(repo_brand_keys)], dtype=np.int64)

    if old_idx.size <= 0 or new_idx.size <= 0:
        raise RuntimeError(
            f"candidate index empty: old_idx={int(old_idx.size)} new_idx={int(new_idx.size)}; check brand overlap map"
        )

    def _pred_idx_for_subset(sub_idx: np.ndarray) -> np.ndarray:
        sub_scores = fused[:, sub_idx]
        local = np.asarray(np.argmax(sub_scores, axis=1), dtype=np.int64)
        return np.asarray(sub_idx[local], dtype=np.int64)

    pred_old = _pred_idx_for_subset(old_idx)
    pred_new = _pred_idx_for_subset(new_idx)
    pred_all = _pred_idx_for_subset(all_idx)

    # No-logo samples should be treated as non-predictable/wrong in brand accuracy.
    pred_old[has_logo.astype(bool) == 0] = -1
    pred_new[has_logo.astype(bool) == 0] = -1
    pred_all[has_logo.astype(bool) == 0] = -1

    m_old = _calc_mode_metrics(
        mode_name="old48_ref_old48_eval",
        gt_brand_keys=gt_brand_keys,
        has_logo=has_logo,
        y_true=y_true,
        pred_idx_global=pred_old,
        repo_brand_keys=repo_brand_keys,
        eval_gt_key_set=old_key_set,
    )
    m_new = _calc_mode_metrics(
        mode_name="new104_ref_new104_eval",
        gt_brand_keys=gt_brand_keys,
        has_logo=has_logo,
        y_true=y_true,
        pred_idx_global=pred_new,
        repo_brand_keys=repo_brand_keys,
        eval_gt_key_set=new_key_set,
    )
    m_all = _calc_mode_metrics(
        mode_name="all152_ref_all152_eval",
        gt_brand_keys=gt_brand_keys,
        has_logo=has_logo,
        y_true=y_true,
        pred_idx_global=pred_all,
        repo_brand_keys=repo_brand_keys,
        eval_gt_key_set=all_key_set,
    )

    print("[brand_mode] ===== Independent Mode Evaluation =====")
    print(f"[brand_mode] summary_json={summary_json}")
    print(f"[brand_mode] feature_cache={feature_cache_path}")
    print(f"[brand_mode] alpha={alpha:.4f}")
    print(
        "[brand_mode] old48: "
        f"correct={m_old['correct']}/{m_old['total_phish']} "
        f"acc={100.0 * float(m_old['acc_on_all_phish']):.2f}% "
        f"acc_has_logo={100.0 * float(m_old['acc_on_has_logo_only']):.2f}%"
    )
    print(
        "[brand_mode] new104: "
        f"correct={m_new['correct']}/{m_new['total_phish']} "
        f"acc={100.0 * float(m_new['acc_on_all_phish']):.2f}% "
        f"acc_has_logo={100.0 * float(m_new['acc_on_has_logo_only']):.2f}%"
    )
    print(
        "[brand_mode] all152: "
        f"correct={m_all['correct']}/{m_all['total_phish']} "
        f"acc={100.0 * float(m_all['acc_on_all_phish']):.2f}% "
        f"acc_has_logo={100.0 * float(m_all['acc_on_has_logo_only']):.2f}%"
    )
    print("[brand_mode] =======================================")

    out_payload = {
        "summary_json": summary_json,
        "feature_cache_path": feature_cache_path,
        "alpha": float(alpha),
        "results": {"old48": m_old, "new104": m_new, "all152": m_all},
    }
    out_path = os.path.join(os.path.dirname(summary_json), "brand_mode_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"[brand_mode] saved_json={out_path}")


if __name__ == "__main__":
    main()
