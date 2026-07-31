"""Evaluate inference evidence and render sample- and brand-level reports."""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from ..components.brand.naming import normalize_brand_key
from .data_contracts import DetectionSample
from .types import AlphaOutput, Metrics, PrecomputedData

def _parse_grid_values(raw: str, fallback: List[float]) -> List[float]:
    txt = str(raw or "").strip()
    if not txt:
        return [float(x) for x in fallback]

    vals: List[float] = []
    if txt.count(":") == 2 and ("," not in txt and ";" not in txt):
        s, e, step = txt.split(":")
        start = float(s.strip())
        end = float(e.strip())
        inc = float(step.strip())
        if inc <= 0:
            raise ValueError(f"Grid step must be > 0, got: {txt}")
        v = start
        while v <= end + 1e-12:
            vals.append(float(v))
            v += inc
    else:
        for token in txt.replace(";", ",").split(","):
            s = token.strip()
            if s:
                vals.append(float(s))

    out: List[float] = []
    seen = set()
    for v in vals:
        k = f"{float(v):.10f}"
        if k in seen:
            continue
        seen.add(k)
        out.append(float(v))
    return out

def _objective_value(metrics: Metrics, objective: str) -> float:
    m = metrics.to_dict()
    key = str(objective or "f1").strip().lower()
    if key == "accuracy":
        return float(m["accuracy"])
    if key == "precision":
        return float(m["precision"])
    if key == "recall":
        return float(m["recall"])
    if key == "brand_acc_on_phish":
        return float(m["brand_acc_on_phish"])
    return float(m["f1"])

def _evaluate_threshold(
    *,
    pre: PrecomputedData,
    alpha_out: AlphaOutput,
    threshold: float,
) -> Tuple[Metrics, np.ndarray]:
    has_logo = pre.has_logo.astype(bool)
    pred = has_logo & (alpha_out.best_score >= float(threshold)) & (alpha_out.url_would_be_phish.astype(bool))
    y = pre.y_true.astype(bool)

    tp = int(np.sum(pred & y))
    tn = int(np.sum((~pred) & (~y)))
    fp = int(np.sum(pred & (~y)))
    fn = int(np.sum((~pred) & y))

    brand_total = int(np.sum(y & np.array([bool(x) for x in pre.gt_brand_keys], dtype=bool)))
    brand_correct = int(np.sum(y & pred & (alpha_out.brand_match.astype(bool))))

    m = Metrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        brand_total_phish=brand_total,
        brand_correct_phish=brand_correct,
    )
    return m, pred.astype(np.uint8)

def _build_prediction_rows(
    *,
    detections: List[DetectionSample],
    pre: PrecomputedData,
    alpha_out: AlphaOutput,
    threshold: float,
    pred_vec: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    n = len(detections)
    for i in range(n):
        det = detections[i]
        has_logo = int(pre.has_logo[i]) == 1
        pred_brand = str(alpha_out.pred_brand_names[i] if has_logo else "")
        score = float(alpha_out.best_score[i] if has_logo else 0.0)
        score_img = float(alpha_out.best_score_img[i] if has_logo else 0.0)
        score_txt = float(alpha_out.best_score_txt[i] if has_logo else 0.0)
        pred_is_phishing = int(pred_vec[i])
        url_would_be_phish = int(alpha_out.url_would_be_phish[i] if has_logo else 0)
        url_match = int(alpha_out.url_match[i] if has_logo else 0)
        screenshot_domain = str(pre.screenshot_domains[i] if has_logo else "")

        if not has_logo:
            decision_reason = "no_logo_box"
        elif score < float(threshold):
            decision_reason = "score_below_threshold"
        else:
            decision_reason = str(alpha_out.url_reason_above_thr[i] or "")

        gt_brand = str(det.gt_brand or "")
        brand_correct = 0
        if bool(det.gt_is_phishing) and gt_brand and pred_brand:
            brand_correct = int(normalize_brand_key(gt_brand) == normalize_brand_key(pred_brand))

        row = {
            "image_path": det.image_path,
            "resolved_image_path": pre.resolved_image_paths[i],
            "url": det.url,
            "sample_key": det.sample_key,
            "gt_is_phishing": int(bool(det.gt_is_phishing)),
            "gt_brand": gt_brand,
            "has_logo": int(has_logo),
            "pred_is_phishing": int(pred_is_phishing),
            "pred_brand": pred_brand,
            "pred_brand_idx": int(alpha_out.pred_brand_idx[i] if has_logo else -1),
            "score": float(score),
            "score_img": float(score_img),
            "score_txt": float(score_txt),
            "alpha_used": float(alpha_out.alpha_vec[i] if has_logo and i < len(alpha_out.alpha_vec) else 0.0),
            "fusion_method": str(alpha_out.fusion_method),
            "gate_k": float(alpha_out.gate_k),
            "gate_b": float(alpha_out.gate_b),
            "url_would_be_phish": int(url_would_be_phish),
            "url_match": int(url_match),
            "screenshot_domain": screenshot_domain,
            "decision_reason": decision_reason,
            "brand_correct_phish_only": int(brand_correct),
        }
        rows.append(row)
    return rows

def _safe_div(numer: int, denom: int) -> float:
    if int(denom) <= 0:
        return 0.0
    return float(numer) / float(denom)

def _build_brand_level_reports(
    *,
    rows: List[Dict[str, object]],
    threshold: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    wrong_pred_counters: Dict[str, Counter[str]] = {}
    overall_reason_counter: Counter[str] = Counter()

    for row in rows:
        if int(row.get("gt_is_phishing", 0) or 0) != 1:
            continue

        gt_brand_raw = str(row.get("gt_brand", "") or "").strip()
        gt_brand_key = normalize_brand_key(gt_brand_raw)
        if not gt_brand_key:
            continue

        rec = grouped.setdefault(
            gt_brand_key,
            {
                "brand_key": gt_brand_key,
                "brand_name": gt_brand_raw,
                "n_total_phish": 0,
                "n_has_logo": 0,
                "n_pred_phish": 0,
                "n_top1_brand_correct": 0,
                "n_final_brand_correct": 0,
                "reason_no_logo_box": 0,
                "reason_brand_wrong": 0,
                "reason_brand_correct_score_below_threshold": 0,
                "reason_brand_correct_url_match": 0,
                "reason_brand_correct_url_mismatch": 0,
            },
        )

        rec["n_total_phish"] = int(rec["n_total_phish"]) + 1

        has_logo = int(row.get("has_logo", 0) or 0) == 1
        pred_is_phishing = int(row.get("pred_is_phishing", 0) or 0) == 1
        brand_correct = int(row.get("brand_correct_phish_only", 0) or 0) == 1
        score = float(row.get("score", 0.0) or 0.0)
        url_match = int(row.get("url_match", 0) or 0) == 1

        if has_logo:
            rec["n_has_logo"] = int(rec["n_has_logo"]) + 1
        if pred_is_phishing:
            rec["n_pred_phish"] = int(rec["n_pred_phish"]) + 1
        if brand_correct:
            rec["n_top1_brand_correct"] = int(rec["n_top1_brand_correct"]) + 1
        if brand_correct and pred_is_phishing:
            rec["n_final_brand_correct"] = int(rec["n_final_brand_correct"]) + 1

        if not has_logo:
            reason = "no_logo_box"
        elif not brand_correct:
            reason = "brand_wrong"
            pred_brand_raw = str(row.get("pred_brand", "") or "").strip()
            if pred_brand_raw:
                wrong_pred_counters.setdefault(gt_brand_key, Counter())
                wrong_pred_counters[gt_brand_key][pred_brand_raw] += 1
        elif score < float(threshold):
            reason = "brand_correct_score_below_threshold"
        else:
            if pred_is_phishing:
                reason = "brand_correct_url_mismatch"
            else:
                reason = "brand_correct_url_match"

        rec[f"reason_{reason}"] = int(rec[f"reason_{reason}"]) + 1
        overall_reason_counter[reason] += 1

    brand_rows: List[Dict[str, object]] = []
    total_phish = 0
    total_final_correct = 0
    total_top1_correct = 0

    for brand_key in sorted(grouped.keys()):
        rec = dict(grouped[brand_key])
        n_total = int(rec["n_total_phish"])
        n_has_logo = int(rec["n_has_logo"])
        n_top1_correct = int(rec["n_top1_brand_correct"])
        n_final_correct = int(rec["n_final_brand_correct"])

        total_phish += n_total
        total_top1_correct += n_top1_correct
        total_final_correct += n_final_correct

        top_confusions = wrong_pred_counters.get(brand_key, Counter()).most_common(5)
        top_confusions_json = json.dumps(
            [{"pred_brand": str(name), "count": int(cnt)} for name, cnt in top_confusions],
            ensure_ascii=False,
        )
        top_confusion_brand = str(top_confusions[0][0]) if top_confusions else ""
        top_confusion_count = int(top_confusions[0][1]) if top_confusions else 0

        rec["top1_brand_acc_on_all_phish"] = float(_safe_div(n_top1_correct, n_total))
        rec["top1_brand_acc_on_has_logo"] = float(_safe_div(n_top1_correct, n_has_logo))
        rec["final_brand_acc_on_all_phish"] = float(_safe_div(n_final_correct, n_total))
        rec["final_brand_acc_on_has_logo"] = float(_safe_div(n_final_correct, n_has_logo))
        rec["pred_phish_rate"] = float(_safe_div(int(rec["n_pred_phish"]), n_total))
        rec["reason_no_logo_box_rate"] = float(_safe_div(int(rec["reason_no_logo_box"]), n_total))
        rec["reason_brand_wrong_rate"] = float(_safe_div(int(rec["reason_brand_wrong"]), n_total))
        rec["reason_brand_correct_score_below_threshold_rate"] = float(
            _safe_div(int(rec["reason_brand_correct_score_below_threshold"]), n_total)
        )
        rec["reason_brand_correct_url_match_rate"] = float(
            _safe_div(int(rec["reason_brand_correct_url_match"]), n_total)
        )
        rec["reason_brand_correct_url_mismatch_rate"] = float(
            _safe_div(int(rec["reason_brand_correct_url_mismatch"]), n_total)
        )
        rec["top_wrong_pred_brand"] = top_confusion_brand
        rec["top_wrong_pred_brand_count"] = top_confusion_count
        rec["top_wrong_pred_brands_json"] = top_confusions_json
        brand_rows.append(rec)

    overview = {
        "n_brands_with_phish": int(len(brand_rows)),
        "n_total_phish": int(total_phish),
        "overall_top1_brand_acc_on_all_phish": float(_safe_div(total_top1_correct, total_phish)),
        "overall_final_brand_acc_on_all_phish": float(_safe_div(total_final_correct, total_phish)),
        "overall_reason_counts": {str(k): int(v) for k, v in overall_reason_counter.items()},
        "threshold": float(threshold),
    }
    return brand_rows, overview
