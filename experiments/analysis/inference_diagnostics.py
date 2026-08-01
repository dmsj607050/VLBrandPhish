from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from inference.vlbrandphish.runtime.paths import default_run_dir, resolve_default_paths
from inference.vlbrandphish.components.brand.naming import normalize_brand_key, parse_semicolon_list
from inference.vlbrandphish.decision.domain import registered_domain
from inference.vlbrandphish.recognition.features import FIXED_DECISION_THRESHOLD


def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(float(str(v)))
    except Exception:
        return int(default)


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(str(v))
    except Exception:
        return float(default)


def _host_from_url_raw(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        host = str(urlparse(raw).hostname or "").strip().lower()
    except Exception:
        return ""
    return host


def _safe_pct(numer: int, denom: int) -> float:
    if int(denom) <= 0:
        return 0.0
    return float(numer) / float(denom)


def _compute_cls_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if int(yt) == 1 and int(yp) == 1:
            tp += 1
        elif int(yt) == 0 and int(yp) == 0:
            tn += 1
        elif int(yt) == 0 and int(yp) == 1:
            fp += 1
        elif int(yt) == 1 and int(yp) == 0:
            fn += 1
    acc = _safe_pct(tp + tn, tp + tn + fp + fn)
    prec = _safe_pct(tp, tp + fp)
    rec = _safe_pct(tp, tp + fn)
    f1 = 0.0 if (prec + rec) <= 0 else (2.0 * prec * rec) / (prec + rec)
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k): str(v) for k, v in row.items()} for row in reader]


def _load_brand_split_map(brands_overlap_csv: str) -> Dict[str, str]:
    rows = _read_csv_rows(brands_overlap_csv)
    out: Dict[str, str] = {}
    for row in rows:
        brand = normalize_brand_key(row.get("brand_dir_7299", ""))
        if not brand:
            continue
        in_277 = _to_int(row.get("in_277", 0), default=0)
        out[brand] = "old48" if in_277 == 1 else "new104"
    return out


def _load_brand_url_maps(
    normal_urls_csv: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    raw_hosts_map: Dict[str, List[str]] = defaultdict(list)
    reg_domains_map: Dict[str, List[str]] = defaultdict(list)

    with open(normal_urls_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = normalize_brand_key(row.get("logo_brand", "") or row.get("phish_brand", ""))
            if not brand:
                continue
            raw_urls = str(row.get("url", "") or row.get("domain", "") or "")
            for it in parse_semicolon_list(raw_urls):
                host = _host_from_url_raw(it)
                reg = registered_domain(it)
                if host and host not in raw_hosts_map[brand]:
                    raw_hosts_map[brand].append(host)
                if reg and reg not in reg_domains_map[brand]:
                    reg_domains_map[brand].append(reg)
    return dict(raw_hosts_map), dict(reg_domains_map)


def _discover_latest_stage2_output(base_dir: str) -> Tuple[str, str]:
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Output base dir not found: {base_dir}")

    candidates: List[Tuple[float, str, str]] = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        pred = os.path.join(path, "predictions.csv")
        summary = os.path.join(path, "summary.json")
        if os.path.isfile(pred):
            try:
                mtime = os.path.getmtime(pred)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, pred, summary))
    if not candidates:
        raise FileNotFoundError(f"No stage2 outputs with predictions.csv under: {base_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, pred_path, summary_path = candidates[0]
    return pred_path, summary_path


def _fixed_threshold() -> float:
    """Return the published fixed decision threshold for diagnostic breakdowns."""
    return float(FIXED_DECISION_THRESHOLD)

def _fmt_pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def _calc_brand_top1(
    rows: Iterable[Dict[str, str]],
    split_map: Dict[str, str],
    split_name: str,
) -> Dict[str, object]:
    total = 0
    total_with_logo = 0
    correct = 0
    correct_with_logo = 0

    for r in rows:
        if _to_int(r.get("gt_is_phishing", 0)) != 1:
            continue
        gt_key = normalize_brand_key(r.get("gt_brand", ""))
        if split_name != "all" and split_map.get(gt_key, "new104") != split_name:
            continue
        total += 1

        has_logo = _to_int(r.get("has_logo", 0)) == 1
        if has_logo:
            total_with_logo += 1

        pred_key = normalize_brand_key(r.get("pred_brand", ""))
        is_correct = has_logo and pred_key and pred_key == gt_key
        if is_correct:
            correct += 1
            correct_with_logo += 1

    return {
        "split": split_name,
        "total_phish": int(total),
        "total_with_logo": int(total_with_logo),
        "correct": int(correct),
        "acc_on_all_phish": float(_safe_pct(correct, total)),
        "acc_on_has_logo_only": float(_safe_pct(correct_with_logo, total_with_logo)),
    }


def _calc_phish_chain_breakdown(
    rows: Iterable[Dict[str, str]],
    split_map: Dict[str, str],
    split_name: str,
    threshold: float,
) -> Dict[str, object]:
    counter = Counter()
    total = 0

    for r in rows:
        if _to_int(r.get("gt_is_phishing", 0)) != 1:
            continue
        gt_key = normalize_brand_key(r.get("gt_brand", ""))
        if split_name != "all" and split_map.get(gt_key, "new104") != split_name:
            continue
        total += 1

        has_logo = _to_int(r.get("has_logo", 0)) == 1
        pred_key = normalize_brand_key(r.get("pred_brand", ""))
        score = _to_float(r.get("score", 0.0), 0.0)
        url_match = _to_int(r.get("url_match", 0)) == 1

        if not has_logo:
            counter["no_logo"] += 1
            continue

        brand_correct = bool(pred_key) and pred_key == gt_key
        if not brand_correct:
            counter["brand_wrong"] += 1
            continue

        if score < float(threshold):
            counter["brand_correct_score_below_threshold"] += 1
            continue

        if url_match:
            counter["brand_correct_url_match"] += 1
        else:
            counter["brand_correct_url_mismatch"] += 1

    out = {"split": split_name, "total": int(total), "counts": dict(counter), "rates": {}}
    rates: Dict[str, float] = {}
    for k in [
        "no_logo",
        "brand_wrong",
        "brand_correct_score_below_threshold",
        "brand_correct_url_match",
        "brand_correct_url_mismatch",
    ]:
        rates[k] = _safe_pct(int(counter.get(k, 0)), total)
    out["rates"] = rates
    return out


def _calc_fn_decomposition(rows: Iterable[Dict[str, str]], threshold: float) -> Dict[str, int]:
    c = Counter()
    for r in rows:
        yt = _to_int(r.get("gt_is_phishing", 0))
        yp = _to_int(r.get("pred_is_phishing", 0))
        if not (yt == 1 and yp == 0):
            continue
        has_logo = _to_int(r.get("has_logo", 0)) == 1
        score = _to_float(r.get("score", 0.0), 0.0)
        url_match = _to_int(r.get("url_match", 0)) == 1
        if not has_logo:
            c["fn_no_logo"] += 1
        elif score < float(threshold):
            c["fn_score_below_threshold"] += 1
        elif url_match:
            c["fn_url_match"] += 1
        else:
            c["fn_other"] += 1
    return {str(k): int(v) for k, v in dict(c).items()}


def _ab_url_metrics(
    rows: Iterable[Dict[str, str]],
    raw_hosts_map: Dict[str, List[str]],
    reg_domains_map: Dict[str, List[str]],
    threshold: float,
) -> Dict[str, object]:
    y_true: List[int] = []
    pred_strict: List[int] = []
    pred_norm: List[int] = []
    diff_counter = Counter()
    missing_brand_in_url_map = 0

    for r in rows:
        yt = _to_int(r.get("gt_is_phishing", 0))
        y_true.append(yt)

        has_logo = _to_int(r.get("has_logo", 0)) == 1
        score = _to_float(r.get("score", 0.0), 0.0)
        if (not has_logo) or (score < float(threshold)):
            pred_strict.append(0)
            pred_norm.append(0)
            continue

        pred_brand_key = normalize_brand_key(r.get("pred_brand", ""))
        if pred_brand_key not in raw_hosts_map and pred_brand_key not in reg_domains_map:
            missing_brand_in_url_map += 1

        official_raw_hosts = set(raw_hosts_map.get(pred_brand_key, []))
        official_reg_domains = set(reg_domains_map.get(pred_brand_key, []))

        shot_host = _host_from_url_raw(r.get("url", ""))
        shot_reg = registered_domain(r.get("url", ""))

        strict_match = bool(shot_host) and shot_host in official_raw_hosts
        norm_match = bool(shot_reg) and shot_reg in official_reg_domains

        ps = 0 if strict_match else 1
        pn = 0 if norm_match else 1
        pred_strict.append(ps)
        pred_norm.append(pn)

        if ps != pn:
            if ps == 1 and pn == 0:
                diff_counter["strict_more_phish"] += 1
            elif ps == 0 and pn == 1:
                diff_counter["norm_more_phish"] += 1

    m_strict = _compute_cls_metrics(y_true, pred_strict)
    m_norm = _compute_cls_metrics(y_true, pred_norm)
    return {
        "strict_host": m_strict,
        "registered_domain": m_norm,
        "decision_diff_counts": dict(diff_counter),
        "missing_brand_in_url_map": int(missing_brand_in_url_map),
    }


def _write_rows(path: str, rows: List[Dict[str, str]], top_k: int) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    out_rows = rows[: max(1, int(top_k))]
    fieldnames = list(out_rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


def main() -> None:
    defaults = resolve_default_paths(__file__)
    output_base = os.path.join(defaults.phishkiller_root, "outputs")

    parser = argparse.ArgumentParser(description="Unified stage2 diagnosis (old/new split + URL AB + error breakdown).")
    parser.add_argument("--predictions-csv", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--normal-urls-csv", type=str, default=defaults.normal_urls_csv)
    parser.add_argument("--brands-overlap-csv", type=str, default=os.path.join(defaults.testb_root, "brands_overlap.csv"))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--out-base-dir", type=str, default=output_base)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    pred_csv = str(args.predictions_csv).strip()
    summary_json = str(args.summary_json).strip()
    if not pred_csv:
        pred_csv, auto_summary = _discover_latest_stage2_output(str(args.out_base_dir))
        if not summary_json:
            summary_json = auto_summary
    pred_csv = os.path.abspath(pred_csv)
    summary_json = os.path.abspath(summary_json) if summary_json else ""
    if not os.path.isfile(pred_csv):
        raise FileNotFoundError(f"predictions.csv not found: {pred_csv}")

    threshold = _fixed_threshold()

    out_dir = str(args.out_dir).strip()
    if not out_dir:
        out_dir = default_run_dir(str(args.out_base_dir), "diagnose")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = _read_csv_rows(pred_csv)
    split_map = _load_brand_split_map(os.path.abspath(args.brands_overlap_csv))
    raw_hosts_map, reg_domains_map = _load_brand_url_maps(os.path.abspath(args.normal_urls_csv))

    print("[diag] ===== Unified Diagnosis =====")
    print(f"[diag] predictions_csv={pred_csv}")
    print(f"[diag] summary_json={summary_json if summary_json else '(none)'}")
    print(f"[diag] rows={len(rows)} threshold={threshold:.6f} top_k={int(args.top_k)}")
    print(f"[diag] split_map_brands={len(split_map)} url_map_brands={len(reg_domains_map)}")

    brand_metrics_all = _calc_brand_top1(rows, split_map, "all")
    brand_metrics_old = _calc_brand_top1(rows, split_map, "old48")
    brand_metrics_new = _calc_brand_top1(rows, split_map, "new104")

    print(
        "[diag] brand_top1(all): "
        f"correct={brand_metrics_all['correct']}/{brand_metrics_all['total_phish']} "
        f"acc={_fmt_pct(brand_metrics_all['acc_on_all_phish'])} "
        f"acc_has_logo={_fmt_pct(brand_metrics_all['acc_on_has_logo_only'])}"
    )
    print(
        "[diag] brand_top1(old48): "
        f"correct={brand_metrics_old['correct']}/{brand_metrics_old['total_phish']} "
        f"acc={_fmt_pct(brand_metrics_old['acc_on_all_phish'])} "
        f"acc_has_logo={_fmt_pct(brand_metrics_old['acc_on_has_logo_only'])}"
    )
    print(
        "[diag] brand_top1(new104): "
        f"correct={brand_metrics_new['correct']}/{brand_metrics_new['total_phish']} "
        f"acc={_fmt_pct(brand_metrics_new['acc_on_all_phish'])} "
        f"acc_has_logo={_fmt_pct(brand_metrics_new['acc_on_has_logo_only'])}"
    )

    chain_all = _calc_phish_chain_breakdown(rows, split_map, "all", threshold=threshold)
    chain_old = _calc_phish_chain_breakdown(rows, split_map, "old48", threshold=threshold)
    chain_new = _calc_phish_chain_breakdown(rows, split_map, "new104", threshold=threshold)

    def _print_chain(tag: str, payload: Dict[str, object]) -> None:
        counts = payload.get("counts", {})
        rates = payload.get("rates", {})
        print(
            f"[diag] chain({tag}): "
            f"no_logo={int(counts.get('no_logo', 0))}({_fmt_pct(rates.get('no_logo', 0.0))}) "
            f"brand_wrong={int(counts.get('brand_wrong', 0))}({_fmt_pct(rates.get('brand_wrong', 0.0))}) "
            f"brand_correct_score_below={int(counts.get('brand_correct_score_below_threshold', 0))}"
            f"({_fmt_pct(rates.get('brand_correct_score_below_threshold', 0.0))}) "
            f"brand_correct_url_match={int(counts.get('brand_correct_url_match', 0))}"
            f"({_fmt_pct(rates.get('brand_correct_url_match', 0.0))}) "
            f"brand_correct_url_mismatch={int(counts.get('brand_correct_url_mismatch', 0))}"
            f"({_fmt_pct(rates.get('brand_correct_url_mismatch', 0.0))})"
        )

    _print_chain("all", chain_all)
    _print_chain("old48", chain_old)
    _print_chain("new104", chain_new)

    y_true = [_to_int(r.get("gt_is_phishing", 0)) for r in rows]
    y_pred = [_to_int(r.get("pred_is_phishing", 0)) for r in rows]
    final_metrics = _compute_cls_metrics(y_true, y_pred)
    fn_decomp = _calc_fn_decomposition(rows, threshold=threshold)

    print(
        "[diag] final_cls(current): "
        f"acc={_fmt_pct(final_metrics['accuracy'])} "
        f"p={_fmt_pct(final_metrics['precision'])} "
        f"r={_fmt_pct(final_metrics['recall'])} "
        f"f1={_fmt_pct(final_metrics['f1'])} "
        f"tp={final_metrics['tp']} tn={final_metrics['tn']} fp={final_metrics['fp']} fn={final_metrics['fn']}"
    )
    print(
        "[diag] fn_decomposition: "
        f"no_logo={int(fn_decomp.get('fn_no_logo', 0))} "
        f"score_below={int(fn_decomp.get('fn_score_below_threshold', 0))} "
        f"url_match={int(fn_decomp.get('fn_url_match', 0))} "
        f"other={int(fn_decomp.get('fn_other', 0))}"
    )

    url_ab = _ab_url_metrics(
        rows,
        raw_hosts_map=raw_hosts_map,
        reg_domains_map=reg_domains_map,
        threshold=threshold,
    )
    strict_m = url_ab["strict_host"]
    norm_m = url_ab["registered_domain"]
    print(
        "[diag] url_ab(strict_host): "
        f"acc={_fmt_pct(strict_m['accuracy'])} p={_fmt_pct(strict_m['precision'])} "
        f"r={_fmt_pct(strict_m['recall'])} f1={_fmt_pct(strict_m['f1'])}"
    )
    print(
        "[diag] url_ab(registered_domain): "
        f"acc={_fmt_pct(norm_m['accuracy'])} p={_fmt_pct(norm_m['precision'])} "
        f"r={_fmt_pct(norm_m['recall'])} f1={_fmt_pct(norm_m['f1'])}"
    )
    print(
        "[diag] url_ab_diff: "
        f"strict_more_phish={int(url_ab['decision_diff_counts'].get('strict_more_phish', 0))} "
        f"norm_more_phish={int(url_ab['decision_diff_counts'].get('norm_more_phish', 0))} "
        f"missing_brand_in_url_map={int(url_ab.get('missing_brand_in_url_map', 0))}"
    )

    # Hard sample exports
    phish_rows = [r for r in rows if _to_int(r.get("gt_is_phishing", 0)) == 1]
    no_logo_rows = [r for r in phish_rows if _to_int(r.get("has_logo", 0)) == 0]
    brand_wrong_rows = [
        r
        for r in phish_rows
        if _to_int(r.get("has_logo", 0)) == 1
        and normalize_brand_key(r.get("pred_brand", "")) != normalize_brand_key(r.get("gt_brand", ""))
    ]
    score_below_rows = [
        r
        for r in phish_rows
        if _to_int(r.get("has_logo", 0)) == 1
        and normalize_brand_key(r.get("pred_brand", "")) == normalize_brand_key(r.get("gt_brand", ""))
        and _to_float(r.get("score", 0.0), 0.0) < float(threshold)
    ]
    url_match_rows = [
        r
        for r in phish_rows
        if _to_int(r.get("has_logo", 0)) == 1
        and normalize_brand_key(r.get("pred_brand", "")) == normalize_brand_key(r.get("gt_brand", ""))
        and _to_float(r.get("score", 0.0), 0.0) >= float(threshold)
        and _to_int(r.get("url_match", 0)) == 1
    ]
    benign_fp_rows = [
        r
        for r in rows
        if _to_int(r.get("gt_is_phishing", 0)) == 0 and _to_int(r.get("pred_is_phishing", 0)) == 1
    ]

    _write_rows(os.path.join(out_dir, "hard_phish_no_logo.csv"), no_logo_rows, int(args.top_k))
    _write_rows(os.path.join(out_dir, "hard_phish_brand_wrong.csv"), brand_wrong_rows, int(args.top_k))
    _write_rows(os.path.join(out_dir, "hard_phish_brand_correct_score_below.csv"), score_below_rows, int(args.top_k))
    _write_rows(os.path.join(out_dir, "hard_phish_brand_correct_url_match.csv"), url_match_rows, int(args.top_k))
    _write_rows(os.path.join(out_dir, "hard_benign_fp.csv"), benign_fp_rows, int(args.top_k))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "predictions_csv": pred_csv,
            "summary_json": summary_json,
            "brands_overlap_csv": os.path.abspath(args.brands_overlap_csv),
            "normal_urls_csv": os.path.abspath(args.normal_urls_csv),
            "threshold": float(threshold),
        },
        "brand_top1": {
            "all": brand_metrics_all,
            "old48": brand_metrics_old,
            "new104": brand_metrics_new,
        },
        "chain_breakdown": {
            "all": chain_all,
            "old48": chain_old,
            "new104": chain_new,
        },
        "final_cls_current": final_metrics,
        "fn_decomposition": fn_decomp,
        "url_ab": url_ab,
        "exports": {
            "hard_phish_no_logo_csv": os.path.join(out_dir, "hard_phish_no_logo.csv"),
            "hard_phish_brand_wrong_csv": os.path.join(out_dir, "hard_phish_brand_wrong.csv"),
            "hard_phish_brand_correct_score_below_csv": os.path.join(out_dir, "hard_phish_brand_correct_score_below.csv"),
            "hard_phish_brand_correct_url_match_csv": os.path.join(out_dir, "hard_phish_brand_correct_url_match.csv"),
            "hard_benign_fp_csv": os.path.join(out_dir, "hard_benign_fp.csv"),
        },
    }

    report_path = os.path.join(out_dir, "diagnosis_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[diag] report_json={report_path}")
    print(f"[diag] out_dir={out_dir}")
    print("[diag] ===========================")


if __name__ == "__main__":
    main()
