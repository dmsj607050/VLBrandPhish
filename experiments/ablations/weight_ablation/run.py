"""Rescore the detector's rank-1 crop without loading an image encoder.

Inputs must be trusted local caches from the same frozen inference run. A
multi-crop query cache is supported, but only slot zero is used; an invalid
first crop never falls back to another slot. Pre-aggregated Top-3 similarities
cannot be used because their per-crop evidence has already been lost.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.components.brand.naming import brand_converter, normalize_brand_key


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def path_key(value):
    return str(value).strip().replace("\\", "/").casefold()


def truth(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


class BenchmarkBrands:
    """Apply the benchmark's CSV aliases and grouped Microsoft identity."""

    MICROSOFT = {"microsoft", "microsoft corporation", "microsoft onedrive",
                 "office365", "outlook", "onedrive"}

    @classmethod
    def base(cls, value):
        key = normalize_brand_key(brand_converter(str(value or "")))
        return "microsoft" if key in cls.MICROSOFT else key

    def __init__(self, path):
        self.aliases = {}
        self.domains = defaultdict(list)
        for row in read_csv(path):
            logo, phish = row.get("logo_brand", ""), row.get("phish_brand", "")
            canonical = self.base(logo)
            if not canonical:
                continue
            for name in (logo, phish, brand_converter(logo), brand_converter(phish)):
                key = normalize_brand_key(name)
                if key:
                    self.aliases[key] = canonical
            for domain in str(row.get("url", "")).replace(",", ";").split(";"):
                domain = domain.strip()
                if domain and domain not in self.domains[canonical]:
                    self.domains[canonical].append(domain)

    def canonical(self, value):
        key = normalize_brand_key(str(value or ""))
        if key in self.MICROSOFT:
            return "microsoft"
        return self.aliases.get(key, self.base(value))

    def domains_for(self, value):
        return self.domains.get(self.canonical(value), [])


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(y, scores, has_logo, gate, correct_brand, threshold):
    import numpy as np
    pred = has_logo & gate & (scores >= np.float32(threshold))
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    tn, fn = int(np.sum(~pred & ~y)), int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "match_acc": int(np.sum(pred & y & correct_brand)) / int(y.sum())
            if y.any() else 0.0}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--repository-cache", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--normal-urls-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-predictions-csv", type=Path)
    parser.add_argument("--domain-rules", type=Path,
                        help="Optional frozen domain.py from the reference experiment.")
    parser.add_argument("--image-weights", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--fixed-threshold", type=float, default=0.4)
    parser.add_argument("--url-policy-mode", default="phishintention",
                        choices=["phishintention", "official_plus_overrides_no_whitelist"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from inference.decision.domain import decide_url_policy
    if args.domain_rules:
        spec = importlib.util.spec_from_file_location("weight_sweep_domain_rules", args.domain_rules)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        decide_url_policy = module.decide_url_policy

    weights = [float(value) for value in args.image_weights.split(",")]
    if not weights or any(not 0 <= value <= 1 for value in weights):
        parser.error("Image weights must be in [0, 1].")
    if args.batch_size < 1 or not 0 <= args.fixed_threshold <= 1:
        parser.error("Invalid batch size or fixed threshold.")
    if args.reference_predictions_csv and 0.4 not in weights:
        parser.error("Reference validation requires image weight 0.4.")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly.")

    query = np.load(args.query_cache, allow_pickle=True)
    features, masks = query["query_feat"], query["query_mask"]
    if features.ndim != 3 or masks.shape != features.shape[:2] or features.shape[1] < 1:
        raise ValueError("Expected an ordered [sample, crop-slot, feature] query cache.")
    # Deliberately retain slot zero even when lower-ranked slots are valid.
    valid = masks[:, 0].astype(bool)
    y = query["y_true"].astype(bool)
    rows = read_csv(args.metadata_csv)
    if len(rows) != len(y):
        raise ValueError("Query cache and metadata have different row counts.")
    cached_paths = [path_key(value) for value in query["resolved_image_paths"].tolist()]
    metadata = {path_key(row["image_path"]): row for row in rows}
    if len(metadata) != len(rows) or len(set(cached_paths)) != len(cached_paths) or \
            set(cached_paths) != set(metadata):
        raise ValueError("Query cache and metadata must contain the same unique image paths.")
    rows = [metadata[key] for key in cached_paths]
    input_paths = cached_paths
    labels = np.array([truth(row.get("gt_is_phishing", row.get("gt_is_phish", "")))
                       for row in rows], dtype=bool)
    if not np.array_equal(y, labels):
        raise ValueError("Query cache labels disagree with metadata.")

    payload = torch.load(args.repository_cache, map_location="cpu", weights_only=False)
    names = list(payload["brand_names"])
    logo_emb = payload["logo_emb"].float().to(device)
    logo_mask = payload["logo_mask"].bool().to(device)
    text_proto = payload["text_proto"].float().to(device)
    n, brands = len(y), len(names)
    sim_img = np.zeros((n, brands), dtype=np.float32)
    sim_txt = np.zeros_like(sim_img)
    indices = np.flatnonzero(valid)
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start:start + args.batch_size]
            q = torch.as_tensor(features[selected, 0, :], device=device)
            image_scores = torch.einsum("qd,bkd->qbk", q, logo_emb)
            image_scores.masked_fill_(~logo_mask.unsqueeze(0), -1e9)
            sim_img[selected] = image_scores.max(-1).values.cpu().numpy()
            sim_txt[selected] = torch.einsum("qd,bd->qb", q, text_proto).cpu().numpy()
    np.savez_compressed(args.out_dir / "top1_similarities.npz", sim_img=sim_img,
                        sim_txt=sim_txt, has_logo=valid, y_true=y,
                        brand_names=np.asarray(names), image_paths=np.asarray(input_paths))
    print(f"Scored {len(indices)}/{n} rank-1 crops against {brands} brands.", flush=True)

    benchmark = BenchmarkBrands(args.normal_urls_csv)
    canonical_gt = [benchmark.canonical(row.get("gt_brand", "")) for row in rows]
    urls = [str(row.get("url", "")) for row in rows]
    gate_cache = {}
    grid, summary, reference_check = [], [], None
    has_reference = logo_mask.any(-1).cpu().numpy()
    for alpha in weights:
        fused = np.float32(alpha) * sim_img + np.float32(1 - alpha) * sim_txt
        fused[:, ~has_reference] = sim_txt[:, ~has_reference]
        best_idx = fused.argmax(1)
        best_score = fused[np.arange(n), best_idx]
        predicted = [names[int(index)] if valid[i] else "" for i, index in enumerate(best_idx)]
        correct = np.array([bool(benchmark.canonical(brand)) and
                            benchmark.canonical(brand) == canonical_gt[i]
                            for i, brand in enumerate(predicted)], dtype=bool)
        gate = np.zeros(n, dtype=bool)
        for i in indices:
            key = (urls[i], predicted[i])
            if key not in gate_cache:
                gate_cache[key] = bool(decide_url_policy(
                    urls[i], benchmark.domains_for(predicted[i]),
                    url_policy_mode=args.url_policy_mode).accept)
            gate[i] = gate_cache[key]
        points = [metric_row(y, best_score, valid, gate, correct, threshold / 100)
                  for threshold in range(101)]
        best = max(points, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
        fixed = metric_row(y, best_score, valid, gate, correct, args.fixed_threshold)
        record = {"image_weight": alpha, "text_weight": round(1 - alpha, 10),
                  "best_threshold": best["threshold"], "max_f1": best["f1"],
                  "precision": best["precision"], "recall": best["recall"],
                  "match_acc_at_best_f1": best["match_acc"],
                  "fixed_threshold": args.fixed_threshold, "f1_at_fixed_threshold": fixed["f1"],
                  "match_acc_at_fixed_threshold": fixed["match_acc"],
                  "n": n, "phish": int(y.sum()), "benign": int((~y).sum()), "topk": 1}
        summary.append(record)
        grid.extend({"image_weight": alpha, **point} for point in points)
        print(json.dumps(record), flush=True)
        if alpha == 0.4:
            write_csv(args.out_dir / "default_predictions.csv", [
                {"image_path": row["image_path"], "gt_is_phish": int(y[i]),
                 "selected_brand": predicted[i], "brand_score": float(best_score[i]),
                 "has_candidate": int(valid[i]), "url_would_be_phish": int(gate[i]),
                 "brand_correct": int(correct[i])} for i, row in enumerate(rows)])
            if args.reference_predictions_csv:
                reference = {path_key(row["image_path"]): row
                             for row in read_csv(args.reference_predictions_csv)}
                if len(reference) != n or set(reference) != set(input_paths):
                    raise ValueError("Reference and query manifests differ.")
                bad_predictions, bad_brands, max_delta = 0, 0, 0.0
                for i, key in enumerate(input_paths):
                    saved = reference[key]
                    saved_candidate = int(saved["selected_candidate_index"]) >= 0
                    saved_pred = saved_candidate and truth(saved["url_would_be_phish"]) and \
                        float(saved["brand_score"]) >= args.fixed_threshold
                    current_pred = bool(valid[i] and gate[i] and
                                        best_score[i] >= args.fixed_threshold)
                    bad_predictions += saved_pred != current_pred
                    bad_brands += (benchmark.canonical(saved["selected_brand"]) !=
                                   benchmark.canonical(predicted[i]))
                    max_delta = max(max_delta, abs(float(saved["brand_score"]) -
                                                   float(best_score[i])) if valid[i] else 0)
                reference_check = {"prediction_mismatches": int(bad_predictions),
                                   "brand_mismatches": int(bad_brands),
                                   "max_score_delta": max_delta, "fixed_metrics": fixed}
                if bad_predictions or bad_brands or max_delta > 5e-5:
                    raise RuntimeError(f"Default operating point did not reproduce reference: {reference_check}")

    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "threshold_grid.csv", grid)
    inputs = {name: {"path": str(getattr(args, name)), "sha256": file_digest(getattr(args, name))}
              for name in ["query_cache", "repository_cache", "metadata_csv", "normal_urls_csv"]}
    if args.reference_predictions_csv:
        inputs["reference_predictions_csv"] = {"path": str(args.reference_predictions_csv),
                                                "sha256": file_digest(args.reference_predictions_csv)}
    if args.domain_rules:
        inputs["domain_rules"] = {"path": str(args.domain_rules), "sha256": file_digest(args.domain_rules)}
    provenance = {"inputs": inputs, "script_sha256": file_digest(__file__), "topk": 1,
                  "source_query_slots": int(features.shape[1]), "n": n, "phish": int(y.sum()),
                  "benign": int((~y).sum()), "brands": brands, "device": str(device),
                  "url_policy_mode": args.url_policy_mode, "reference_check": reference_check,
                  "threshold_grid": [0, 1, 0.01],
                  "threshold_selection": "max F1 on the evaluation set; ties prefer precision then higher threshold",
                  "elapsed_seconds": time.monotonic() - started}
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Completed in {provenance['elapsed_seconds']:.1f}s: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
