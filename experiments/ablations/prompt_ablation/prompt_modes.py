from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:
    average_precision_score = None
    roc_auc_score = None

from inference.runtime.paths import default_run_dir, resolve_default_paths
from inference.components.brand.naming import normalize_brand_key
from inference.data.detection_csv import DetectionSample, read_detection_top3_csv
from inference.runtime.cache import (
    build_query_cache_key,
    hash_dir_tree_signature,
    hash_file_signature,
    load_brand_repository_cache,
    load_precomputed_cache,
    load_query_cache,
    save_brand_repository_cache,
    save_precomputed_cache,
    save_query_cache,
)
from inference.recognition.features import (
    FIXED_DECISION_THRESHOLD,
    FIXED_IMAGE_WEIGHT,
    FIXED_TEXT_WEIGHT,
    compute_fixed_fusion_output,
    precompute_query_features,
)
from inference.evaluation.metrics import (
    build_brand_level_reports as _build_brand_level_reports,
    build_prediction_rows as _build_prediction_rows,
    evaluate_threshold as _evaluate_threshold,
)
from inference.data.image_paths import inspect_detection_csv_format
from inference.runtime.progress import iter_with_progress
from inference.runtime.diagnostics import print_runtime_format_info
from inference.runtime.types import Metrics, PrecomputedData, QueryFeatureData
from inference.components.brand.encoders import CLIPEncoder
from inference.components.brand.repository import BrandRepository, list_brand_dirs, list_logo_paths, load_brand_domains, read_dbi_texts


SIMPLE13_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "{b}",
    "{b} logo",
    "logo of {b}",
    "official logo of {b}",
    "{b} brand logo",
    "a photo of the {b} logo",
    "a clean logo of {b}",
    "a vector logo of {b}",
    "{b} logomark",
    "{b} wordmark",
    "the {b} logo on a website",
    "the {b} logo in a webpage screenshot",
    "a cropped {b} logo",
)


@dataclass(frozen=True)
class PromptModeSpec:
    name: str
    display_name: str
    text_source: str
    text_score_mode: str


PROMPT_MODE_SPECS: Dict[str, PromptModeSpec] = {
    "dbi5": PromptModeSpec("dbi5", "DBI-5", "dbi5", "proto_mean"),
    "simple13": PromptModeSpec("simple13", "simple13", "simple13", "max_per_line"),
    "simple13_mean": PromptModeSpec("simple13_mean", "simple13_mean", "simple13", "proto_mean"),
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _discover_latest_stage2_summary(outputs_dir: str) -> str:
    """Find the most recently modified stage2 summary.json under *outputs_dir*."""
    outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")

    cands: List[Tuple[float, str]] = []
    for name in os.listdir(outputs_dir):
        d = os.path.join(outputs_dir, name)
        if not os.path.isdir(d):
            continue
        if not str(name).startswith(("inference_", "stage2_")):
            continue
        summary_json = os.path.join(d, "summary.json")
        if not os.path.isfile(summary_json):
            continue
        try:
            ts = os.path.getmtime(summary_json)
        except OSError:
            ts = 0.0
        cands.append((ts, summary_json))

    if not cands:
        raise FileNotFoundError(f"no stage2 summary found under: {outputs_dir}")

    cands.sort(key=lambda x: x[0], reverse=True)
    return os.path.abspath(cands[0][1])


def _build_simple13_prompts(brand_name: str) -> List[str]:
    """Expand 13 brand-agnostic templates with one brand name."""
    b = str(brand_name or "").strip()
    return [tpl.format(b=b) for tpl in SIMPLE13_PROMPT_TEMPLATES]


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if v is None:
        return ""
    return str(v)


def _write_markdown_table(path: str, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---" for _ in columns]) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt_cell(row.get(col, "")) for col in columns) + " |\n")


def _build_prompt_repo_cache_key(
    *,
    brand_ckpt: str,
    logo_library_dir: str,
    normal_urls_csv: str,
    dbi5_dir: str,
    max_logo_per_brand: int,
    mode_spec: PromptModeSpec,
) -> str:
    """Stable repo cache key that captures mode, checkpoint, and all input files."""
    if mode_spec.name == "dbi5":
        h = hashlib.md5()
        h.update(b"repo_cache_v3_strict_host_textmean")
        hash_file_signature(brand_ckpt, h)
        hash_file_signature(normal_urls_csv, h)
        hash_dir_tree_signature(
            logo_library_dir,
            h,
            suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
        )
        hash_dir_tree_signature(dbi5_dir, h, suffixes=(".txt",))
        h.update(f":max_logo_per_brand={int(max_logo_per_brand)}".encode("utf-8"))
        return h.hexdigest()[:16]

    h = hashlib.md5()
    h.update(b"prompt_mode_repo_cache_v1")
    h.update(str(mode_spec.name).encode("utf-8"))
    h.update(str(mode_spec.text_score_mode).encode("utf-8"))
    hash_file_signature(brand_ckpt, h)
    hash_file_signature(normal_urls_csv, h)
    hash_dir_tree_signature(
        logo_library_dir,
        h,
        suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    )
    for tpl in SIMPLE13_PROMPT_TEMPLATES:
        h.update(str(tpl).encode("utf-8"))
    h.update(f":max_logo_per_brand={int(max_logo_per_brand)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _make_mode_feature_cache_path(
    *,
    cache_dir: str,
    det_csv: str,
    benign_root: str,
    phish_root: str,
    max_samples: int,
    query_cache_path: str,
    repo_cache_key: str,
    mode_spec: PromptModeSpec,
) -> str:
    """Deterministic feature-cache path keyed on mode plus repo snapshot."""
    if mode_spec.name == "dbi5":
        h = hashlib.md5()
        h.update(b"stage2_feature_cache_v1")
        hash_file_signature(det_csv, h)
        h.update(str(os.path.abspath(benign_root)).encode("utf-8", errors="ignore"))
        h.update(str(os.path.abspath(phish_root)).encode("utf-8", errors="ignore"))
        h.update(f":max_samples={int(max_samples)}".encode("utf-8"))
        h.update(f":repo={str(repo_cache_key)}".encode("utf-8"))
        feature_key = h.hexdigest()[:16]
        return os.path.join(os.path.abspath(cache_dir), f"stage2_precompute_{feature_key}.npz")

    h = hashlib.md5()
    h.update(b"prompt_mode_feature_cache_v1")
    hash_file_signature(query_cache_path, h)
    h.update(f":repo={str(repo_cache_key)}".encode("utf-8"))
    h.update(f":mode={str(mode_spec.name)}".encode("utf-8"))
    h.update(f":text_score={str(mode_spec.text_score_mode)}".encode("utf-8"))
    feature_key = h.hexdigest()[:16]
    return os.path.join(os.path.abspath(cache_dir), f"prompt_mode_precompute_{feature_key}.npz")


def _build_repo_brand_meta(repo: BrandRepository) -> Dict[str, Any]:
    return {
        "n_brands": int(len(repo.brand_names)),
        "brand_names": list(repo.brand_names),
        "brand_domains": dict(repo.brand_domains),
        "logo_count_per_brand": repo.logo_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().astype(int).tolist(),
        "text_count_per_brand": repo.text_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().astype(int).tolist(),
    }


def _build_brand_repository_for_mode(
    *,
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    encoder: CLIPEncoder,
    max_logo_per_brand: int,
    image_batch_size: int,
    text_batch_size: int,
    mode_spec: PromptModeSpec,
) -> BrandRepository:
    logo_library_dir = os.path.abspath(logo_library_dir)
    dbi5_dir = os.path.abspath(dbi5_dir)
    normal_urls_csv = os.path.abspath(normal_urls_csv)

    brand_names = list_brand_dirs(logo_library_dir)
    if not brand_names:
        raise FileNotFoundError(f"no brands found under logo library: {logo_library_dir}")

    domains_map = load_brand_domains(normal_urls_csv)
    dbi_brand_names = set(list_brand_dirs(dbi5_dir)) if mode_spec.text_source == "dbi5" else set()

    logo_paths_by_brand: List[List[str]] = []
    text_lines_by_brand: List[List[str]] = []
    brand_domains: Dict[str, List[str]] = {}

    for brand in brand_names:
        brand_logo_dir = os.path.join(logo_library_dir, brand)
        logo_paths = list_logo_paths(brand_logo_dir)
        if int(max_logo_per_brand) > 0:
            logo_paths = logo_paths[: int(max_logo_per_brand)]
        if not logo_paths:
            raise ValueError(f"Brand '{brand}' has no logo images")

        brand_key = normalize_brand_key(brand)
        if brand_key not in domains_map:
            raise ValueError(f"Brand '{brand}' missing in normal URLs CSV: {normal_urls_csv}")

        if mode_spec.text_source == "dbi5":
            if brand not in dbi_brand_names:
                raise ValueError(f"Brand '{brand}' missing in DBI-5 directory: {dbi5_dir}")
            try:
                text_lines = read_dbi_texts(os.path.join(dbi5_dir, brand))
            except Exception:
                # Be robust to partially generated prompt libraries.
                text_lines = [str(brand)]
                print(
                    f"[warn] mode={mode_spec.name} missing text file for brand='{brand}' "
                    f"in {os.path.join(dbi5_dir, brand)}; fallback to brand name"
                )
        else:
            text_lines = _build_simple13_prompts(brand)

        if not text_lines:
            raise ValueError(f"Brand '{brand}' has empty text lines for mode={mode_spec.name}")

        logo_paths_by_brand.append(logo_paths)
        text_lines_by_brand.append(text_lines)
        brand_domains[brand] = list(domains_map[brand_key])

    n_brand = len(brand_names)
    max_logo = max(len(x) for x in logo_paths_by_brand)
    max_text = max(len(x) for x in text_lines_by_brand)
    d = int(encoder.embed_dim)

    logo_emb = torch.zeros((n_brand, max_logo, d), dtype=torch.float32, device=encoder.device)
    logo_mask = torch.zeros((n_brand, max_logo), dtype=torch.bool, device=encoder.device)
    text_emb = torch.zeros((n_brand, max_text, d), dtype=torch.float32, device=encoder.device)
    text_mask = torch.zeros((n_brand, max_text), dtype=torch.bool, device=encoder.device)

    flat_logo_items: List[Tuple[int, int, str]] = []
    for bid, paths in enumerate(logo_paths_by_brand):
        for j, path in enumerate(paths):
            flat_logo_items.append((bid, j, path))

    bad_logo_paths: List[str] = []
    for start in range(0, len(flat_logo_items), int(image_batch_size)):
        chunk = flat_logo_items[start : start + int(image_batch_size)]
        imgs: List[Image.Image] = []
        idxs: List[Tuple[int, int]] = []
        for bid, j, path in chunk:
            try:
                with Image.open(path) as im:
                    imgs.append(im.convert("RGB"))
                idxs.append((bid, j))
            except (UnidentifiedImageError, OSError, ValueError):
                bad_logo_paths.append(path)
                continue

        if not imgs:
            continue

        feats = encoder.encode_images(imgs, batch_size=max(1, len(imgs)))
        for k, (bid, j) in enumerate(idxs):
            logo_emb[bid, j] = feats[k]
            logo_mask[bid, j] = True

    if bad_logo_paths:
        print(f"[repo] mode={mode_spec.name} skipped_bad_logo_images={len(bad_logo_paths)}")
        for p in bad_logo_paths[:10]:
            print(f"[repo] mode={mode_spec.name} bad_logo_image={p}")

    valid_logo_per_brand = logo_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().astype(int).tolist()
    brands_without_valid_logo = [brand_names[i] for i, n_valid in enumerate(valid_logo_per_brand) if int(n_valid) <= 0]
    if brands_without_valid_logo:
        preview = ", ".join(brands_without_valid_logo[:10])
        raise RuntimeError(f"Some brands have zero decodable logo images: {preview}")

    flat_text_items: List[Tuple[int, int, str]] = []
    for bid, lines in enumerate(text_lines_by_brand):
        for j, line in enumerate(lines):
            flat_text_items.append((bid, j, line))

    for start in range(0, len(flat_text_items), int(text_batch_size)):
        chunk = flat_text_items[start : start + int(text_batch_size)]
        txts = [item[2] for item in chunk]
        feats = encoder.encode_texts(txts, batch_size=max(1, len(txts)))
        for k, (bid, j, _) in enumerate(chunk):
            text_emb[bid, j] = feats[k]
            text_mask[bid, j] = True

    txt_valid = text_mask.to(dtype=torch.float32).unsqueeze(-1)
    txt_sum = (text_emb * txt_valid).sum(dim=1)
    txt_cnt = txt_valid.sum(dim=1).clamp_min(1.0)
    text_proto = txt_sum / txt_cnt
    text_proto = text_proto / (text_proto.norm(dim=-1, keepdim=True) + 1e-6)

    return BrandRepository(
        brand_names=brand_names,
        logo_emb=logo_emb,
        logo_mask=logo_mask,
        text_emb=text_emb,
        text_mask=text_mask,
        text_proto=text_proto,
        brand_domains=brand_domains,
    )


def compute_similarity_from_query_features_mode(
    *,
    query_data: QueryFeatureData,
    repo: BrandRepository,
    mode_spec: PromptModeSpec,
    progress: bool,
) -> PrecomputedData:
    n = int(query_data.query_feat.shape[0])
    b = int(len(repo.brand_names))
    sim_img = np.zeros((n, b), dtype=np.float32)
    sim_txt = np.zeros((n, b), dtype=np.float32)
    has_logo = np.zeros((n,), dtype=np.uint8)

    resolve_stats = dict(query_data.resolve_stats)
    resolve_stats["sim_fail"] = int(resolve_stats.get("sim_fail", 0))
    failure_examples: List[str] = []

    iterator = iter_with_progress(range(n), description=f"{mode_spec.name}_similarity", enabled=bool(progress))
    for i in iterator:
        valid = np.asarray(query_data.query_mask[i], dtype=np.uint8).astype(bool)
        if not bool(valid.any()):
            continue

        try:
            q_np = np.asarray(query_data.query_feat[i][valid], dtype=np.float32)
            q_t = torch.from_numpy(q_np).to(repo.logo_emb.device)

            sim_img_full = torch.einsum("qd,bkd->qbk", q_t, repo.logo_emb)
            sim_img_full = sim_img_full.masked_fill(~repo.logo_mask.unsqueeze(0), -1e9)
            sim_img_qb = sim_img_full.max(dim=-1).values

            if mode_spec.text_score_mode == "max_per_line":
                sim_txt_full = torch.einsum("qd,btd->qbt", q_t, repo.text_emb)
                sim_txt_full = sim_txt_full.masked_fill(~repo.text_mask.unsqueeze(0), -1e9)
                sim_txt_qb = sim_txt_full.max(dim=-1).values
            else:
                sim_txt_qb = torch.einsum("qd,bd->qb", q_t, repo.text_proto)

            sim_img[i, :] = sim_img_qb.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            sim_txt[i, :] = sim_txt_qb.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            has_logo[i] = 1
        except Exception as e:
            resolve_stats["sim_fail"] = int(resolve_stats.get("sim_fail", 0)) + 1
            if len(failure_examples) < 10:
                failure_examples.append(f"sim_fail idx={i} mode={mode_spec.name} err={type(e).__name__}: {e}")
            continue

    if int(resolve_stats.get("sim_fail", 0)) > 0:
        print(f"[stage2] mode={mode_spec.name} similarity_failures={int(resolve_stats['sim_fail'])}")
        for msg in failure_examples:
            print(f"[stage2] mode={mode_spec.name} failure_example={msg}")

    return PrecomputedData(
        sim_img=sim_img,
        sim_txt=sim_txt,
        has_logo=has_logo,
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=[str(x) for x in query_data.gt_brand_keys],
        screenshot_domains=[str(x) for x in query_data.screenshot_domains],
        resolved_image_paths=[str(x) for x in query_data.resolved_image_paths],
        resolve_stats={str(k): int(v) for k, v in resolve_stats.items()},
    )


def _compute_curve_metrics(*, y_true: np.ndarray, curve_score: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute AUROC and average precision; return None values when only one class is present."""
    out: Dict[str, Optional[float]] = {
        "auroc": None,
        "average_precision": None,
    }
    if int(np.unique(y_true).size) < 2:
        return out

    if roc_auc_score is not None:
        try:
            out["auroc"] = float(roc_auc_score(y_true.astype(np.uint8), curve_score.astype(np.float32)))
        except Exception:
            out["auroc"] = None

    if average_precision_score is not None:
        try:
            out["average_precision"] = float(
                average_precision_score(y_true.astype(np.uint8), curve_score.astype(np.float32))
            )
        except Exception:
            out["average_precision"] = None

    return out


def _compare_row_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a per-mode result dict into a single comparison-table row."""
    metrics = dict(result.get("metrics", {}))
    overview = dict(result.get("brand_level_overview", {}))
    curves = dict(result.get("curve_metrics", {}))

    tp = int(metrics.get("tp", 0) or 0)
    tn = int(metrics.get("tn", 0) or 0)
    fp = int(metrics.get("fp", 0) or 0)
    fn = int(metrics.get("fn", 0) or 0)
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    return {
        "mode": str(result.get("mode", "")),
        "display_name": str(result.get("display_name", "")),
        "image_weight": float(result.get("image_weight", FIXED_IMAGE_WEIGHT) or FIXED_IMAGE_WEIGHT),
        "threshold": float(result.get("threshold", FIXED_DECISION_THRESHOLD) or FIXED_DECISION_THRESHOLD),
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("f1", 0.0) or 0.0),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "brand_acc_on_phish": float(metrics.get("brand_acc_on_phish", 0.0) or 0.0),
        "top1_brand_acc_on_all_phish": float(overview.get("overall_top1_brand_acc_on_all_phish", 0.0) or 0.0),
        "final_brand_acc_on_all_phish": float(overview.get("overall_final_brand_acc_on_all_phish", 0.0) or 0.0),
        "auroc": curves.get("auroc", None),
        "average_precision": curves.get("average_precision", None),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "query_cache_hit": int(result.get("query_cache_hit", 0) or 0),
        "repo_cache_hit": int(result.get("repo_cache_hit", 0) or 0),
        "feature_cache_hit": int(result.get("feature_cache_hit", 0) or 0),
        "repo_cache_path": str(result.get("repo_cache_path", "")),
        "feature_cache_path": str(result.get("feature_cache_path", "")),
    }


def _resolve_mode_list(raw: str) -> List[PromptModeSpec]:
    """Parse a comma/semicolon-separated list of mode names into PromptModeSpec objects."""
    tokens = [str(x).strip().lower() for x in str(raw or "").replace(";", ",").split(",")]
    out: List[PromptModeSpec] = []
    seen = set()
    for token in tokens:
        if not token:
            continue
        if token not in PROMPT_MODE_SPECS:
            raise ValueError(f"Unsupported mode: {token}")
        if token in seen:
            continue
        seen.add(token)
        out.append(PROMPT_MODE_SPECS[token])
    if not out:
        raise ValueError("No valid modes provided")
    return out


def load_query_cache_or_build(
    *,
    query_cache_path: str,
    rebuild_query_cache: bool,
    detections: List[DetectionSample],
    encoder: CLIPEncoder,
    benign_root: str,
    phish_root: str,
    progress: bool,
) -> Tuple[QueryFeatureData, int]:
    """Load an existing query cache or build and save a new one from scratch."""
    cache_hit = 0
    if (not bool(rebuild_query_cache)) and os.path.isfile(query_cache_path):
        query_data = load_query_cache(query_cache_path)
        cache_hit = 1
        print(f"[cache] stage2_query hit: {query_cache_path}")
        return query_data, cache_hit

    print(f"[cache] stage2_query rebuild: {query_cache_path}")
    query_data = precompute_query_features(
        detections=detections,
        encoder=encoder,
        benign_root=benign_root,
        phish_root=phish_root,
        progress=bool(progress),
    )
    save_query_cache(query_cache_path, query_data)
    print(f"[cache] stage2_query saved: {query_cache_path}")
    return query_data, cache_hit


def _evaluate_single_mode(
    *,
    mode_spec: PromptModeSpec,
    out_dir: str,
    cache_dir: str,
    detections: List[DetectionSample],
    det_csv: str,
    query_data: QueryFeatureData,
    query_cache_path: str,
    query_cache_hit: int,
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    brand_ckpt: str,
    benign_root: str,
    phish_root: str,
    encoder: CLIPEncoder,
    max_samples: int,
    max_logo_per_brand: int,
    image_batch_size: int,
    text_batch_size: int,
    rebuild_repo_cache: bool,
    rebuild_feature_cache: bool,
    progress: bool,
    summary_json: str,
) -> Dict[str, Any]:
    """Evaluate one prompt source under the fixed public inference protocol."""
    mode_out_dir = _ensure_dir(os.path.join(os.path.abspath(out_dir), mode_spec.name))
    repo_cache_key = _build_prompt_repo_cache_key(
        brand_ckpt=brand_ckpt,
        logo_library_dir=logo_library_dir,
        normal_urls_csv=normal_urls_csv,
        dbi5_dir=dbi5_dir,
        max_logo_per_brand=max_logo_per_brand,
        mode_spec=mode_spec,
    )
    repo_prefix = "brand_repo" if mode_spec.name == "dbi5" else "prompt_mode_brand_repo"
    repo_cache_path = os.path.join(os.path.abspath(cache_dir), f"{repo_prefix}_{repo_cache_key}.pt")

    repo_cache_hit = 0
    repo: Optional[BrandRepository] = None
    repository_started = time.perf_counter()
    if (not bool(rebuild_repo_cache)) and os.path.isfile(repo_cache_path):
        try:
            repo = load_brand_repository_cache(repo_cache_path, encoder.device)
            repo_cache_hit = 1
            print(f"[cache] mode={mode_spec.name} brand_repo hit: {repo_cache_path}")
        except Exception as exc:
            print(f"[cache] mode={mode_spec.name} brand_repo load failed, rebuilding: {type(exc).__name__}: {exc}")

    if repo is None:
        repo = _build_brand_repository_for_mode(
            logo_library_dir=logo_library_dir,
            dbi5_dir=dbi5_dir,
            normal_urls_csv=normal_urls_csv,
            encoder=encoder,
            max_logo_per_brand=max_logo_per_brand,
            image_batch_size=image_batch_size,
            text_batch_size=text_batch_size,
            mode_spec=mode_spec,
        )
        save_brand_repository_cache(repo_cache_path, repo)
        print(f"[cache] mode={mode_spec.name} brand_repo saved: {repo_cache_path}")

    repository_elapsed = time.perf_counter() - repository_started
    diagnostic_args = argparse.Namespace(
        threshold=FIXED_DECISION_THRESHOLD,
        max_logo_per_brand=int(max_logo_per_brand),
        image_batch_size=int(image_batch_size),
        text_batch_size=int(text_batch_size),
        device=str(encoder.device),
        progress=bool(progress),
    )
    print_runtime_format_info(
        det_csv=det_csv,
        csv_info=inspect_detection_csv_format(det_csv),
        detections=detections,
        repo=repo,
        args=diagnostic_args,
        image_weight=FIXED_IMAGE_WEIGHT,
    )

    feature_cache_path = _make_mode_feature_cache_path(
        cache_dir=cache_dir,
        det_csv=det_csv,
        benign_root=benign_root,
        phish_root=phish_root,
        max_samples=max_samples,
        query_cache_path=query_cache_path,
        repo_cache_key=repo_cache_key,
        mode_spec=mode_spec,
    )
    feature_cache_hit = 0
    pre: Optional[PrecomputedData] = None
    feature_started = time.perf_counter()
    if (not bool(rebuild_feature_cache)) and os.path.isfile(feature_cache_path):
        try:
            pre = load_precomputed_cache(feature_cache_path)
            feature_cache_hit = 1
            print(f"[cache] mode={mode_spec.name} stage2_feature hit: {feature_cache_path}")
        except Exception as exc:
            print(f"[cache] mode={mode_spec.name} stage2_feature load failed, recomputing: {type(exc).__name__}: {exc}")

    similarity_elapsed = 0.0
    if pre is None:
        similarity_started = time.perf_counter()
        pre = compute_similarity_from_query_features_mode(
            query_data=query_data,
            repo=repo,
            mode_spec=mode_spec,
            progress=bool(progress),
        )
        similarity_elapsed = time.perf_counter() - similarity_started
        save_precomputed_cache(feature_cache_path, pre)
        print(f"[cache] mode={mode_spec.name} stage2_feature saved: {feature_cache_path}")

    feature_elapsed = time.perf_counter() - feature_started
    fixed_fusion = compute_fixed_fusion_output(
        pre=pre,
        repo=repo,
        screenshot_urls=[str(detection.url or "") for detection in detections],
    )
    decision_started = time.perf_counter()
    metrics, pred_vec = _evaluate_threshold(
        pre=pre,
        fusion_output=fixed_fusion,
        threshold=FIXED_DECISION_THRESHOLD,
    )
    decision_elapsed = time.perf_counter() - decision_started

    rows = _build_prediction_rows(
        detections=detections,
        pre=pre,
        fusion_output=fixed_fusion,
        threshold=FIXED_DECISION_THRESHOLD,
        pred_vec=pred_vec,
    )
    pred_csv_path = os.path.join(mode_out_dir, "predictions.csv")
    _write_csv_rows(pred_csv_path, rows)
    brand_level_rows, brand_level_overview = _build_brand_level_reports(rows=rows, threshold=FIXED_DECISION_THRESHOLD)
    brand_level_csv_path = os.path.join(mode_out_dir, "brand_level_diagnosis.csv")
    brand_level_json_path = os.path.join(mode_out_dir, "brand_level_diagnosis.json")
    if brand_level_rows:
        _write_csv_rows(brand_level_csv_path, brand_level_rows)
    else:
        with open(brand_level_csv_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
    _save_json(brand_level_json_path, {"overview": brand_level_overview, "brands": brand_level_rows})

    metrics_dict = metrics.to_dict()
    curve_score = fixed_fusion.best_score.astype(np.float32).copy()
    curve_score[pre.has_logo.astype(bool) == 0] = 0.0
    curve_score[fixed_fusion.url_would_be_phish.astype(bool) == 0] = 0.0
    curve_metrics = _compute_curve_metrics(y_true=np.asarray(pre.y_true, dtype=np.uint8), curve_score=curve_score)

    summary = {
        "prompt_mode": {
            "name": mode_spec.name,
            "display_name": mode_spec.display_name,
            "text_source": mode_spec.text_source,
            "text_score_mode": mode_spec.text_score_mode,
        },
        "base_summary_json": os.path.abspath(summary_json),
        "decision": {
            "fusion_strategy": "fixed_linear",
            "image_weight": float(FIXED_IMAGE_WEIGHT),
            "text_weight": float(FIXED_TEXT_WEIGHT),
            "threshold": float(FIXED_DECISION_THRESHOLD),
        },
        "runtime": {
            "repo_build_sec": float(repository_elapsed),
            "feature_precompute_sec": float(feature_elapsed),
            "similarity_from_query_sec": float(similarity_elapsed),
            "fixed_decision_sec": float(decision_elapsed),
            "repo_cache_hit": int(repo_cache_hit),
            "query_cache_hit": int(query_cache_hit),
            "feature_cache_hit": int(feature_cache_hit),
        },
        "metrics": metrics_dict,
        "curve_metrics": curve_metrics,
        "brand_level_overview": brand_level_overview,
        "path_resolution": dict(pre.resolve_stats),
        "outputs": {
            "predictions_csv": os.path.abspath(pred_csv_path),
            "brand_level_diagnosis_csv": os.path.abspath(brand_level_csv_path),
            "brand_level_diagnosis_json": os.path.abspath(brand_level_json_path),
        },
    }

    metrics_path = os.path.join(mode_out_dir, "metrics.json")
    summary_path = os.path.join(mode_out_dir, "summary.json")
    repo_meta_path = os.path.join(mode_out_dir, "brand_repository_meta.json")
    run_args_path = os.path.join(mode_out_dir, "run_args.json")
    _save_json(metrics_path, metrics_dict)
    _save_json(summary_path, summary)
    _save_json(repo_meta_path, _build_repo_brand_meta(repo))
    _save_json(
        run_args_path,
        {
            "mode": mode_spec.name,
            "fixed_image_weight": float(FIXED_IMAGE_WEIGHT),
            "fixed_text_weight": float(FIXED_TEXT_WEIGHT),
            "fixed_threshold": float(FIXED_DECISION_THRESHOLD),
            "rebuild_repo_cache": bool(rebuild_repo_cache),
            "rebuild_feature_cache": bool(rebuild_feature_cache),
        },
    )

    print(
        f"[stage2] mode={mode_spec.name} fixed fusion: "
        f"image={FIXED_IMAGE_WEIGHT:.2f} text={1.0 - FIXED_IMAGE_WEIGHT:.2f} "
        f"threshold={FIXED_DECISION_THRESHOLD:.2f}"
    )
    return {
        "mode": mode_spec.name,
        "display_name": mode_spec.display_name,
        "summary_path": os.path.abspath(summary_path),
        "metrics": metrics_dict,
        "curve_metrics": curve_metrics,
        "brand_level_overview": brand_level_overview,
        "image_weight": float(FIXED_IMAGE_WEIGHT),
        "threshold": float(FIXED_DECISION_THRESHOLD),
        "repo_cache_hit": int(repo_cache_hit),
        "query_cache_hit": int(query_cache_hit),
        "feature_cache_hit": int(feature_cache_hit),
        "repo_cache_path": os.path.abspath(repo_cache_path),
        "feature_cache_path": os.path.abspath(feature_cache_path),
    }

def parse_args() -> argparse.Namespace:
    defaults = resolve_default_paths(__file__)
    default_outputs_dir = os.path.join(defaults.phishkiller_root, "outputs")
    default_cache_dir = os.path.join(default_outputs_dir, "cache")

    parser = argparse.ArgumentParser(
        description=(
            "Compare OpenBrandPhish end-to-end metrics across DBI-5/simple13/simple13_mean "
            "by reusing the existing stage2 query cache."
        )
    )
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--outputs-dir", type=str, default=default_outputs_dir)
    parser.add_argument("--det-csv", type=str, default="")
    parser.add_argument("--brand-ckpt", type=str, default="")
    parser.add_argument("--logo-library-dir", type=str, default="")
    parser.add_argument("--dbi5-dir", type=str, default="")
    parser.add_argument("--normal-urls-csv", type=str, default="")
    parser.add_argument("--benign-root", type=str, default="")
    parser.add_argument("--phish-root", type=str, default="")
    parser.add_argument("--query-cache-path", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default=default_cache_dir)
    parser.add_argument("--modes", type=str, default="dbi5,simple13,simple13_mean")
    parser.add_argument("--max-samples", type=int, default=-10**9)
    parser.add_argument("--max-logo-per-brand", type=int, default=-10**9)
    parser.add_argument("--image-batch-size", type=int, default=-10**9)
    parser.add_argument("--text-batch-size", type=int, default=-10**9)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--out-base-dir", type=str, default=default_outputs_dir)
    parser.add_argument("--rebuild-query-cache", action="store_true")
    parser.add_argument("--rebuild-repo-cache", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = resolve_default_paths(__file__)

    summary_json = str(args.summary_json or "").strip()
    if not summary_json:
        summary_json = _discover_latest_stage2_summary(str(args.outputs_dir))
    summary_json = os.path.abspath(summary_json)
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"summary.json not found: {summary_json}")

    with open(summary_json, "r", encoding="utf-8") as f:
        base_summary = json.load(f)

    base_cfg = dict(base_summary.get("config", {}) or {})
    base_cache = dict(base_summary.get("cache", {}) or {})

    det_csv = os.path.abspath(str(args.det_csv or base_cfg.get("det_csv") or defaults.det_csv))
    brand_ckpt = os.path.abspath(str(args.brand_ckpt or base_cfg.get("brand_ckpt") or defaults.brand_ckpt))
    logo_library_dir = os.path.abspath(str(args.logo_library_dir or base_cfg.get("logo_library_dir") or defaults.logo_library_dir))
    dbi5_dir = os.path.abspath(str(args.dbi5_dir or base_cfg.get("dbi5_dir") or defaults.dbi5_dir))
    normal_urls_csv = os.path.abspath(str(args.normal_urls_csv or base_cfg.get("normal_urls_csv") or defaults.normal_urls_csv))
    benign_root = os.path.abspath(str(args.benign_root or base_cfg.get("benign_root") or defaults.benign_root))
    phish_root = os.path.abspath(str(args.phish_root or base_cfg.get("phish_root") or defaults.phish_root))
    cache_dir = os.path.abspath(
        str(args.cache_dir or base_cache.get("cache_dir") or os.path.join(default_outputs_dir, "cache"))
    )

    max_samples = int(base_cfg.get("max_samples", -1)) if int(args.max_samples) == -10**9 else int(args.max_samples)
    max_logo_per_brand = int(base_cfg.get("max_logo_per_brand", -1)) if int(args.max_logo_per_brand) == -10**9 else int(args.max_logo_per_brand)
    image_batch_size = int(base_cfg.get("image_batch_size", 64)) if int(args.image_batch_size) == -10**9 else int(args.image_batch_size)
    text_batch_size = int(base_cfg.get("text_batch_size", 256)) if int(args.text_batch_size) == -10**9 else int(args.text_batch_size)
    device = str(args.device or base_cfg.get("device") or "cuda:0")

    query_cache_path = str(args.query_cache_path or base_cache.get("query_cache_path") or "").strip()
    if not query_cache_path:
        if not bool(args.rebuild_query_cache):
            raise ValueError("query_cache_path missing in summary.json; pass --query-cache-path or use --rebuild-query-cache")
        query_cache_key = build_query_cache_key(
            brand_ckpt=brand_ckpt,
            det_csv=det_csv,
            max_samples=int(max_samples),
            topk=int(base_cfg.get('topk', 3)),
            benign_root=benign_root,
            phish_root=phish_root,
        )
        query_cache_path = os.path.join(cache_dir, f"stage2_query_{query_cache_key}.npz")
    query_cache_path = os.path.abspath(query_cache_path)

    mode_specs = _resolve_mode_list(str(args.modes))

    out_dir = str(args.out_dir or "").strip()
    if not out_dir:
        out_dir = default_run_dir(str(args.out_base_dir), "prompt_compare")
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)
    _ensure_dir(cache_dir)

    detections = read_detection_top3_csv(det_csv)
    if int(max_samples) > 0:
        detections = detections[: int(max_samples)]
    print(f"[prompt_compare] loaded detections={len(detections)} from {det_csv}")
    print(f"[prompt_compare] summary_json={summary_json}")
    print(f"[prompt_compare] query_cache_path={query_cache_path}")
    print(f"[prompt_compare] modes={[m.name for m in mode_specs]}")

    t_encoder = time.perf_counter()
    encoder = CLIPEncoder(brand_ckpt, device=device)
    print(f"[prompt_compare] CLIP loaded in {time.perf_counter() - t_encoder:.2f}s device={encoder.device}")

    query_data, query_cache_hit = load_query_cache_or_build(
        query_cache_path=query_cache_path,
        rebuild_query_cache=bool(args.rebuild_query_cache),
        detections=detections,
        encoder=encoder,
        benign_root=benign_root,
        phish_root=phish_root,
        progress=bool(args.progress),
    )

    if int(query_data.query_feat.shape[0]) != int(len(detections)):
        raise RuntimeError(
            f"query cache sample count mismatch: cache_n={int(query_data.query_feat.shape[0])} "
            f"vs detections_n={int(len(detections))}. If you changed max_samples or det_csv, rebuild query cache."
        )


    results: List[Dict[str, Any]] = []
    for mode_spec in mode_specs:
        result = _evaluate_single_mode(
            mode_spec=mode_spec,
            out_dir=out_dir,
            cache_dir=cache_dir,
            detections=detections,
            det_csv=det_csv,
            query_data=query_data,
            query_cache_path=query_cache_path,
            query_cache_hit=query_cache_hit,
            logo_library_dir=logo_library_dir,
            dbi5_dir=dbi5_dir,
            normal_urls_csv=normal_urls_csv,
            brand_ckpt=brand_ckpt,
            benign_root=benign_root,
            phish_root=phish_root,
            encoder=encoder,
            max_samples=max_samples,
            max_logo_per_brand=max_logo_per_brand,
            image_batch_size=image_batch_size,
            text_batch_size=text_batch_size,
            rebuild_repo_cache=bool(args.rebuild_repo_cache),
            rebuild_feature_cache=bool(args.rebuild_feature_cache),
            progress=bool(args.progress),
            summary_json=summary_json,
        )
        results.append(result)

    compare_rows = [_compare_row_from_result(r) for r in results]
    compare_csv_path = os.path.join(out_dir, "metrics_compare.csv")
    compare_md_path = os.path.join(out_dir, "metrics_compare.md")
    compare_json_path = os.path.join(out_dir, "metrics_compare.json")
    summary_out_path = os.path.join(out_dir, "prompt_compare_summary.json")
    run_args_path = os.path.join(out_dir, "run_args.json")

    _write_csv_rows(compare_csv_path, compare_rows)
    _write_markdown_table(
        compare_md_path,
        compare_rows,
        columns=[
            "mode",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "brand_acc_on_phish",
            "top1_brand_acc_on_all_phish",
            "final_brand_acc_on_all_phish",
            "auroc",
            "average_precision",
        ],
    )
    _save_json(compare_json_path, {"rows": compare_rows})
    _save_json(
        summary_out_path,
        {
            "base_summary_json": os.path.abspath(summary_json),
            "query_cache_path": os.path.abspath(query_cache_path),
            "assumptions": {
                "dbi5": "Use the existing DBI-5 text files and mean text prototype.",
                "simple13": "Generate 13 prompts from the brand name and use max text similarity across prompts.",
                "simple13_mean": "Generate 13 prompts from the brand name and use the mean prompt prototype.",
            },
            "results": results,
            "table_csv": os.path.abspath(compare_csv_path),
            "table_md": os.path.abspath(compare_md_path),
            "table_json": os.path.abspath(compare_json_path),
        },
    )
    _save_json(
        run_args_path,
        {
            **vars(args),
            "summary_json": os.path.abspath(summary_json),
            "det_csv": det_csv,
            "brand_ckpt": brand_ckpt,
            "logo_library_dir": logo_library_dir,
            "dbi5_dir": dbi5_dir,
            "normal_urls_csv": normal_urls_csv,
            "benign_root": benign_root,
            "phish_root": phish_root,
            "cache_dir": cache_dir,
            "query_cache_path": query_cache_path,
            "max_samples": int(max_samples),
            "max_logo_per_brand": int(max_logo_per_brand),
            "image_batch_size": int(image_batch_size),
            "text_batch_size": int(text_batch_size),
            "device": str(device),
            "resolved_modes": [m.name for m in mode_specs],
            "fixed_image_weight": float(FIXED_IMAGE_WEIGHT),
            "fixed_threshold": float(FIXED_DECISION_THRESHOLD),
        },
    )

    print(f"[prompt_compare] done out_dir={out_dir}")
    print(f"[prompt_compare] compare_csv={compare_csv_path}")
    print(f"[prompt_compare] compare_md={compare_md_path}")
    for row in compare_rows:
        print(
            f"[prompt_compare] {row['mode']}: "
            f"acc={float(row['accuracy']):.4f} "
            f"p={float(row['precision']):.4f} "
            f"r={float(row['recall']):.4f} "
            f"f1={float(row['f1']):.4f} "
            f"brand_acc={float(row['brand_acc_on_phish']):.4f} "
            f"auroc={_fmt_cell(row.get('auroc', ''))} "
            f"ap={_fmt_cell(row.get('average_precision', ''))}"
        )


if __name__ == "__main__":
    main()
