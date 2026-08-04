"""Versioned, filesystem-backed caches for the inference workflow."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from ..data.detection_csv import DetectionSample
from ..components.brand.repository import BrandRepository
from .types import PrecomputedData, QueryFeatureData


# Input signatures deliberately use metadata, not content, so cache-key creation is fast.

def hash_file_signature(path: str, h: "hashlib._Hash") -> None:
    """Hash a file path and metadata without reading the full file contents."""
    path = os.path.abspath(str(path))
    h.update(path.encode("utf-8", errors="ignore"))
    if not os.path.isfile(path):
        h.update(b":missing")
        return
    stat = os.stat(path)
    h.update(f":{int(stat.st_size)}:{int(stat.st_mtime_ns)}".encode("utf-8"))


def hash_dir_tree_signature(root: str, h: "hashlib._Hash", suffixes: Optional[Tuple[str, ...]] = None) -> None:
    """Hash a deterministic directory listing for cache invalidation."""
    root = os.path.abspath(str(root))
    h.update(root.encode("utf-8", errors="ignore"))
    if not os.path.isdir(root):
        h.update(b":missing")
        return

    allowed = tuple(str(s).lower() for s in (suffixes or ()) if str(s))
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        h.update(f"D:{os.path.relpath(directory, root)}".encode("utf-8", errors="ignore"))
        for filename in file_names:
            if allowed and not str(filename).lower().endswith(allowed):
                continue
            path = os.path.join(directory, filename)
            relative_path = os.path.relpath(path, root)
            try:
                stat = os.stat(path)
                h.update(f"F:{relative_path}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}".encode("utf-8", errors="ignore"))
            except OSError:
                h.update(f"F:{relative_path}:missing".encode("utf-8", errors="ignore"))


def build_repository_cache_key(
    *,
    brand_ckpt: str,
    model_dir: str = "",
    logo_library_dir: str = "",
    dbi5_dir: str,
    normal_urls_csv: str,
    max_logo_per_brand: int,
    brand_text_mode: str,
    brands_overlap_csv: str = "",
    extra_suffix: str = "",
) -> str:
    """Build a content-hash key that invalidates the brand-repository cache.

    The key folds in file signatures (size + mtime) of the checkpoint, model dir,
    normal-urls CSV, the logo library image tree, and (for dbi5 mode) the prompt
    text tree, plus the text mode and logo cap. Any change in these inputs yields
    a different key, forcing a fresh repository build.

    ``extra_suffix`` is an optional caller-controlled string folded into the
    hash after all content signatures; it is used for restrict_brands_to or
    other experiment-level parameterizations that would otherwise collide with
    a full-library cache.
    """
    h = hashlib.md5()
    # Version the cache key whenever repository feature semantics change.
    h.update(b"repo_cache_v8_extra_suffix_plus_restrict_brands")
    hash_file_signature(brand_ckpt, h)
    if model_dir:
        hash_dir_tree_signature(
            model_dir,
            h,
            suffixes=(".json", ".txt", ".model", ".safetensors", ".bin"),
        )
    hash_file_signature(normal_urls_csv, h)
    hash_dir_tree_signature(logo_library_dir, h, suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    if str(brand_text_mode or "dbi5").strip().lower() == "dbi5":
        hash_dir_tree_signature(dbi5_dir, h, suffixes=(".txt",))
    if brands_overlap_csv:
        hash_file_signature(brands_overlap_csv, h)
    h.update(f":brand_text_mode={str(brand_text_mode or 'dbi5').strip().lower()}".encode("utf-8"))
    h.update(f":max_logo_per_brand={int(max_logo_per_brand)}".encode("utf-8"))
    h.update(f":brands_overlap_csv={'1' if brands_overlap_csv else '0'}".encode("utf-8"))
    if extra_suffix:
        h.update(f":extra_suffix={str(extra_suffix)}".encode("utf-8"))
    return h.hexdigest()[:16]


# Repository cache serializes CPU tensors so it can later load onto any selected device.

def save_brand_repository_cache(path: str, repo: BrandRepository) -> None:
    """Persist an encoded brand repository for reuse in later runs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "brand_names": list(repo.brand_names),
            "logo_emb": repo.logo_emb.detach().cpu(),
            "logo_mask": repo.logo_mask.detach().cpu(),
            "text_emb": repo.text_emb.detach().cpu(),
            "text_mask": repo.text_mask.detach().cpu(),
            "text_proto": repo.text_proto.detach().cpu(),
            "brand_domains": dict(repo.brand_domains),
        },
        os.path.abspath(path),
    )


def load_brand_repository_cache(path: str, device: torch.device) -> BrandRepository:
    """Load and validate a previously encoded brand repository cache."""
    payload = torch.load(os.path.abspath(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid repository cache format: {path}")
    required = ("brand_names", "logo_emb", "logo_mask", "text_emb", "text_mask", "brand_domains")
    for key in required:
        if key not in payload:
            raise KeyError(f"Missing key in repository cache: {key}")

    text_embeddings = torch.as_tensor(payload["text_emb"]).to(device)
    text_mask = torch.as_tensor(payload["text_mask"]).to(device).bool()
    if "text_proto" in payload:
        text_prototype = torch.as_tensor(payload["text_proto"]).to(device)
    else:
        valid = text_mask.to(dtype=torch.float32).unsqueeze(-1)
        text_prototype = (text_embeddings * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        text_prototype = text_prototype / (text_prototype.norm(dim=-1, keepdim=True) + 1e-6)

    return BrandRepository(
        brand_names=[str(value) for value in list(payload["brand_names"])],
        logo_emb=torch.as_tensor(payload["logo_emb"]).to(device),
        logo_mask=torch.as_tensor(payload["logo_mask"]).to(device).bool(),
        text_emb=text_embeddings,
        text_mask=text_mask,
        text_proto=text_prototype,
        brand_domains={str(key): [str(value) for value in list(values or [])] for key, values in dict(payload["brand_domains"]).items()},
    )


# Query and feature caches are separate: changing brand references need not re-encode screenshots.

def build_feature_cache_key(
    *, det_csv: str, max_samples: int, topk: int, benign_root: str, phish_root: str, repo_cache_key: str
) -> str:
    """Build a cache key for screenshot features and their repository context."""
    h = hashlib.md5()
    h.update(b"stage2_feature_cache_v3_fixed_linear")
    hash_file_signature(det_csv, h)
    h.update(str(os.path.abspath(benign_root)).encode("utf-8", errors="ignore"))
    h.update(str(os.path.abspath(phish_root)).encode("utf-8", errors="ignore"))
    h.update(f":max_samples={int(max_samples)}".encode("utf-8"))
    h.update(f":topk={int(topk)}".encode("utf-8"))
    h.update(f":repo={str(repo_cache_key)}".encode("utf-8"))
    return h.hexdigest()[:16]


def build_query_cache_key(
    *, brand_ckpt: str, det_csv: str, max_samples: int, topk: int, benign_root: str, phish_root: str
) -> str:
    """Build a cache key for query crops before repository scoring."""
    h = hashlib.md5()
    h.update(b"stage2_query_cache_v3_fixed_linear")
    hash_file_signature(brand_ckpt, h)
    hash_file_signature(det_csv, h)
    h.update(str(os.path.abspath(benign_root)).encode("utf-8", errors="ignore"))
    h.update(str(os.path.abspath(phish_root)).encode("utf-8", errors="ignore"))
    h.update(f":max_samples={int(max_samples)}".encode("utf-8"))
    h.update(f":topk={int(topk)}".encode("utf-8"))
    return h.hexdigest()[:16]


def save_query_cache(path: str, query_data: QueryFeatureData) -> None:
    """Persist query crop features in a compact NPZ file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        os.path.abspath(path),
        query_feat=np.asarray(query_data.query_feat, dtype=np.float32),
        query_mask=np.asarray(query_data.query_mask, dtype=np.uint8),
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=np.asarray([str(value) for value in query_data.gt_brand_keys], dtype=object),
        screenshot_domains=np.asarray([str(value) for value in query_data.screenshot_domains], dtype=object),
        resolved_image_paths=np.asarray([str(value) for value in query_data.resolved_image_paths], dtype=object),
        resolve_stats_json=np.asarray([json.dumps(query_data.resolve_stats, ensure_ascii=False)], dtype=object),
    )


def load_query_cache(path: str) -> QueryFeatureData:
    """Load query crop features from an NPZ cache."""
    archive = np.load(os.path.abspath(path), allow_pickle=True)
    stats = json.loads(str(np.asarray(archive["resolve_stats_json"], dtype=object).reshape(-1)[0]))
    if not isinstance(stats, dict):
        raise RuntimeError("Invalid resolve_stats in query cache")
    query_features = np.asarray(archive["query_feat"], dtype=np.float32)
    return QueryFeatureData(
        query_feat=query_features,
        query_mask=np.asarray(archive["query_mask"], dtype=np.uint8),
        y_true=np.asarray(archive["y_true"], dtype=np.uint8),
        gt_brand_keys=[str(value) for value in np.asarray(archive["gt_brand_keys"], dtype=object).tolist()],
        screenshot_domains=[str(value) for value in np.asarray(archive["screenshot_domains"], dtype=object).tolist()],
        resolved_image_paths=[str(value) for value in np.asarray(archive["resolved_image_paths"], dtype=object).tolist()],
        resolve_stats={str(key): int(value) for key, value in stats.items()},
    )


def truncate_detections_topk(detections: List[DetectionSample], topk: int) -> List[DetectionSample]:
    """Return copies of detections retaining no more than the requested top-k boxes."""
    if int(topk) <= 0:
        return detections
    return [
        DetectionSample(
            image_path=str(detection.image_path),
            url=str(detection.url),
            sample_key=str(detection.sample_key),
            gt_is_phishing=bool(detection.gt_is_phishing),
            gt_brand=detection.gt_brand,
            bboxes=list(detection.bboxes[: int(topk)]),
            det_scores=list(detection.det_scores[: int(topk)]),
        )
        for detection in detections
    ]


def slice_query_data_topk(query_data: QueryFeatureData, topk: int) -> QueryFeatureData:
    """Slice cached query tensors to a requested number of detector boxes."""
    if int(topk) <= 0:
        return query_data
    if query_data.query_feat.ndim != 3 or query_data.query_mask.ndim != 2:
        raise RuntimeError("Invalid query cache tensor rank")
    if int(query_data.query_feat.shape[1]) <= int(topk):
        return query_data
    stats = dict(query_data.resolve_stats)
    stats["topk_truncated_from"] = int(query_data.query_feat.shape[1])
    stats["topk_truncated_to"] = int(topk)
    return QueryFeatureData(
        query_feat=np.asarray(query_data.query_feat[:, : int(topk), :], dtype=np.float32),
        query_mask=np.asarray(query_data.query_mask[:, : int(topk)], dtype=np.uint8),
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=[str(value) for value in query_data.gt_brand_keys],
        screenshot_domains=[str(value) for value in query_data.screenshot_domains],
        resolved_image_paths=[str(value) for value in query_data.resolved_image_paths],
        resolve_stats={str(key): int(value) for key, value in stats.items()},
    )


def find_compatible_query_cache(*, cache_dir: str, n_samples: int, topk: int) -> Tuple[str, Optional[QueryFeatureData]]:
    """Search the cache dir for a query cache with the same sample count and a topk >= requested.

    Allows reusing a previously saved query cache when only the ``--topk`` argument
    shrank, avoiding a costly re-encode of all screenshots. The caller slices the
    found cache down to the requested topk via :func:`slice_query_data_topk`.
    """
    try:
        paths = sorted(
            os.path.join(os.path.abspath(cache_dir), filename)
            for filename in os.listdir(os.path.abspath(cache_dir))
            if str(filename).startswith("stage2_query_") and str(filename).endswith(".npz")
        )
    except OSError:
        return "", None

    candidates: List[Tuple[int, str, QueryFeatureData]] = []
    for path in paths:
        try:
            query_data = load_query_cache(path)
        except Exception:
            continue
        if query_data.query_feat.ndim != 3 or query_data.query_mask.ndim != 2:
            continue
        if int(query_data.query_feat.shape[0]) != int(n_samples) or int(query_data.query_mask.shape[0]) != int(n_samples):
            continue
        cached_topk = int(query_data.query_feat.shape[1])
        if int(query_data.query_mask.shape[1]) != cached_topk or cached_topk < int(topk):
            continue
        candidates.append((cached_topk, path, query_data))
    if not candidates:
        return "", None

    _, path, query_data = candidates[0]
    return str(path), query_data


def save_precomputed_cache(path: str, precomputed: PrecomputedData) -> None:
    """Persist precomputed brand similarities in a compact NPZ cache."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        os.path.abspath(path),
        sim_img=precomputed.sim_img.astype(np.float32),
        sim_txt=precomputed.sim_txt.astype(np.float32),
        has_logo=precomputed.has_logo.astype(np.uint8),
        y_true=precomputed.y_true.astype(np.uint8),
        gt_brand_keys=np.asarray([str(value) for value in precomputed.gt_brand_keys], dtype=object),
        screenshot_domains=np.asarray([str(value) for value in precomputed.screenshot_domains], dtype=object),
        resolved_image_paths=np.asarray([str(value) for value in precomputed.resolved_image_paths], dtype=object),
        resolve_stats_json=np.asarray([json.dumps(precomputed.resolve_stats, ensure_ascii=False)], dtype=object),
    )


def load_precomputed_cache(path: str) -> PrecomputedData:
    """Load precomputed brand similarities from an NPZ cache."""
    archive = np.load(os.path.abspath(path), allow_pickle=True)
    stats = json.loads(str(np.asarray(archive["resolve_stats_json"], dtype=object).reshape(-1)[0]))
    if not isinstance(stats, dict):
        raise RuntimeError("Invalid resolve_stats in feature cache")
    sim_img = np.asarray(archive["sim_img"], dtype=np.float32)
    return PrecomputedData(
        sim_img=sim_img,
        sim_txt=np.asarray(archive["sim_txt"], dtype=np.float32),
        has_logo=np.asarray(archive["has_logo"], dtype=np.uint8),
        y_true=np.asarray(archive["y_true"], dtype=np.uint8),
        gt_brand_keys=[str(value) for value in np.asarray(archive["gt_brand_keys"], dtype=object).tolist()],
        screenshot_domains=[str(value) for value in np.asarray(archive["screenshot_domains"], dtype=object).tolist()],
        resolved_image_paths=[str(value) for value in np.asarray(archive["resolved_image_paths"], dtype=object).tolist()],
        resolve_stats={str(key): int(value) for key, value in stats.items()},
    )
