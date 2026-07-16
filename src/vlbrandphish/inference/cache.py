"""Versioned, filesystem-backed caches for the inference workflow."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from ..data.detections import DetectionSample
from ..recognition.repository import BrandRepository
from .types import PrecomputedData, QueryFeatureData


def _hash_file_signature(path: str, h: "hashlib._Hash") -> None:
    path = os.path.abspath(str(path))
    h.update(path.encode("utf-8", errors="ignore"))
    if not os.path.isfile(path):
        h.update(b":missing")
        return
    stat = os.stat(path)
    h.update(f":{int(stat.st_size)}:{int(stat.st_mtime_ns)}".encode("utf-8"))


def _hash_dir_tree_signature(root: str, h: "hashlib._Hash", suffixes: Optional[Tuple[str, ...]] = None) -> None:
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


def _build_repo_cache_key(
    *,
    brand_ckpt: str,
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    max_logo_per_brand: int,
    brand_text_mode: str,
) -> str:
    h = hashlib.md5()
    h.update(b"repo_cache_v5_matched_domain_textmean")
    _hash_file_signature(brand_ckpt, h)
    _hash_file_signature(normal_urls_csv, h)
    _hash_dir_tree_signature(logo_library_dir, h, suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    if str(brand_text_mode or "dbi5").strip().lower() == "dbi5":
        _hash_dir_tree_signature(dbi5_dir, h, suffixes=(".txt",))
    h.update(f":brand_text_mode={str(brand_text_mode or 'dbi5').strip().lower()}".encode("utf-8"))
    h.update(f":max_logo_per_brand={int(max_logo_per_brand)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _save_brand_repo_cache(path: str, repo: BrandRepository) -> None:
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


def _load_brand_repo_cache(path: str, device: torch.device) -> BrandRepository:
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


def _build_feature_cache_key(
    *, det_csv: str, max_samples: int, topk: int, benign_root: str, phish_root: str, repo_cache_key: str
) -> str:
    h = hashlib.md5()
    h.update(b"stage2_feature_cache_v2")
    _hash_file_signature(det_csv, h)
    h.update(str(os.path.abspath(benign_root)).encode("utf-8", errors="ignore"))
    h.update(str(os.path.abspath(phish_root)).encode("utf-8", errors="ignore"))
    h.update(f":max_samples={int(max_samples)}".encode("utf-8"))
    h.update(f":topk={int(topk)}".encode("utf-8"))
    h.update(f":repo={str(repo_cache_key)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _build_query_cache_key(
    *, brand_ckpt: str, det_csv: str, max_samples: int, topk: int, benign_root: str, phish_root: str
) -> str:
    h = hashlib.md5()
    h.update(b"stage2_query_cache_v2")
    _hash_file_signature(brand_ckpt, h)
    _hash_file_signature(det_csv, h)
    h.update(str(os.path.abspath(benign_root)).encode("utf-8", errors="ignore"))
    h.update(str(os.path.abspath(phish_root)).encode("utf-8", errors="ignore"))
    h.update(f":max_samples={int(max_samples)}".encode("utf-8"))
    h.update(f":topk={int(topk)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _save_query_cache(path: str, query_data: QueryFeatureData) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        os.path.abspath(path),
        query_feat=np.asarray(query_data.query_feat, dtype=np.float32),
        route_weights=np.asarray(query_data.route_weights, dtype=np.float32),
        query_mask=np.asarray(query_data.query_mask, dtype=np.uint8),
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=np.asarray([str(value) for value in query_data.gt_brand_keys], dtype=object),
        screenshot_domains=np.asarray([str(value) for value in query_data.screenshot_domains], dtype=object),
        resolved_image_paths=np.asarray([str(value) for value in query_data.resolved_image_paths], dtype=object),
        resolve_stats_json=np.asarray([json.dumps(query_data.resolve_stats, ensure_ascii=False)], dtype=object),
    )


def _load_query_cache(path: str) -> QueryFeatureData:
    archive = np.load(os.path.abspath(path), allow_pickle=True)
    stats = json.loads(str(np.asarray(archive["resolve_stats_json"], dtype=object).reshape(-1)[0]))
    if not isinstance(stats, dict):
        raise RuntimeError("Invalid resolve_stats in query cache")
    query_features = np.asarray(archive["query_feat"], dtype=np.float32)
    return QueryFeatureData(
        query_feat=query_features,
        route_weights=(
            np.asarray(archive["route_weights"], dtype=np.float32)
            if "route_weights" in archive.files
            else np.zeros((int(query_features.shape[0]), int(query_features.shape[1]), 3), dtype=np.float32)
        ),
        query_mask=np.asarray(archive["query_mask"], dtype=np.uint8),
        y_true=np.asarray(archive["y_true"], dtype=np.uint8),
        gt_brand_keys=[str(value) for value in np.asarray(archive["gt_brand_keys"], dtype=object).tolist()],
        screenshot_domains=[str(value) for value in np.asarray(archive["screenshot_domains"], dtype=object).tolist()],
        resolved_image_paths=[str(value) for value in np.asarray(archive["resolved_image_paths"], dtype=object).tolist()],
        resolve_stats={str(key): int(value) for key, value in stats.items()},
    )


def _truncate_detections_topk(detections: List[DetectionSample], topk: int) -> List[DetectionSample]:
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


def _slice_query_data_topk(query_data: QueryFeatureData, topk: int) -> QueryFeatureData:
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
        route_weights=np.asarray(query_data.route_weights[:, : int(topk), :], dtype=np.float32),
        query_mask=np.asarray(query_data.query_mask[:, : int(topk)], dtype=np.uint8),
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=[str(value) for value in query_data.gt_brand_keys],
        screenshot_domains=[str(value) for value in query_data.screenshot_domains],
        resolved_image_paths=[str(value) for value in query_data.resolved_image_paths],
        resolve_stats={str(key): int(value) for key, value in stats.items()},
    )


def _find_compatible_query_cache(*, cache_dir: str, n_samples: int, topk: int) -> Tuple[str, Optional[QueryFeatureData]]:
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
            query_data = _load_query_cache(path)
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
    candidates.sort(key=lambda item: (int(item[0]), str(item[1]).lower()))
    _, path, query_data = candidates[0]
    return str(path), query_data


def _save_precomputed_cache(path: str, precomputed: PrecomputedData) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        os.path.abspath(path),
        sim_img=precomputed.sim_img.astype(np.float32),
        sim_txt=precomputed.sim_txt.astype(np.float32),
        router_alpha=precomputed.router_alpha.astype(np.float32),
        has_logo=precomputed.has_logo.astype(np.uint8),
        y_true=precomputed.y_true.astype(np.uint8),
        gt_brand_keys=np.asarray([str(value) for value in precomputed.gt_brand_keys], dtype=object),
        screenshot_domains=np.asarray([str(value) for value in precomputed.screenshot_domains], dtype=object),
        resolved_image_paths=np.asarray([str(value) for value in precomputed.resolved_image_paths], dtype=object),
        resolve_stats_json=np.asarray([json.dumps(precomputed.resolve_stats, ensure_ascii=False)], dtype=object),
    )


def _load_precomputed_cache(path: str) -> PrecomputedData:
    archive = np.load(os.path.abspath(path), allow_pickle=True)
    stats = json.loads(str(np.asarray(archive["resolve_stats_json"], dtype=object).reshape(-1)[0]))
    if not isinstance(stats, dict):
        raise RuntimeError("Invalid resolve_stats in feature cache")
    sim_img = np.asarray(archive["sim_img"], dtype=np.float32)
    return PrecomputedData(
        sim_img=sim_img,
        sim_txt=np.asarray(archive["sim_txt"], dtype=np.float32),
        router_alpha=(
            np.asarray(archive["router_alpha"], dtype=np.float32)
            if "router_alpha" in archive.files
            else np.full((int(sim_img.shape[0]),), 0.5, dtype=np.float32)
        ),
        has_logo=np.asarray(archive["has_logo"], dtype=np.uint8),
        y_true=np.asarray(archive["y_true"], dtype=np.uint8),
        gt_brand_keys=[str(value) for value in np.asarray(archive["gt_brand_keys"], dtype=object).tolist()],
        screenshot_domains=[str(value) for value in np.asarray(archive["screenshot_domains"], dtype=object).tolist()],
        resolved_image_paths=[str(value) for value in np.asarray(archive["resolved_image_paths"], dtype=object).tolist()],
        resolve_stats={str(key): int(value) for key, value in stats.items()},
    )
