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

from ..config.paths import default_run_dir, resolve_default_paths
from ..core.brands import normalize_brand_key
from ..data.detections import DetectionSample, read_detection_top3_csv
from ..decision.domains import phishintention_domain_accept, registered_domain
from ..recognition.encoders import CLIPEncoder
from ..recognition.repository import BrandRepository, build_brand_repository, score_query_features
from .types import AlphaOutput, Metrics, PrecomputedData, QueryFeatureData

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _iter_items(items, *, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        yield from tqdm(items, desc=desc)
        return
    yield from items


def _safe_crop(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Optional[Image.Image]:
    w, h = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, float(w)))
    x2 = max(0.0, min(x2, float(w)))
    y1 = max(0.0, min(y1, float(h)))
    y2 = max(0.0, min(y2, float(h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def _resolve_benign_path(benign_root: str, sample_key: str) -> str:
    direct = os.path.join(benign_root, sample_key, f"{sample_key}.png")
    if os.path.isfile(direct):
        return direct
    sample_dir = os.path.join(benign_root, sample_key)
    if os.path.isdir(sample_dir):
        for fn in sorted(os.listdir(sample_dir)):
            if os.path.splitext(fn)[0] == sample_key:
                p = os.path.join(sample_dir, fn)
                if os.path.isfile(p):
                    return p
    return ""


def _resolve_phish_path(phish_root: str, sample_key: str, gt_brand: Optional[str]) -> str:
    if "/" in sample_key:
        brand, stem = sample_key.split("/", 1)
    else:
        brand = str(gt_brand or "").strip()
        stem = sample_key

    if not brand or not stem:
        return ""

    screens_dir = os.path.join(phish_root, brand, "screens")
    if not os.path.isdir(screens_dir):
        return ""

    direct = os.path.join(screens_dir, f"{stem}.png")
    if os.path.isfile(direct):
        return direct

    for fn in sorted(os.listdir(screens_dir)):
        if os.path.splitext(fn)[0] == stem:
            p = os.path.join(screens_dir, fn)
            if os.path.isfile(p):
                return p
    return ""


def resolve_image_path(det: DetectionSample, benign_root: str, phish_root: str) -> str:
    if det.image_path and os.path.isfile(det.image_path):
        return os.path.abspath(det.image_path)

    if det.gt_is_phishing:
        p = _resolve_phish_path(phish_root, det.sample_key, det.gt_brand)
    else:
        p = _resolve_benign_path(benign_root, det.sample_key)

    if p and os.path.isfile(p):
        return os.path.abspath(p)

    raise FileNotFoundError(
        "Cannot resolve screenshot path for sample "
        f"sample_key='{det.sample_key}', gt_is_phishing={int(det.gt_is_phishing)}, image_path='{det.image_path}'"
    )


def _inspect_detection_csv_format(path: str) -> Dict[str, object]:
    row_count = 0
    min_cols = 10**9
    max_cols = 0
    first_nonempty: List[str] = []

    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not first_nonempty:
                first_nonempty = [str(x) for x in row]
            n = len(row)
            row_count += 1
            min_cols = min(min_cols, n)
            max_cols = max(max_cols, n)

    if row_count == 0:
        min_cols = 0

    has_header = bool(first_nonempty) and str(first_nonempty[0]).strip().lower() == "image_path"
    return {
        "row_count": int(row_count),
        "has_header": bool(has_header),
        "first_row_ncols": int(len(first_nonempty)),
        "min_cols": int(min_cols),
        "max_cols": int(max_cols),
        "first_row_preview": first_nonempty[:8],
    }


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


def _build_alpha_vector(
    *,
    pre: PrecomputedData,
    fusion_method: str,
    alpha: float,
    gate_k: float,
    gate_b: float,
) -> np.ndarray:
    n = int(pre.sim_img.shape[0]) if pre.sim_img.ndim >= 1 else 0
    method = str(fusion_method or "").strip().lower()
    if method == "dynamic_margin_gate":
        margin_img = _top2_margin(pre.sim_img)
        margin_txt = _top2_margin(pre.sim_txt)
        delta_margin = np.asarray(margin_img - margin_txt, dtype=np.float32)
        alpha_vec = _sigmoid(float(gate_k) * delta_margin + float(gate_b))
    elif method == "router_gate":
        alpha_vec = np.asarray(pre.router_alpha, dtype=np.float32)
    else:
        alpha_vec = np.full((max(n, 0),), float(alpha), dtype=np.float32)
    return alpha_vec


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


def _hash_file_signature(path: str, h: "hashlib._Hash") -> None:
    p = os.path.abspath(str(path))
    h.update(p.encode("utf-8", errors="ignore"))
    if not os.path.isfile(p):
        h.update(b":missing")
        return
    st = os.stat(p)
    h.update(f":{int(st.st_size)}:{int(st.st_mtime_ns)}".encode("utf-8"))


def _hash_dir_tree_signature(root: str, h: "hashlib._Hash", suffixes: Optional[Tuple[str, ...]] = None) -> None:
    r = os.path.abspath(str(root))
    h.update(r.encode("utf-8", errors="ignore"))
    if not os.path.isdir(r):
        h.update(b":missing")
        return

    keep_suffixes = tuple(str(s).lower() for s in (suffixes or tuple()) if str(s))
    for dirpath, dirnames, filenames in os.walk(r):
        dirnames.sort()
        filenames.sort()
        rel_dir = os.path.relpath(dirpath, r)
        h.update(f"D:{rel_dir}".encode("utf-8", errors="ignore"))
        for fn in filenames:
            fnl = str(fn).lower()
            if keep_suffixes and (not fnl.endswith(keep_suffixes)):
                continue
            fp = os.path.join(dirpath, fn)
            rel_fp = os.path.relpath(fp, r)
            try:
                st = os.stat(fp)
                h.update(f"F:{rel_fp}:{int(st.st_size)}:{int(st.st_mtime_ns)}".encode("utf-8", errors="ignore"))
            except OSError:
                h.update(f"F:{rel_fp}:missing".encode("utf-8", errors="ignore"))


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
    _hash_dir_tree_signature(
        logo_library_dir,
        h,
        suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    )
    if str(brand_text_mode or "dbi5").strip().lower() == "dbi5":
        _hash_dir_tree_signature(dbi5_dir, h, suffixes=(".txt",))
    h.update(f":brand_text_mode={str(brand_text_mode or 'dbi5').strip().lower()}".encode("utf-8"))
    h.update(f":max_logo_per_brand={int(max_logo_per_brand)}".encode("utf-8"))
    return h.hexdigest()[:16]


def _save_brand_repo_cache(path: str, repo: BrandRepository) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "brand_names": list(repo.brand_names),
        "logo_emb": repo.logo_emb.detach().cpu(),
        "logo_mask": repo.logo_mask.detach().cpu(),
        "text_emb": repo.text_emb.detach().cpu(),
        "text_mask": repo.text_mask.detach().cpu(),
        "text_proto": repo.text_proto.detach().cpu(),
        "brand_domains": dict(repo.brand_domains),
    }
    torch.save(payload, os.path.abspath(path))


def _load_brand_repo_cache(path: str, device: torch.device) -> BrandRepository:
    payload = torch.load(os.path.abspath(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid repo cache format: {path}")
    for k in ("brand_names", "logo_emb", "logo_mask", "text_emb", "text_mask", "brand_domains"):
        if k not in payload:
            raise KeyError(f"Missing key in repo cache: {k}")

    text_emb = torch.as_tensor(payload["text_emb"]).to(device)
    text_mask = torch.as_tensor(payload["text_mask"]).to(device).bool()
    if "text_proto" in payload:
        text_proto = torch.as_tensor(payload["text_proto"]).to(device)
    else:
        # Backward compatibility for old cache files.
        txt_valid = text_mask.to(dtype=torch.float32).unsqueeze(-1)
        txt_sum = (text_emb * txt_valid).sum(dim=1)
        txt_cnt = txt_valid.sum(dim=1).clamp_min(1.0)
        text_proto = txt_sum / txt_cnt
        text_proto = text_proto / (text_proto.norm(dim=-1, keepdim=True) + 1e-6)

    return BrandRepository(
        brand_names=[str(x) for x in list(payload["brand_names"])],
        logo_emb=torch.as_tensor(payload["logo_emb"]).to(device),
        logo_mask=torch.as_tensor(payload["logo_mask"]).to(device).bool(),
        text_emb=text_emb,
        text_mask=text_mask,
        text_proto=text_proto,
        brand_domains={str(k): [str(x) for x in list(v or [])] for k, v in dict(payload["brand_domains"]).items()},
    )


def _build_feature_cache_key(
    *,
    det_csv: str,
    max_samples: int,
    topk: int,
    benign_root: str,
    phish_root: str,
    repo_cache_key: str,
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
    *,
    brand_ckpt: str,
    det_csv: str,
    max_samples: int,
    topk: int,
    benign_root: str,
    phish_root: str,
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


def _save_query_cache(path: str, q: QueryFeatureData) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        os.path.abspath(path),
        query_feat=np.asarray(q.query_feat, dtype=np.float32),
        route_weights=np.asarray(q.route_weights, dtype=np.float32),
        query_mask=np.asarray(q.query_mask, dtype=np.uint8),
        y_true=np.asarray(q.y_true, dtype=np.uint8),
        gt_brand_keys=np.asarray([str(x) for x in q.gt_brand_keys], dtype=object),
        screenshot_domains=np.asarray([str(x) for x in q.screenshot_domains], dtype=object),
        resolved_image_paths=np.asarray([str(x) for x in q.resolved_image_paths], dtype=object),
        resolve_stats_json=np.asarray([json.dumps(q.resolve_stats, ensure_ascii=False)], dtype=object),
    )


def _load_query_cache(path: str) -> QueryFeatureData:
    z = np.load(os.path.abspath(path), allow_pickle=True)
    resolve_stats_json = str(np.asarray(z["resolve_stats_json"], dtype=object).reshape(-1)[0])
    resolve_stats = json.loads(resolve_stats_json)
    if not isinstance(resolve_stats, dict):
        raise RuntimeError("Invalid resolve_stats in query cache")
    return QueryFeatureData(
        query_feat=np.asarray(z["query_feat"], dtype=np.float32),
        route_weights=(
            np.asarray(z["route_weights"], dtype=np.float32)
            if "route_weights" in z.files
            else np.zeros(
                (
                    int(np.asarray(z["query_feat"]).shape[0]),
                    int(np.asarray(z["query_feat"]).shape[1]),
                    3,
                ),
                dtype=np.float32,
            )
        ),
        query_mask=np.asarray(z["query_mask"], dtype=np.uint8),
        y_true=np.asarray(z["y_true"], dtype=np.uint8),
        gt_brand_keys=[str(x) for x in np.asarray(z["gt_brand_keys"], dtype=object).tolist()],
        screenshot_domains=[str(x) for x in np.asarray(z["screenshot_domains"], dtype=object).tolist()],
        resolved_image_paths=[str(x) for x in np.asarray(z["resolved_image_paths"], dtype=object).tolist()],
        resolve_stats={str(k): int(v) for k, v in resolve_stats.items()},
    )


def _truncate_detections_topk(detections: List[DetectionSample], topk: int) -> List[DetectionSample]:
    k = int(topk)
    if k <= 0:
        return detections
    out: List[DetectionSample] = []
    for det in detections:
        out.append(
            DetectionSample(
                image_path=str(det.image_path),
                url=str(det.url),
                sample_key=str(det.sample_key),
                gt_is_phishing=bool(det.gt_is_phishing),
                gt_brand=det.gt_brand,
                bboxes=list(det.bboxes[:k]),
                det_scores=list(det.det_scores[:k]),
            )
        )
    return out


def _slice_query_data_topk(query_data: QueryFeatureData, topk: int) -> QueryFeatureData:
    k = int(topk)
    if k <= 0:
        return query_data
    if query_data.query_feat.ndim != 3 or query_data.query_mask.ndim != 2:
        raise RuntimeError("Invalid query cache tensor rank")
    cur_q = int(query_data.query_feat.shape[1])
    if cur_q <= k:
        return query_data
    resolve_stats = dict(query_data.resolve_stats)
    resolve_stats["topk_truncated_from"] = int(cur_q)
    resolve_stats["topk_truncated_to"] = int(k)
    return QueryFeatureData(
        query_feat=np.asarray(query_data.query_feat[:, :k, :], dtype=np.float32),
        route_weights=np.asarray(query_data.route_weights[:, :k, :], dtype=np.float32),
        query_mask=np.asarray(query_data.query_mask[:, :k], dtype=np.uint8),
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=[str(x) for x in query_data.gt_brand_keys],
        screenshot_domains=[str(x) for x in query_data.screenshot_domains],
        resolved_image_paths=[str(x) for x in query_data.resolved_image_paths],
        resolve_stats={str(key): int(val) for key, val in resolve_stats.items()},
    )


def _find_compatible_query_cache(*, cache_dir: str, n_samples: int, topk: int) -> Tuple[str, Optional[QueryFeatureData]]:
    try:
        names = sorted(
            [
                os.path.join(os.path.abspath(cache_dir), fn)
                for fn in os.listdir(os.path.abspath(cache_dir))
                if str(fn).startswith("stage2_query_") and str(fn).endswith(".npz")
            ]
        )
    except Exception:
        return "", None

    candidates: List[Tuple[int, str, QueryFeatureData]] = []
    for path in names:
        try:
            query_data = _load_query_cache(path)
        except Exception:
            continue
        if query_data.query_feat.ndim != 3 or query_data.query_mask.ndim != 2:
            continue
        if int(query_data.query_feat.shape[0]) != int(n_samples):
            continue
        if int(query_data.query_mask.shape[0]) != int(n_samples):
            continue
        q = int(query_data.query_feat.shape[1])
        if int(query_data.query_mask.shape[1]) != q:
            continue
        if q < int(topk):
            continue
        candidates.append((q, path, query_data))

    if not candidates:
        return "", None
    candidates.sort(key=lambda item: (int(item[0]), str(item[1]).lower()))
    _, path, query_data = candidates[0]
    return str(path), query_data


def _save_precomputed_cache(path: str, pre: PrecomputedData) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        os.path.abspath(path),
        sim_img=pre.sim_img.astype(np.float32),
        sim_txt=pre.sim_txt.astype(np.float32),
        router_alpha=pre.router_alpha.astype(np.float32),
        has_logo=pre.has_logo.astype(np.uint8),
        y_true=pre.y_true.astype(np.uint8),
        gt_brand_keys=np.asarray([str(x) for x in pre.gt_brand_keys], dtype=object),
        screenshot_domains=np.asarray([str(x) for x in pre.screenshot_domains], dtype=object),
        resolved_image_paths=np.asarray([str(x) for x in pre.resolved_image_paths], dtype=object),
        resolve_stats_json=np.asarray([json.dumps(pre.resolve_stats, ensure_ascii=False)], dtype=object),
    )


def _load_precomputed_cache(path: str) -> PrecomputedData:
    z = np.load(os.path.abspath(path), allow_pickle=True)
    resolve_stats_json = str(np.asarray(z["resolve_stats_json"], dtype=object).reshape(-1)[0])
    resolve_stats = json.loads(resolve_stats_json)
    if not isinstance(resolve_stats, dict):
        raise RuntimeError("Invalid resolve_stats in feature cache")
    return PrecomputedData(
        sim_img=np.asarray(z["sim_img"], dtype=np.float32),
        sim_txt=np.asarray(z["sim_txt"], dtype=np.float32),
        router_alpha=(
            np.asarray(z["router_alpha"], dtype=np.float32)
            if "router_alpha" in z.files
            else np.full((int(np.asarray(z["sim_img"]).shape[0]),), 0.5, dtype=np.float32)
        ),
        has_logo=np.asarray(z["has_logo"], dtype=np.uint8),
        y_true=np.asarray(z["y_true"], dtype=np.uint8),
        gt_brand_keys=[str(x) for x in np.asarray(z["gt_brand_keys"], dtype=object).tolist()],
        screenshot_domains=[str(x) for x in np.asarray(z["screenshot_domains"], dtype=object).tolist()],
        resolved_image_paths=[str(x) for x in np.asarray(z["resolved_image_paths"], dtype=object).tolist()],
        resolve_stats={str(k): int(v) for k, v in resolve_stats.items()},
    )


# Cache serialization and signatures are maintained in their own module. The
# aliases retain the existing internal call surface while downstream scripts
# migrate to ``inference.cache`` directly.
from .cache import (
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


def _precompute_query_features(
    *,
    detections: List[DetectionSample],
    encoder,
    benign_root: str,
    phish_root: str,
    progress: bool,
) -> QueryFeatureData:
    n = len(detections)
    max_q = max([len(d.bboxes) for d in detections] + [1])
    d = int(getattr(encoder, "embed_dim", 768))
    query_feat = np.zeros((n, int(max_q), int(d)), dtype=np.float32)
    route_weights = np.zeros((n, int(max_q), 3), dtype=np.float32)
    query_mask = np.zeros((n, int(max_q)), dtype=np.uint8)

    y_true = np.zeros((n,), dtype=np.uint8)
    gt_brand_keys: List[str] = []
    screenshot_domains: List[str] = []
    resolved_paths: List[str] = []

    resolve_stats = {
        "direct_image_path_exists": 0,
        "resolved_from_sample_key": 0,
        "resolve_fail": 0,
        "open_fail": 0,
        "crop_fail": 0,
        "encode_fail": 0,
    }
    failure_examples: List[str] = []

    iterator = _iter_items(range(n), desc="stage2_query_encode", enabled=bool(progress))
    for i in iterator:
        det = detections[i]
        y_true[i] = 1 if bool(det.gt_is_phishing) else 0
        gt_brand_keys.append(normalize_brand_key(det.gt_brand))
        screenshot_domains.append(registered_domain(det.url))

        direct_ok = bool(det.image_path) and os.path.isfile(det.image_path)
        try:
            image_path = resolve_image_path(det, benign_root=benign_root, phish_root=phish_root)
            if direct_ok:
                resolve_stats["direct_image_path_exists"] += 1
            else:
                resolve_stats["resolved_from_sample_key"] += 1
        except Exception as e:
            resolve_stats["resolve_fail"] += 1
            image_path = str(det.image_path or "")
            if len(failure_examples) < 10:
                failure_examples.append(
                    f"resolve_fail sample_key={det.sample_key} gt_is_phish={int(bool(det.gt_is_phishing))} "
                    f"err={type(e).__name__}: {e}"
                )
            resolved_paths.append(image_path)
            continue
        resolved_paths.append(image_path)

        if not det.bboxes:
            continue

        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                crops: List[Image.Image] = []
                crop_slots: List[int] = []
                for j, box in enumerate(det.bboxes[: int(max_q)]):
                    crop = _safe_crop(rgb, box)
                    if crop is not None:
                        crops.append(crop)
                        crop_slots.append(int(j))
        except Exception as e:
            resolve_stats["open_fail"] += 1
            if len(failure_examples) < 10:
                failure_examples.append(
                    f"open_fail sample_key={det.sample_key} path={image_path} err={type(e).__name__}: {e}"
                )
            continue

        if not crops:
            resolve_stats["crop_fail"] += 1
            if len(failure_examples) < 10:
                failure_examples.append(f"crop_fail sample_key={det.sample_key} path={image_path}")
            continue

        try:
            feats = encoder.encode_images(crops, batch_size=max(1, len(crops)))
            route_query_feats = encoder.encode_query_images(crops, batch_size=max(1, len(crops)))
            route_w = encoder.route_weights_from_query_features(route_query_feats).detach().cpu().numpy().astype(np.float32)
            feats_np = feats.detach().cpu().numpy().astype(np.float32)
            for k, slot in enumerate(crop_slots):
                if int(slot) >= int(max_q):
                    continue
                query_feat[i, int(slot), :] = feats_np[int(k)]
                route_weights[i, int(slot), :] = route_w[int(k)]
                query_mask[i, int(slot)] = 1
        except Exception as e:
            resolve_stats["encode_fail"] += 1
            if len(failure_examples) < 10:
                failure_examples.append(
                    f"encode_fail sample_key={det.sample_key} path={image_path} err={type(e).__name__}: {e}"
                )
            continue

    total_fail = (
        int(resolve_stats["resolve_fail"])
        + int(resolve_stats["open_fail"])
        + int(resolve_stats["crop_fail"])
        + int(resolve_stats["encode_fail"])
    )
    if total_fail > 0:
        print(
            "[inference] query_encode_failures: "
            f"resolve_fail={int(resolve_stats['resolve_fail'])} "
            f"open_fail={int(resolve_stats['open_fail'])} "
            f"crop_fail={int(resolve_stats['crop_fail'])} "
            f"encode_fail={int(resolve_stats['encode_fail'])}"
        )
        for msg in failure_examples:
            print(f"[inference] failure_example: {msg}")

    return QueryFeatureData(
        query_feat=query_feat,
        route_weights=route_weights,
        query_mask=query_mask,
        y_true=y_true,
        gt_brand_keys=gt_brand_keys,
        screenshot_domains=screenshot_domains,
        resolved_image_paths=resolved_paths,
        resolve_stats=resolve_stats,
    )


def _compute_similarity_from_query_features(
    *,
    query_data: QueryFeatureData,
    repo: BrandRepository,
    progress: bool,
) -> PrecomputedData:
    n = int(query_data.query_feat.shape[0])
    b = int(len(repo.brand_names))
    sim_img = np.zeros((n, b), dtype=np.float32)
    sim_txt = np.zeros((n, b), dtype=np.float32)
    router_alpha = np.full((n,), 0.5, dtype=np.float32)
    has_logo = np.zeros((n,), dtype=np.uint8)

    resolve_stats = dict(query_data.resolve_stats)
    resolve_stats["sim_fail"] = int(resolve_stats.get("sim_fail", 0))
    failure_examples: List[str] = []

    iterator = _iter_items(range(n), desc="stage2_similarity", enabled=bool(progress))
    for i in iterator:
        valid = np.asarray(query_data.query_mask[i], dtype=np.uint8).astype(bool)
        if not bool(valid.any()):
            continue
        try:
            q_np = np.asarray(query_data.query_feat[i][valid], dtype=np.float32)
            q_t = torch.from_numpy(q_np).to(repo.logo_emb.device)
            _, sim_img_qb, sim_txt_qb = score_query_features(q_t, repo, alpha=0.5)
            sim_img[i, :] = sim_img_qb.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            sim_txt[i, :] = sim_txt_qb.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            route_w_np = np.asarray(query_data.route_weights[i][valid], dtype=np.float32)
            if route_w_np.ndim == 2 and route_w_np.shape[1] >= 2:
                router_alpha[i] = float(np.clip(route_w_np[:, 1].mean(), 0.0, 1.0))
            has_logo[i] = 1
        except Exception as e:
            resolve_stats["sim_fail"] = int(resolve_stats.get("sim_fail", 0)) + 1
            if len(failure_examples) < 10:
                failure_examples.append(f"sim_fail idx={i} err={type(e).__name__}: {e}")
            continue

    if int(resolve_stats.get("sim_fail", 0)) > 0:
        print(f"[inference] similarity_failures: sim_fail={int(resolve_stats['sim_fail'])}")
        for msg in failure_examples:
            print(f"[inference] failure_example: {msg}")

    return PrecomputedData(
        sim_img=sim_img,
        sim_txt=sim_txt,
        router_alpha=router_alpha,
        has_logo=has_logo,
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=[str(x) for x in query_data.gt_brand_keys],
        screenshot_domains=[str(x) for x in query_data.screenshot_domains],
        resolved_image_paths=[str(x) for x in query_data.resolved_image_paths],
        resolve_stats={str(k): int(v) for k, v in resolve_stats.items()},
    )


def _precompute_similarity_mats(
    *,
    detections: List[DetectionSample],
    encoder,
    repo: BrandRepository,
    benign_root: str,
    phish_root: str,
    progress: bool,
) -> PrecomputedData:
    q = _precompute_query_features(
        detections=detections,
        encoder=encoder,
        benign_root=benign_root,
        phish_root=phish_root,
        progress=progress,
    )
    return _compute_similarity_from_query_features(query_data=q, repo=repo, progress=progress)


def _compute_alpha_output(
    *,
    fusion_method: str,
    alpha: float,
    gate_k: float,
    gate_b: float,
    pre: PrecomputedData,
    repo: BrandRepository,
    screenshot_urls: List[str],
) -> AlphaOutput:
    alpha_vec = _build_alpha_vector(
        pre=pre,
        fusion_method=str(fusion_method),
        alpha=float(alpha),
        gate_k=float(gate_k),
        gate_b=float(gate_b),
    )
    alpha_col = alpha_vec.reshape(-1, 1)
    fused = alpha_col * pre.sim_img + (1.0 - alpha_col) * pre.sim_txt
    n = int(fused.shape[0])
    b = int(fused.shape[1]) if fused.ndim == 2 else 0
    if n <= 0 or b <= 0:
        zeros_i = np.full((max(n, 0),), -1, dtype=np.int32)
        zeros_f = np.zeros((max(n, 0),), dtype=np.float32)
        zeros_u = np.zeros((max(n, 0),), dtype=np.uint8)
        return AlphaOutput(
            fusion_method=str(fusion_method),
            alpha=float(alpha),
            gate_k=float(gate_k),
            gate_b=float(gate_b),
            alpha_vec=zeros_f.copy(),
            pred_brand_idx=zeros_i,
            pred_brand_names=["" for _ in range(max(n, 0))],
            best_score=zeros_f,
            best_score_img=zeros_f.copy(),
            best_score_txt=zeros_f.copy(),
            url_would_be_phish=zeros_u,
            url_match=zeros_u.copy(),
            brand_match=zeros_u.copy(),
            url_reason_above_thr=["" for _ in range(max(n, 0))],
        )

    best_idx = np.asarray(np.argmax(fused, axis=1), dtype=np.int32)
    best_score = np.asarray(np.max(fused, axis=1), dtype=np.float32)
    best_score_img = np.asarray(pre.sim_img[np.arange(n), best_idx], dtype=np.float32)
    best_score_txt = np.asarray(pre.sim_txt[np.arange(n), best_idx], dtype=np.float32)

    has_logo_bool = pre.has_logo.astype(bool)
    best_idx[~has_logo_bool] = -1
    best_score[~has_logo_bool] = 0.0
    best_score_img[~has_logo_bool] = 0.0
    best_score_txt[~has_logo_bool] = 0.0

    url_would_be_phish = np.zeros((n,), dtype=np.uint8)
    url_match = np.zeros((n,), dtype=np.uint8)
    brand_match = np.zeros((n,), dtype=np.uint8)
    pred_brand_names = ["" for _ in range(n)]
    url_reason_above_thr = ["" for _ in range(n)]

    for i in range(n):
        if not has_logo_bool[i]:
            continue

        idx = int(best_idx[i])
        if idx < 0:
            continue

        pred_brand = str(repo.brand_names[idx])
        pred_brand_names[i] = pred_brand
        shot_url = str(screenshot_urls[i] if i < len(screenshot_urls) else "")
        matched_domain = list(repo.brand_domains.get(pred_brand) or [])
        domain_result = phishintention_domain_accept(shot_url, matched_domain)
        url_is_phish = bool(domain_result.accept)
        url_is_match = not bool(domain_result.accept)
        url_reason = str(domain_result.reason or "")
        url_would_be_phish[i] = 1 if bool(url_is_phish) else 0
        url_match[i] = 1 if bool(url_is_match) else 0
        url_reason_above_thr[i] = str(url_reason or "")

        if int(pre.y_true[i]) == 1 and str(pre.gt_brand_keys[i]):
            brand_match[i] = 1 if normalize_brand_key(pred_brand) == pre.gt_brand_keys[i] else 0

    return AlphaOutput(
        fusion_method=str(fusion_method),
        alpha=float(alpha_vec.mean()) if alpha_vec.size > 0 else float(alpha),
        gate_k=float(gate_k),
        gate_b=float(gate_b),
        alpha_vec=alpha_vec,
        pred_brand_idx=best_idx,
        pred_brand_names=pred_brand_names,
        best_score=best_score,
        best_score_img=best_score_img,
        best_score_txt=best_score_txt,
        url_would_be_phish=url_would_be_phish,
        url_match=url_match,
        brand_match=brand_match,
        url_reason_above_thr=url_reason_above_thr,
    )


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
    t0 = time.perf_counter()
    encoder = CLIPEncoder(brand_ckpt_abs, device=str(args.device))
    print(f"[inference] encoder loaded in {time.perf_counter() - t0:.2f}s")

    repo_cache_key = _build_repo_cache_key(
        brand_ckpt=brand_ckpt_abs,
        logo_library_dir=logo_library_dir_abs,
        dbi5_dir=dbi5_dir_abs,
        normal_urls_csv=normal_urls_csv_abs,
        max_logo_per_brand=int(args.max_logo_per_brand),
        brand_text_mode=str(args.brand_text_mode),
    )
    repo_cache_path = os.path.join(cache_dir, f"brand_repo_{repo_cache_key}.pt")
    repo_cache_hit = False
    t1 = time.perf_counter()
    repo: Optional[BrandRepository] = None
    if (not bool(args.no_repo_cache)) and (not bool(args.rebuild_repo_cache)) and os.path.isfile(repo_cache_path):
        try:
            repo = _load_brand_repo_cache(repo_cache_path, encoder.device)
            repo_cache_hit = True
            print(f"[cache] brand_repo hit: {repo_cache_path}")
        except Exception as e:
            print(f"[cache] brand_repo load failed, rebuilding: {type(e).__name__}: {e}")

    if repo is None:
        repo = build_brand_repository(
            logo_library_dir=logo_library_dir_abs,
            dbi5_dir=dbi5_dir_abs,
            normal_urls_csv=normal_urls_csv_abs,
            encoder=encoder,
            max_logo_per_brand=int(args.max_logo_per_brand),
            image_batch_size=int(args.image_batch_size),
            text_batch_size=int(args.text_batch_size),
            brand_text_mode=str(args.brand_text_mode),
        )
        if not bool(args.no_repo_cache):
            try:
                _save_brand_repo_cache(repo_cache_path, repo)
                print(f"[cache] brand_repo saved: {repo_cache_path}")
            except Exception as e:
                print(f"[cache] brand_repo save failed: {type(e).__name__}: {e}")

    print(
        f"[inference] brand repository ready in {time.perf_counter() - t1:.2f}s "
        f"brands={len(repo.brand_names)} cache_hit={int(repo_cache_hit)}"
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
