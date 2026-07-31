"""Compute brand evidence from detected logo crops and shared model components."""

from __future__ import annotations

from typing import List
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from ..components.brand.naming import normalize_brand_key
from ..components.brand.repository import BrandRepository, score_query_features
from .data_contracts import DetectionSample
from .domain_verification import phishintention_domain_accept, registered_domain
from .input_loading import _iter_items, _safe_crop, resolve_image_path
from .types import AlphaOutput, PrecomputedData, QueryFeatureData

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
