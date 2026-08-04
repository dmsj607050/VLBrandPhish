"""Compute fixed image/text brand evidence from detected logo crops."""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image

from ..components.brand.naming import normalize_brand_key
from ..components.brand.repository import BrandRepository, score_query_features
from ..data.detection_csv import DetectionSample
from ..data.image_paths import crop_logo_region, resolve_image_path
from ..decision.domain import decide_url_policy, registered_domain
from ..runtime.progress import iter_with_progress
from ..runtime.types import FixedFusionOutput, PrecomputedData, QueryFeatureData


# This is the only inference fusion rule used by VLBrandPhish.
FIXED_IMAGE_WEIGHT = 0.8
FIXED_TEXT_WEIGHT = 1.0 - FIXED_IMAGE_WEIGHT
FIXED_DECISION_THRESHOLD = 0.62


def precompute_query_features(
    *,
    detections: List[DetectionSample],
    encoder,
    benign_root: str,
    phish_root: str,
    progress: bool,
) -> QueryFeatureData:
    """Resolve screenshots and encode retained detector crops exactly once.

    Image features are cached independently of the brand repository. Failures are
    counted and reported, but one unreadable screenshot never aborts the full run.
    """
    n = len(detections)
    max_q = max([len(d.bboxes) for d in detections] + [1])
    embedding_dim = int(getattr(encoder, "embed_dim", 768))
    query_feat = np.zeros((n, int(max_q), int(embedding_dim)), dtype=np.float32)
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

    iterator = iter_with_progress(range(n), description="stage2_query_encode", enabled=bool(progress))
    for i in iterator:
        detection = detections[i]
        y_true[i] = 1 if bool(detection.gt_is_phishing) else 0
        gt_brand_keys.append(normalize_brand_key(detection.gt_brand))
        screenshot_domains.append(registered_domain(detection.url))

        direct_ok = bool(detection.image_path) and os.path.isfile(detection.image_path)
        try:
            image_path = resolve_image_path(detection, benign_root=benign_root, phish_root=phish_root)
            if direct_ok:
                resolve_stats["direct_image_path_exists"] += 1
            else:
                resolve_stats["resolved_from_sample_key"] += 1
        except Exception as exc:
            resolve_stats["resolve_fail"] += 1
            image_path = str(detection.image_path or "")
            if len(failure_examples) < 10:
                failure_examples.append(
                    f"resolve_fail sample_key={detection.sample_key} "
                    f"gt_is_phish={int(bool(detection.gt_is_phishing))} "
                    f"err={type(exc).__name__}: {exc}"
                )
            resolved_paths.append(image_path)
            continue
        resolved_paths.append(image_path)

        if not detection.bboxes:
            continue

        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                crops: List[Image.Image] = []
                crop_slots: List[int] = []
                for slot, box in enumerate(detection.bboxes[: int(max_q)]):
                    crop = crop_logo_region(rgb, box)
                    if crop is not None:
                        crops.append(crop)
                        crop_slots.append(int(slot))
        except Exception as exc:
            resolve_stats["open_fail"] += 1
            if len(failure_examples) < 10:
                failure_examples.append(
                    f"open_fail sample_key={detection.sample_key} path={image_path} "
                    f"err={type(exc).__name__}: {exc}"
                )
            continue

        if not crops:
            resolve_stats["crop_fail"] += 1
            if len(failure_examples) < 10:
                failure_examples.append(f"crop_fail sample_key={detection.sample_key} path={image_path}")
            continue

        try:
            features = encoder.encode_images(crops, batch_size=max(1, len(crops)))
            features_np = features.detach().cpu().numpy().astype(np.float32)
            for crop_index, slot in enumerate(crop_slots):
                if int(slot) >= int(max_q):
                    continue
                query_feat[i, int(slot), :] = features_np[int(crop_index)]
                query_mask[i, int(slot)] = 1
        except Exception as exc:
            resolve_stats["encode_fail"] += 1
            if len(failure_examples) < 10:
                failure_examples.append(
                    f"encode_fail sample_key={detection.sample_key} path={image_path} "
                    f"err={type(exc).__name__}: {exc}"
                )
            continue

    total_failures = sum(
        int(resolve_stats[name]) for name in ("resolve_fail", "open_fail", "crop_fail", "encode_fail")
    )
    if total_failures > 0:
        print(
            "[inference] query_encode_failures: "
            f"resolve_fail={int(resolve_stats['resolve_fail'])} "
            f"open_fail={int(resolve_stats['open_fail'])} "
            f"crop_fail={int(resolve_stats['crop_fail'])} "
            f"encode_fail={int(resolve_stats['encode_fail'])}"
        )
        for message in failure_examples:
            print(f"[inference] failure_example: {message}")

    return QueryFeatureData(
        query_feat=query_feat,
        query_mask=query_mask,
        y_true=y_true,
        gt_brand_keys=gt_brand_keys,
        screenshot_domains=screenshot_domains,
        resolved_image_paths=resolved_paths,
        resolve_stats=resolve_stats,
    )


def compute_similarity_from_query_features(
    *,
    query_data: QueryFeatureData,
    repo: BrandRepository,
    progress: bool,
) -> PrecomputedData:
    """Score cached crop embeddings against all brand image and text references."""
    n_samples = int(query_data.query_feat.shape[0])
    n_brands = int(len(repo.brand_names))
    sim_img = np.zeros((n_samples, n_brands), dtype=np.float32)
    sim_txt = np.zeros((n_samples, n_brands), dtype=np.float32)
    has_logo = np.zeros((n_samples,), dtype=np.uint8)

    resolve_stats = dict(query_data.resolve_stats)
    resolve_stats["sim_fail"] = int(resolve_stats.get("sim_fail", 0))
    failure_examples: List[str] = []

    iterator = iter_with_progress(range(n_samples), description="stage2_similarity", enabled=bool(progress))
    for i in iterator:
        valid_crops = np.asarray(query_data.query_mask[i], dtype=np.uint8).astype(bool)
        if not bool(valid_crops.any()):
            continue
        try:
            crop_features = np.asarray(query_data.query_feat[i][valid_crops], dtype=np.float32)
            crop_features_tensor = torch.from_numpy(crop_features).to(repo.logo_emb.device)
            image_scores, text_scores = score_query_features(crop_features_tensor, repo)
            sim_img[i, :] = image_scores.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            sim_txt[i, :] = text_scores.max(dim=0).values.detach().cpu().numpy().astype(np.float32)
            has_logo[i] = 1
        except Exception as exc:
            resolve_stats["sim_fail"] = int(resolve_stats.get("sim_fail", 0)) + 1
            if len(failure_examples) < 10:
                failure_examples.append(f"sim_fail idx={i} err={type(exc).__name__}: {exc}")

    if int(resolve_stats.get("sim_fail", 0)) > 0:
        print(f"[inference] similarity_failures: sim_fail={int(resolve_stats['sim_fail'])}")
        for message in failure_examples:
            print(f"[inference] failure_example: {message}")

    return PrecomputedData(
        sim_img=sim_img,
        sim_txt=sim_txt,
        has_logo=has_logo,
        y_true=np.asarray(query_data.y_true, dtype=np.uint8),
        gt_brand_keys=[str(value) for value in query_data.gt_brand_keys],
        screenshot_domains=[str(value) for value in query_data.screenshot_domains],
        resolved_image_paths=[str(value) for value in query_data.resolved_image_paths],
        resolve_stats={str(key): int(value) for key, value in resolve_stats.items()},
    )


def compute_fixed_fusion_output(
    *,
    pre: PrecomputedData,
    repo: BrandRepository,
    screenshot_urls: List[str],
    url_policy_mode: str = "phishintention",
) -> FixedFusionOutput:
    """Apply the sole fixed linear fusion and the brand-domain consistency rule.

    ``url_policy_mode`` selects the URL validation strategy:
    ``phishintention`` (default) or ``official_plus_overrides_no_whitelist``
    (matches the 02_e2e_vlbrandphish_dataset experiment).
    """
    fused = FIXED_IMAGE_WEIGHT * pre.sim_img + FIXED_TEXT_WEIGHT * pre.sim_txt
    n_samples = int(fused.shape[0])
    n_brands = int(fused.shape[1]) if fused.ndim == 2 else 0

    if n_samples <= 0 or n_brands <= 0:
        zeros_int = np.full((max(n_samples, 0),), -1, dtype=np.int32)
        zeros_float = np.zeros((max(n_samples, 0),), dtype=np.float32)
        zeros_uint = np.zeros((max(n_samples, 0),), dtype=np.uint8)
        return FixedFusionOutput(
            image_weight=FIXED_IMAGE_WEIGHT,
            pred_brand_idx=zeros_int,
            pred_brand_names=["" for _ in range(max(n_samples, 0))],
            best_score=zeros_float,
            best_score_img=zeros_float.copy(),
            best_score_txt=zeros_float.copy(),
            url_would_be_phish=zeros_uint,
            url_match=zeros_uint.copy(),
            brand_match=zeros_uint.copy(),
            url_reason_above_thr=["" for _ in range(max(n_samples, 0))],
        )

    best_idx = np.asarray(np.argmax(fused, axis=1), dtype=np.int32)
    best_score = np.asarray(np.max(fused, axis=1), dtype=np.float32)
    best_score_img = np.asarray(pre.sim_img[np.arange(n_samples), best_idx], dtype=np.float32)
    best_score_txt = np.asarray(pre.sim_txt[np.arange(n_samples), best_idx], dtype=np.float32)

    has_logo = pre.has_logo.astype(bool)
    best_idx[~has_logo] = -1
    best_score[~has_logo] = 0.0
    best_score_img[~has_logo] = 0.0
    best_score_txt[~has_logo] = 0.0

    url_would_be_phish = np.zeros((n_samples,), dtype=np.uint8)
    url_match = np.zeros((n_samples,), dtype=np.uint8)
    brand_match = np.zeros((n_samples,), dtype=np.uint8)
    pred_brand_names = ["" for _ in range(n_samples)]
    url_reason_above_thr = ["" for _ in range(n_samples)]

    for i in range(n_samples):
        if not has_logo[i]:
            continue
        brand_index = int(best_idx[i])
        if brand_index < 0:
            continue

        pred_brand = str(repo.brand_names[brand_index])
        pred_brand_names[i] = pred_brand
        screenshot_url = str(screenshot_urls[i] if i < len(screenshot_urls) else "")
        domain_result = decide_url_policy(
            screenshot_url,
            list(repo.brand_domains.get(pred_brand) or []),
            url_policy_mode=url_policy_mode,
        )
        url_would_be_phish[i] = 1 if bool(domain_result.accept) else 0
        url_match[i] = 0 if bool(domain_result.accept) else 1
        url_reason_above_thr[i] = str(domain_result.reason or "")

        if int(pre.y_true[i]) == 1 and str(pre.gt_brand_keys[i]):
            brand_match[i] = 1 if normalize_brand_key(pred_brand) == pre.gt_brand_keys[i] else 0

    return FixedFusionOutput(
        image_weight=FIXED_IMAGE_WEIGHT,
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