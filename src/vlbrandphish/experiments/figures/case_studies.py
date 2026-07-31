from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageDraw

from ...inference.data_contracts import DetectionSample, read_detection_top3_csv
from ...components.brand.repository import _list_logo_paths


CASE_DEFS: Dict[str, Dict[str, Any]] = {
    "case1_text_fail_image_rescue": {
        "display_name": "Case I",
        "row_title": "Text-only fails, image cues rescue the prediction",
        "description_cn": "仅图文失败，加上图图后成功",
        "figure_label_en": "Text-only wrong\nFusion recovers",
        "failure_mode": "text",
        "preferred_gt_brands": ["Facebook", "MetaMask", "OpenSea", "USPS", "myGov", "Roblox", "Produbanco"],
    },
    "case2_image_fail_text_rescue": {
        "display_name": "Case II",
        "row_title": "Image-only fails, text cues rescue the prediction",
        "description_cn": "仅图图失败，加上图文后成功",
        "figure_label_en": "Image-only wrong\nFusion recovers",
        "failure_mode": "image",
        "preferred_gt_brands": ["WhatsApp", "Netflix", "Instagram", "eBay", "DHL", "bet365"],
    },
    "case3_both_fail_fusion_success": {
        "display_name": "Case III",
        "row_title": "Both single branches fail, but fusion succeeds",
        "description_cn": "仅图文失败、仅图图失败，但融合成功",
        "figure_label_en": "Both single branches wrong\nFusion succeeds",
        "failure_mode": "both",
        "preferred_gt_brands": ["Binance", "MetaMask", "WeTransfer", "Kraken", "PayPal", "Facebook", "LeBonCoin"],
    },
}

MANUAL_OVERRIDE_SAMPLE_KEYS: Dict[str, str] = {
    "case1_text_fail_image_rescue": "myGov/myGov_00002",
    "case2_image_fail_text_rescue": "Netflix/Netflix_00038",
    "case3_both_fail_fusion_success": "Binance/BINANCE_00022",
}

COMPACT_BRAND_DISPLAY: Dict[str, str] = {
    "Deutsche Kreditbank": "DKB",
    "Interac Association": "Interac",
    "Meta Platforms, Inc": "Meta",
    "British Telecommunications": "BT",
}


@dataclass(frozen=True)
class ModePrediction:
    mode: str
    pred_idx: int
    pred_brand: str
    correct: bool
    top_score: float
    second_score: float
    margin: float
    query_slot: int
    image_score_at_slot: float
    text_score_at_slot: float
    fusion_score_at_slot: float
    ref_logo_path: str
    ref_logo_score: float


@dataclass(frozen=True)
class SelectedCase:
    case_key: str
    case_display_name: str
    case_title: str
    case_desc_cn: str
    sample_index: int
    sample_key: str
    url: str
    gt_brand: str
    resolved_image_path: str
    n_query_boxes: int
    text_mode: ModePrediction
    image_mode: ModePrediction
    fusion_mode: ModePrediction


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _norm_brand(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _brand_display_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(COMPACT_BRAND_DISPLAY.get(raw, raw))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return np.zeros((arr.shape[0] if arr.ndim >= 1 else 0,), dtype=np.float32)
    if arr.shape[1] == 1:
        return np.asarray(arr[:, 0], dtype=np.float32)
    part = np.partition(arr, kth=arr.shape[1] - 2, axis=1)
    top2 = np.asarray(part[:, -2:], dtype=np.float32)
    top2.sort(axis=1)
    return np.asarray(top2[:, 1] - top2[:, 0], dtype=np.float32)


def _compute_fusion_outputs(summary: Dict[str, Any], sim_img: np.ndarray, sim_txt: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
    sel = dict(summary.get("selection") or {})
    method = str(sel.get("fusion_method", "linear_alpha") or "linear_alpha").strip().lower()
    alpha = float(sel.get("best_alpha", 0.5) or 0.5)
    gate_k = float(sel.get("best_gate_k", 0.0) or 0.0)
    gate_b = float(sel.get("best_gate_b", 0.0) or 0.0)

    if method == "dynamic_margin_gate":
        margin_img = _top2_margin(sim_img)
        margin_txt = _top2_margin(sim_txt)
        alpha_vec = _sigmoid_np(float(gate_k) * (margin_img - margin_txt) + float(gate_b))
    else:
        method = "linear_alpha"
        alpha_vec = np.full((int(sim_img.shape[0]),), float(alpha), dtype=np.float32)

    fused = np.asarray(alpha_vec.reshape(-1, 1) * sim_img + (1.0 - alpha_vec.reshape(-1, 1)) * sim_txt, dtype=np.float32)
    return method, alpha_vec.astype(np.float32), fused


def _discover_latest_stage2_summary(outputs_dir: str) -> str:
    root = Path(outputs_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"outputs dir not found: {root}")

    cands: List[Tuple[float, Path]] = []
    for item in root.iterdir():
        if not item.is_dir() or not item.name.startswith(("inference_", "stage2_")):
            continue
        summary_path = item / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            ts = summary_path.stat().st_mtime
        except OSError:
            ts = 0.0
        cands.append((ts, summary_path))

    if not cands:
        raise FileNotFoundError(f"no stage2 summary found under: {root}")

    cands.sort(key=lambda x: x[0], reverse=True)
    return str(cands[0][1])


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected dict JSON: {path}")
    return obj


def _build_logo_paths_by_brand(logo_library_dir: str, brand_names: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for brand in brand_names:
        out[str(brand)] = [str(Path(p).resolve()) for p in _list_logo_paths(os.path.join(logo_library_dir, str(brand)))]
    return out


def _wrap_lines(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width)) if text else ""


def _imread_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _pil_resample_bicubic() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BICUBIC)
    return int(Image.BICUBIC)


def _resize_to_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty image")
    scale = min(float(max_w) / float(w), float(max_h) / float(h))
    scale = max(scale, 1e-6)
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    pil = Image.fromarray(img)
    pil = pil.resize((new_w, new_h), resample=_pil_resample_bicubic())
    return np.asarray(pil.convert("RGB"))


def _build_query_canvas(
    page_img: np.ndarray, det: DetectionSample, crop_slot: int
) -> Tuple[np.ndarray, Tuple[float, float, float, float] | None]:
    arrow_coords: Tuple[float, float, float, float] | None = None
    if 0 <= int(crop_slot) < len(det.bboxes):
        x1, y1, x2, y2 = [int(max(0, round(v))) for v in det.bboxes[int(crop_slot)]]
        crop = page_img[y1:y2, x1:x2]
        if crop.size <= 0:
            crop = page_img
        bw = max(1, int(x2 - x1))
        bh = max(1, int(y2 - y1))
        pad_l = max(60, int(round(1.3 * bw)))
        pad_r = max(60, int(round(1.3 * bw)))
        pad_t = max(50, int(round(1.0 * bh)))
        pad_b = max(120, int(round(2.0 * bh)))
        cx1 = max(0, int(x1 - pad_l))
        cy1 = max(0, int(y1 - pad_t))
        cx2 = min(int(page_img.shape[1]), int(x2 + pad_r))
        cy2 = min(int(page_img.shape[0]), int(y2 + pad_b))
        context_img = page_img[cy1:cy2, cx1:cx2].copy()
        rx1 = int(x1 - cx1)
        ry1 = int(y1 - cy1)
        rx2 = int(x2 - cx1)
        ry2 = int(y2 - cy1)
        context_pil = Image.fromarray(context_img)
        draw_ctx = ImageDraw.Draw(context_pil)
        draw_ctx.rectangle((rx1, ry1, rx2, ry2), outline="#2E8B57", width=5)
        context_img = np.asarray(context_pil.convert("RGB"))
    else:
        crop = page_img
        context_img = page_img

    canvas_h = 820
    canvas_w = 980
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    page_box_w = canvas_w - 56
    page_box_h = 500
    page_thumb = _resize_to_fit(context_img, page_box_w, page_box_h)
    thumb_pil = Image.fromarray(page_thumb)
    draw = ImageDraw.Draw(thumb_pil)
    draw.rectangle((1, 1, thumb_pil.size[0] - 2, thumb_pil.size[1] - 2), outline="#8A9199", width=2)
    page_thumb = np.asarray(thumb_pil.convert("RGB"))

    thumb_y = 20
    thumb_x = int((canvas_w - page_thumb.shape[1]) // 2)
    canvas[thumb_y : thumb_y + page_thumb.shape[0], thumb_x : thumb_x + page_thumb.shape[1]] = page_thumb
    if 0 <= int(crop_slot) < len(det.bboxes):
        sx = float(page_thumb.shape[1]) / float(max(1, context_img.shape[1]))
        sy = float(page_thumb.shape[0]) / float(max(1, context_img.shape[0]))
        box_center_x = float(thumb_x) + 0.5 * float(rx1 + rx2) * sx
        box_bottom_y = float(thumb_y) + float(ry2) * sy

    crop_box_w = canvas_w - 120
    crop_box_h = 220
    crop_img = _resize_to_fit(crop, crop_box_w, crop_box_h)
    panel_pad_x = 14
    panel_pad_y = 12
    crop_panel_w = crop_img.shape[1] + 2 * panel_pad_x
    crop_panel_h = crop_img.shape[0] + 2 * panel_pad_y
    crop_panel_y = canvas_h - crop_panel_h - 18
    crop_panel_x = int((canvas_w - crop_panel_w) // 2)
    crop_y = crop_panel_y + panel_pad_y
    crop_x = crop_panel_x + panel_pad_x
    panel_x2 = min(canvas_w, crop_panel_x + crop_panel_w)
    panel_y2 = min(canvas_h, crop_panel_y + crop_panel_h)
    canvas[crop_panel_y:panel_y2, crop_panel_x:panel_x2] = 255
    canvas[crop_y : crop_y + crop_img.shape[0], crop_x : crop_x + crop_img.shape[1]] = crop_img
    panel_pil = Image.fromarray(canvas)
    draw_panel = ImageDraw.Draw(panel_pil)
    draw_panel.rectangle(
        (crop_panel_x, crop_panel_y, panel_x2 - 1, panel_y2 - 1),
        outline="#9AA3AD",
        width=2,
    )
    canvas = np.asarray(panel_pil.convert("RGB"))
    if 0 <= int(crop_slot) < len(det.bboxes):
        arrow_coords = (
            box_center_x,
            box_bottom_y + 8.0,
            float(crop_panel_x + crop_panel_w / 2.0),
            float(crop_panel_y + 3.0),
        )
    return canvas, arrow_coords


def _mode_scores_for_brand(
    *,
    q_feats: np.ndarray,
    logo_mat: np.ndarray,
    text_proto: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    txt_per_q = np.asarray(q_feats @ text_proto, dtype=np.float32).reshape(-1)
    img_qk = np.asarray(q_feats @ logo_mat.T, dtype=np.float32)
    img_per_q = np.asarray(img_qk.max(axis=1), dtype=np.float32).reshape(-1)
    fused_per_q = np.asarray(float(alpha) * img_per_q + (1.0 - float(alpha)) * txt_per_q, dtype=np.float32).reshape(-1)

    img_q, img_k = np.unravel_index(int(np.argmax(img_qk)), img_qk.shape)
    txt_q = int(np.argmax(txt_per_q))
    txt_k = int(np.argmax(img_qk[txt_q]))
    fusion_q = int(np.argmax(fused_per_q))
    fusion_k = int(np.argmax(img_qk[fusion_q]))

    return {
        "text": {
            "query_slot_local": txt_q,
            "logo_slot_local": txt_k,
            "text_score": float(txt_per_q[txt_q]),
            "image_score": float(img_per_q[txt_q]),
            "fusion_score": float(fused_per_q[txt_q]),
            "ref_score": float(img_qk[txt_q, txt_k]),
        },
        "image": {
            "query_slot_local": int(img_q),
            "logo_slot_local": int(img_k),
            "text_score": float(txt_per_q[img_q]),
            "image_score": float(img_per_q[img_q]),
            "fusion_score": float(fused_per_q[img_q]),
            "ref_score": float(img_qk[img_q, img_k]),
        },
        "fusion": {
            "query_slot_local": fusion_q,
            "logo_slot_local": fusion_k,
            "text_score": float(txt_per_q[fusion_q]),
            "image_score": float(img_per_q[fusion_q]),
            "fusion_score": float(fused_per_q[fusion_q]),
            "ref_score": float(img_qk[fusion_q, fusion_k]),
        },
    }


def _selection_rank_tuple(
    *,
    case_key: str,
    gt_brand: str,
    n_query_boxes: int,
    fusion_margin: float,
    fusion_score: float,
    text_margin: float,
    image_margin: float,
    text_pred: str,
    image_pred: str,
) -> Tuple[Any, ...]:
    pref_list = [str(x) for x in CASE_DEFS[case_key].get("preferred_gt_brands", [])]
    try:
        pref_rank = pref_list.index(str(gt_brand))
    except ValueError:
        pref_rank = len(pref_list) + 100
    disagreement_bonus = 0 if _norm_brand(text_pred) != _norm_brand(image_pred) else 1
    return (
        int(pref_rank),
        int(disagreement_bonus),
        -float(fusion_margin),
        -float(fusion_score),
        int(n_query_boxes),
        -float(max(text_margin, image_margin)),
    )


def _build_mode_prediction(
    *,
    mode: str,
    sample_index: int,
    pred_idx: int,
    pred_brand: str,
    gt_brand_key: str,
    mode_matrix: np.ndarray,
    query_feat: np.ndarray,
    query_mask: np.ndarray,
    text_proto_all: np.ndarray,
    logo_emb_all: np.ndarray,
    logo_mask_all: np.ndarray,
    logo_paths_by_brand: Dict[str, List[str]],
    alpha_value: float,
) -> ModePrediction:
    mode_scores = np.asarray(mode_matrix[sample_index], dtype=np.float32).reshape(-1)
    sorted_scores = np.sort(mode_scores)
    second_score = float(sorted_scores[-2]) if sorted_scores.size >= 2 else float(mode_scores[pred_idx])
    top_score = float(mode_scores[pred_idx])
    margin = float(top_score - second_score)

    valid_mask = np.asarray(query_mask[sample_index], dtype=np.uint8).astype(bool)
    valid_local_slots = np.where(valid_mask)[0].astype(int).tolist()
    q_feats = np.asarray(query_feat[sample_index][valid_mask], dtype=np.float32)
    if q_feats.ndim != 2 or q_feats.shape[0] <= 0:
        raise RuntimeError(f"Sample {sample_index} has no valid query features")

    brand_logo_mask = np.asarray(logo_mask_all[pred_idx], dtype=bool)
    brand_logo_emb = np.asarray(logo_emb_all[pred_idx][brand_logo_mask], dtype=np.float32)
    if brand_logo_emb.ndim != 2 or brand_logo_emb.shape[0] <= 0:
        raise RuntimeError(f"Predicted brand '{pred_brand}' has no valid logo embeddings")

    brand_logo_paths = list(logo_paths_by_brand.get(pred_brand) or [])
    usable_logo_count = min(len(brand_logo_paths), int(brand_logo_emb.shape[0]))
    if usable_logo_count <= 0:
        raise RuntimeError(
            f"Logo assets unavailable for brand '{pred_brand}': "
            f"paths={len(brand_logo_paths)} valid_emb={int(brand_logo_emb.shape[0])}"
        )
    if usable_logo_count < int(brand_logo_emb.shape[0]):
        brand_logo_emb = np.asarray(brand_logo_emb[:usable_logo_count], dtype=np.float32)
        brand_logo_paths = brand_logo_paths[:usable_logo_count]

    brand_text_proto = np.asarray(text_proto_all[pred_idx], dtype=np.float32).reshape(-1)
    per_mode = _mode_scores_for_brand(
        q_feats=q_feats,
        logo_mat=brand_logo_emb,
        text_proto=brand_text_proto,
        alpha=float(alpha_value),
    )[mode]

    local_q = int(per_mode["query_slot_local"])
    global_q = int(valid_local_slots[local_q])
    logo_local = int(per_mode["logo_slot_local"])
    ref_logo_path = brand_logo_paths[logo_local]

    return ModePrediction(
        mode=mode,
        pred_idx=int(pred_idx),
        pred_brand=str(pred_brand),
        correct=bool(_norm_brand(pred_brand) == str(gt_brand_key)),
        top_score=float(top_score),
        second_score=float(second_score),
        margin=float(margin),
        query_slot=int(global_q),
        image_score_at_slot=float(per_mode["image_score"]),
        text_score_at_slot=float(per_mode["text_score"]),
        fusion_score_at_slot=float(per_mode["fusion_score"]),
        ref_logo_path=str(ref_logo_path),
        ref_logo_score=float(per_mode["ref_score"]),
    )


def _save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _select_cases(
    *,
    detections: Sequence[DetectionSample],
    summary: Dict[str, Any],
    pre_cache: Dict[str, np.ndarray],
    query_cache: Dict[str, np.ndarray],
    repo_payload: Dict[str, Any],
) -> Tuple[List[SelectedCase], Dict[str, List[Dict[str, Any]]]]:
    brand_names = [str(x) for x in list(repo_payload["brand_names"])]
    sim_img = np.asarray(pre_cache["sim_img"], dtype=np.float32)
    sim_txt = np.asarray(pre_cache["sim_txt"], dtype=np.float32)
    y_true = np.asarray(pre_cache["y_true"], dtype=np.uint8).astype(bool)
    has_logo = np.asarray(pre_cache["has_logo"], dtype=np.uint8).astype(bool)
    gt_brand_keys = [str(x) for x in np.asarray(pre_cache["gt_brand_keys"], dtype=object).tolist()]
    resolved_paths = [str(x) for x in np.asarray(pre_cache["resolved_image_paths"], dtype=object).tolist()]
    query_feat = np.asarray(query_cache["query_feat"], dtype=np.float32)
    query_mask = np.asarray(query_cache["query_mask"], dtype=np.uint8)

    text_proto_all = np.asarray(torch.as_tensor(repo_payload["text_proto"]).cpu(), dtype=np.float32)
    logo_emb_all = np.asarray(torch.as_tensor(repo_payload["logo_emb"]).cpu(), dtype=np.float32)
    logo_mask_all = np.asarray(torch.as_tensor(repo_payload["logo_mask"]).cpu(), dtype=bool)
    logo_paths_by_brand = _build_logo_paths_by_brand(summary["config"]["logo_library_dir"], brand_names)

    fusion_method, alpha_vec, fused = _compute_fusion_outputs(summary, sim_img, sim_txt)

    pred_img_idx = np.asarray(np.argmax(sim_img, axis=1), dtype=np.int32)
    pred_txt_idx = np.asarray(np.argmax(sim_txt, axis=1), dtype=np.int32)
    pred_fusion_idx = np.asarray(np.argmax(fused, axis=1), dtype=np.int32)

    pred_img_brand = np.asarray([brand_names[i] for i in pred_img_idx], dtype=object)
    pred_txt_brand = np.asarray([brand_names[i] for i in pred_txt_idx], dtype=object)
    pred_fusion_brand = np.asarray([brand_names[i] for i in pred_fusion_idx], dtype=object)

    gt_nonempty = np.asarray([bool(x) for x in gt_brand_keys], dtype=bool)
    valid_mask = y_true & has_logo & gt_nonempty

    ok_img = np.asarray([_norm_brand(pred_img_brand[i]) == gt_brand_keys[i] for i in range(len(gt_brand_keys))], dtype=bool)
    ok_txt = np.asarray([_norm_brand(pred_txt_brand[i]) == gt_brand_keys[i] for i in range(len(gt_brand_keys))], dtype=bool)
    ok_fusion = np.asarray(
        [_norm_brand(pred_fusion_brand[i]) == gt_brand_keys[i] for i in range(len(gt_brand_keys))],
        dtype=bool,
    )
    n_query_boxes = np.asarray(query_mask.sum(axis=1), dtype=np.int32)

    case_masks = {
        "case1_text_fail_image_rescue": valid_mask & (~ok_txt) & ok_img & ok_fusion,
        "case2_image_fail_text_rescue": valid_mask & (~ok_img) & ok_txt & ok_fusion,
        "case3_both_fail_fusion_success": valid_mask & (~ok_img) & (~ok_txt) & ok_fusion,
    }

    selected: List[SelectedCase] = []
    candidate_tables: Dict[str, List[Dict[str, Any]]] = {}

    for case_key, mask in case_masks.items():
        idxs = np.where(mask)[0].astype(int).tolist()
        ranked_rows: List[Dict[str, Any]] = []
        for i in idxs:
            score_text_top2 = np.sort(sim_txt[i])
            score_img_top2 = np.sort(sim_img[i])
            score_fusion_top2 = np.sort(fused[i])
            margin_text = float(score_text_top2[-1] - score_text_top2[-2])
            margin_image = float(score_img_top2[-1] - score_img_top2[-2])
            margin_fusion = float(score_fusion_top2[-1] - score_fusion_top2[-2])
            rank_tuple = _selection_rank_tuple(
                case_key=case_key,
                gt_brand=str(detections[i].gt_brand or ""),
                n_query_boxes=int(n_query_boxes[i]),
                fusion_margin=float(margin_fusion),
                fusion_score=float(fused[i, pred_fusion_idx[i]]),
                text_margin=float(margin_text),
                image_margin=float(margin_image),
                text_pred=str(pred_txt_brand[i]),
                image_pred=str(pred_img_brand[i]),
            )
            ranked_rows.append(
                {
                    "case_key": case_key,
                    "sample_index": int(i),
                    "sample_key": str(detections[i].sample_key),
                    "gt_brand": str(detections[i].gt_brand or ""),
                    "url": str(detections[i].url or ""),
                    "n_query_boxes": int(n_query_boxes[i]),
                    "pred_text": str(pred_txt_brand[i]),
                    "pred_image": str(pred_img_brand[i]),
                    "pred_fusion": str(pred_fusion_brand[i]),
                    "score_text": float(sim_txt[i, pred_txt_idx[i]]),
                    "score_image": float(sim_img[i, pred_img_idx[i]]),
                    "score_fusion": float(fused[i, pred_fusion_idx[i]]),
                    "margin_text": float(margin_text),
                    "margin_image": float(margin_image),
                    "margin_fusion": float(margin_fusion),
                    "rank_tuple": rank_tuple,
                }
            )

        ranked_rows.sort(key=lambda r: tuple(r["rank_tuple"]))
        for row in ranked_rows:
            row["rank_tuple"] = list(row["rank_tuple"])
        candidate_tables[case_key] = ranked_rows
        if not ranked_rows:
            raise RuntimeError(f"No candidate found for {case_key}")

        override_key = str(MANUAL_OVERRIDE_SAMPLE_KEYS.get(case_key, "") or "").strip()
        if override_key:
            override_row = next((row for row in ranked_rows if str(row["sample_key"]) == override_key), None)
            best_row = override_row if override_row is not None else ranked_rows[0]
        else:
            best_row = ranked_rows[0]
        i = int(best_row["sample_index"])
        gt_brand_key = str(gt_brand_keys[i])
        gt_brand = str(detections[i].gt_brand or "")

        text_mode = _build_mode_prediction(
            mode="text",
            sample_index=i,
            pred_idx=int(pred_txt_idx[i]),
            pred_brand=str(pred_txt_brand[i]),
            gt_brand_key=gt_brand_key,
            mode_matrix=sim_txt,
            query_feat=query_feat,
            query_mask=query_mask,
            text_proto_all=text_proto_all,
            logo_emb_all=logo_emb_all,
            logo_mask_all=logo_mask_all,
            logo_paths_by_brand=logo_paths_by_brand,
            alpha_value=0.5,
        )
        image_mode = _build_mode_prediction(
            mode="image",
            sample_index=i,
            pred_idx=int(pred_img_idx[i]),
            pred_brand=str(pred_img_brand[i]),
            gt_brand_key=gt_brand_key,
            mode_matrix=sim_img,
            query_feat=query_feat,
            query_mask=query_mask,
            text_proto_all=text_proto_all,
            logo_emb_all=logo_emb_all,
            logo_mask_all=logo_mask_all,
            logo_paths_by_brand=logo_paths_by_brand,
            alpha_value=0.5,
        )
        fusion_mode = _build_mode_prediction(
            mode="fusion",
            sample_index=i,
            pred_idx=int(pred_fusion_idx[i]),
            pred_brand=str(pred_fusion_brand[i]),
            gt_brand_key=gt_brand_key,
            mode_matrix=fused,
            query_feat=query_feat,
            query_mask=query_mask,
            text_proto_all=text_proto_all,
            logo_emb_all=logo_emb_all,
            logo_mask_all=logo_mask_all,
            logo_paths_by_brand=logo_paths_by_brand,
            alpha_value=float(alpha_vec[i]),
        )

        case_cfg = CASE_DEFS[case_key]
        selected.append(
            SelectedCase(
                case_key=case_key,
                case_display_name=str(case_cfg["display_name"]),
                case_title=str(case_cfg["row_title"]),
                case_desc_cn=str(case_cfg["description_cn"]),
                sample_index=int(i),
                sample_key=str(detections[i].sample_key),
                url=str(detections[i].url or ""),
                gt_brand=str(gt_brand),
                resolved_image_path=str(resolved_paths[i]),
                n_query_boxes=int(n_query_boxes[i]),
                text_mode=text_mode,
                image_mode=image_mode,
                fusion_mode=fusion_mode,
            )
        )

    return selected, candidate_tables


def _panel_border(ax: Any, color: str) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3.0)
        spine.set_edgecolor(color)


def _draw_query_panel(ax: Any, case: SelectedCase, det: DetectionSample) -> None:
    img = _imread_rgb(case.resolved_image_path)
    ax.imshow(img)
    ax.axis("off")

    for j, box in enumerate(det.bboxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        lw = 1.8
        color = "#BDBDBD"
        if j == int(case.fusion_mode.query_slot):
            color = "#2E8B57"
            lw = 3.2
        rect = patches.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), linewidth=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1 + 2,
            max(8.0, y1 - 4),
            f"#{j + 1}",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="none", alpha=0.95),
        )

    ax.set_title(_wrap_lines(f"GT: {case.gt_brand}", 24), fontsize=10.2, fontweight="bold", pad=6)
    ax.text(0.01, -0.10, _wrap_lines(f"{case.sample_key}\n{case.url}", 44), transform=ax.transAxes, fontsize=8.0, va="top")

    crop_slot = int(case.fusion_mode.query_slot)
    if 0 <= crop_slot < len(det.bboxes):
        x1, y1, x2, y2 = [int(max(0, round(v))) for v in det.bboxes[crop_slot]]
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            inset = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=1.1)
            inset.imshow(crop)
            inset.set_title(f"Crop #{crop_slot + 1}", fontsize=8, pad=4)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.8)
                spine.set_edgecolor("#2E8B57")


def _draw_reference_panel(ax: Any, mode_pred: ModePrediction, mode_label: str, gt_brand: str) -> None:
    img = _imread_rgb(mode_pred.ref_logo_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    status = "Correct" if mode_pred.correct else "Wrong"
    color = "#2E8B57" if mode_pred.correct else "#B23A48"
    _panel_border(ax, color)

    ax.set_title(_wrap_lines(f"{mode_label}\n{mode_pred.pred_brand}", 20), fontsize=10.0, fontweight="bold", pad=6)
    detail = (
        f"{status} | box #{mode_pred.query_slot + 1}\n"
        f"mode={mode_pred.top_score:.3f}  margin={mode_pred.margin:.3f}\n"
        f"img={mode_pred.image_score_at_slot:.3f}  txt={mode_pred.text_score_at_slot:.3f}"
    )
    ax.text(0.5, -0.12, detail, transform=ax.transAxes, ha="center", va="top", fontsize=8.0, color=color)
    gt_tag = "GT" if _norm_brand(mode_pred.pred_brand) == _norm_brand(gt_brand) else "non-GT"
    ax.text(
        0.02,
        0.02,
        gt_tag,
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.20", facecolor=color, edgecolor="none", alpha=0.95),
    )


def _draw_query_focus_panel(ax: Any, case: SelectedCase, det: DetectionSample) -> None:
    page_img = _imread_rgb(case.resolved_image_path)
    crop_slot = int(case.fusion_mode.query_slot)
    canvas, arrow_coords = _build_query_canvas(page_img, det, crop_slot)
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([])
    _panel_border(ax, "#C9CED6")
    ax.set_title(f"GT: {case.gt_brand}", fontsize=11.5, fontweight="bold", pad=8)
    if arrow_coords is not None:
        start_x, start_y, end_x, end_y = arrow_coords
        ann = ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1.35, color="#59636E", shrinkA=0, shrinkB=0),
        )
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])


def _draw_method_card(ax: Any, mode_pred: ModePrediction, mode_label: str, gt_brand: str) -> None:
    img = _imread_rgb(mode_pred.ref_logo_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    correct = bool(mode_pred.correct)
    color = "#248A3D" if correct else "#C23B4B"
    _panel_border(ax, color)
    status = "Correct" if correct else "Wrong"
    pred_label = _brand_display_name(mode_pred.pred_brand)
    ax.text(
        0.5,
        -0.07,
        _wrap_lines(f"{pred_label}\n{status}", 18),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        color=color,
        fontweight="bold",
    )
    if _norm_brand(mode_pred.pred_brand) == _norm_brand(gt_brand):
        tag = "GT"
    else:
        tag = "non-GT"
    ax.text(
        0.02,
        0.02,
        tag,
        transform=ax.transAxes,
        fontsize=7.8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.16", facecolor=color, edgecolor="none", alpha=0.95),
    )
    ax.text(
        0.98,
        0.98,
        mode_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#1F2A44",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#D4D8DD", alpha=0.92),
    )


def _render_topconf_figure(selected_cases: Sequence[SelectedCase], detections: Sequence[DetectionSample], out_path: str) -> None:
    fig, axes = plt.subplots(nrows=len(selected_cases), ncols=4, figsize=(15.8, 10.4), dpi=240)
    if len(selected_cases) == 1:
        axes = np.asarray([axes], dtype=object)

    headers = ["Query crop + page context", "Text-only", "Image-only", "Fusion"]
    for j, header in enumerate(headers):
        axes[0, j].text(
            0.5,
            1.09,
            header,
            transform=axes[0, j].transAxes,
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color="#1F2A44",
        )

    for row, case in enumerate(selected_cases):
        det = detections[int(case.sample_index)]
        _draw_query_focus_panel(axes[row, 0], case, det)
        _draw_method_card(axes[row, 1], case.text_mode, "Text-only", case.gt_brand)
        _draw_method_card(axes[row, 2], case.image_mode, "Image-only", case.gt_brand)
        _draw_method_card(axes[row, 3], case.fusion_mode, "Fusion", case.gt_brand)
        axes[row, 0].text(
            -0.16,
            0.5,
            f"{case.case_display_name}\n{CASE_DEFS[case.case_key]['figure_label_en']}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#1F2A44",
        )

    fig.suptitle("Complementary Benefits of Multi-modal Brand Matching on Real Phishing Pages", fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.02,
        "Green borders indicate the correct brand; red borders indicate an incorrect brand. Each method panel shows the nearest real reference logo returned by that branch.",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color="#36454F",
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.08, wspace=0.14, hspace=0.42)
    _ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _render_twocol_main_figure(
    selected_cases: Sequence[SelectedCase],
    detections: Sequence[DetectionSample],
    out_path: str,
) -> None:
    # Sized for a typical two-column figure* width (~7.1-7.3 inches) so the text is tuned
    # for final print scale rather than for an oversized preview.
    fig, axes = plt.subplots(
        nrows=len(selected_cases),
        ncols=4,
        figsize=(7.25, 5.15),
        dpi=420,
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.18, 1.0, 1.0, 1.0]},
    )
    if len(selected_cases) == 1:
        axes = np.asarray([axes], dtype=object)

    fig.patch.set_facecolor("white")
    for row, case in enumerate(selected_cases):
        det = detections[int(case.sample_index)]
        _draw_query_focus_panel(axes[row, 0], case, det)
        _draw_method_card(axes[row, 1], case.text_mode, "Text", case.gt_brand)
        _draw_method_card(axes[row, 2], case.image_mode, "Image", case.gt_brand)
        _draw_method_card(axes[row, 3], case.fusion_mode, "Fusion", case.gt_brand)

        axes[row, 0].text(
            -0.17,
            0.50,
            f"({chr(ord('a') + row)}) {case.case_display_name}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#1F2A44",
        )
        axes[row, 0].text(
            -0.27,
            0.50,
            CASE_DEFS[case.case_key]["figure_label_en"],
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#36454F",
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.055, wspace=0.14, hspace=0.34)
    _ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _render_main_figure(selected_cases: Sequence[SelectedCase], detections: Sequence[DetectionSample], out_path: str) -> None:
    fig, axes = plt.subplots(nrows=len(selected_cases), ncols=4, figsize=(16.8, 12.6), dpi=220, constrained_layout=False)
    if len(selected_cases) == 1:
        axes = np.asarray([axes], dtype=object)

    fig.patch.set_facecolor("white")

    for row, case in enumerate(selected_cases):
        det = detections[int(case.sample_index)]
        _draw_query_panel(axes[row, 0], case, det)
        _draw_reference_panel(axes[row, 1], case.text_mode, "Text-only", case.gt_brand)
        _draw_reference_panel(axes[row, 2], case.image_mode, "Image-only", case.gt_brand)
        _draw_reference_panel(axes[row, 3], case.fusion_mode, "Fusion", case.gt_brand)
        axes[row, 0].text(
            -0.15,
            0.5,
            f"{case.case_display_name}\n{CASE_DEFS[case.case_key]['figure_label_en']}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#1F2A44",
        )

    fig.suptitle("Real Case Studies of Complementary Multi-modal Brand Matching", fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.012,
        (
            "Reference panels show the nearest real logo image inside the brand predicted by each mode. "
            "The left panel highlights the fusion-selected detection box."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#36454F",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.075, wspace=0.12, hspace=0.62)
    _ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _render_context_figure(selected_cases: Sequence[SelectedCase], detections: Sequence[DetectionSample], out_path: str) -> None:
    fig, axes = plt.subplots(nrows=len(selected_cases), ncols=1, figsize=(13.8, 16.2), dpi=220)
    if len(selected_cases) == 1:
        axes = [axes]

    for row, case in enumerate(selected_cases):
        ax = axes[row]
        det = detections[int(case.sample_index)]
        img = _imread_rgb(case.resolved_image_path)
        ax.imshow(img)
        ax.axis("off")
        for j, box in enumerate(det.bboxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            color = "#2E8B57" if j == int(case.fusion_mode.query_slot) else "#C0C0C0"
            lw = 3.2 if j == int(case.fusion_mode.query_slot) else 1.6
            rect = patches.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), linewidth=lw, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x1 + 2,
                max(8.0, y1 - 4),
                f"#{j + 1}",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=color, edgecolor="none", alpha=0.95),
            )

        title = (
            f"{case.case_display_name} | GT {case.gt_brand} | "
            f"Text: {case.text_mode.pred_brand} ({'correct' if case.text_mode.correct else 'wrong'}) | "
            f"Image: {case.image_mode.pred_brand} ({'correct' if case.image_mode.correct else 'wrong'}) | "
            f"Fusion: {case.fusion_mode.pred_brand} ({'correct' if case.fusion_mode.correct else 'wrong'})"
        )
        ax.set_title(_wrap_lines(title, 92), fontsize=10.3, pad=10, fontweight="bold")
        ax.text(0.01, -0.045, _wrap_lines(f"{case.sample_key} | {case.url}", 122), transform=ax.transAxes, fontsize=8.6, va="top")

    fig.suptitle("Page-level Context for the Selected Case Studies", fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.94, bottom=0.035, hspace=0.28)
    _ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _selected_rows_for_export(selected_cases: Sequence[SelectedCase]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in selected_cases:
        rows.append(
            {
                "case_key": case.case_key,
                "case_display_name": case.case_display_name,
                "case_desc_cn": case.case_desc_cn,
                "sample_index": int(case.sample_index),
                "sample_key": case.sample_key,
                "gt_brand": case.gt_brand,
                "url": case.url,
                "resolved_image_path": case.resolved_image_path,
                "n_query_boxes": int(case.n_query_boxes),
                "pred_text": case.text_mode.pred_brand,
                "pred_image": case.image_mode.pred_brand,
                "pred_fusion": case.fusion_mode.pred_brand,
                "text_correct": int(case.text_mode.correct),
                "image_correct": int(case.image_mode.correct),
                "fusion_correct": int(case.fusion_mode.correct),
                "text_query_slot": int(case.text_mode.query_slot),
                "image_query_slot": int(case.image_mode.query_slot),
                "fusion_query_slot": int(case.fusion_mode.query_slot),
                "text_score": float(case.text_mode.top_score),
                "image_score": float(case.image_mode.top_score),
                "fusion_score": float(case.fusion_mode.top_score),
                "text_margin": float(case.text_mode.margin),
                "image_margin": float(case.image_mode.margin),
                "fusion_margin": float(case.fusion_mode.margin),
                "text_ref_logo_path": case.text_mode.ref_logo_path,
                "image_ref_logo_path": case.image_mode.ref_logo_path,
                "fusion_ref_logo_path": case.fusion_mode.ref_logo_path,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready complementary case study figures from OpenBrandPhish cache.")
    parser.add_argument("--stage2-summary", type=str, default="", help="Path to stage2 summary.json. Empty means latest stage2_* output.")
    parser.add_argument("--outputs-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Explicit output directory. Default: outputs/paper_case_studies_20260328",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stage2_summary = str(args.stage2_summary or "").strip()
    if not stage2_summary:
        stage2_summary = _discover_latest_stage2_summary(str(args.outputs_dir))
    stage2_summary = os.path.abspath(stage2_summary)
    summary = _load_json(stage2_summary)

    out_dir = str(args.output_dir or "").strip()
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "paper_case_studies_20260328")
    out_dir = _ensure_dir(os.path.abspath(out_dir))

    det_csv = os.path.abspath(summary["config"]["det_csv"])
    detections = read_detection_top3_csv(det_csv)
    pre_cache = np.load(os.path.abspath(summary["cache"]["feature_cache_path"]), allow_pickle=True)
    query_cache = np.load(os.path.abspath(summary["cache"]["query_cache_path"]), allow_pickle=True)
    repo_payload = torch.load(os.path.abspath(summary["cache"]["repo_cache_path"]), map_location="cpu")

    if len(detections) != int(np.asarray(pre_cache["y_true"]).shape[0]):
        raise RuntimeError("Detection CSV order/length does not match feature cache")
    if len(detections) != int(np.asarray(query_cache["y_true"]).shape[0]):
        raise RuntimeError("Detection CSV order/length does not match query cache")

    selected_cases, candidate_tables = _select_cases(
        detections=detections,
        summary=summary,
        pre_cache=pre_cache,
        query_cache=query_cache,
        repo_payload=repo_payload,
    )

    twocol_png = os.path.join(out_dir, "paper_case_studies_twocol_main.png")
    topconf_png = os.path.join(out_dir, "paper_case_studies_topconf_main.png")
    detailed_png = os.path.join(out_dir, "paper_case_studies_detailed.png")
    context_png = os.path.join(out_dir, "paper_case_studies_context.png")
    _render_twocol_main_figure(selected_cases, detections, twocol_png)
    _render_topconf_figure(selected_cases, detections, topconf_png)
    _render_main_figure(selected_cases, detections, detailed_png)
    _render_context_figure(selected_cases, detections, context_png)

    selected_rows = _selected_rows_for_export(selected_cases)
    _save_csv(os.path.join(out_dir, "selected_cases.csv"), selected_rows)
    _save_json(os.path.join(out_dir, "selected_cases.json"), selected_rows)

    candidate_top_rows: List[Dict[str, Any]] = []
    for case_key, rows in candidate_tables.items():
        for row in rows[:30]:
            item = dict(row)
            item["rank_tuple"] = json.dumps(item["rank_tuple"], ensure_ascii=False)
            candidate_top_rows.append(item)
    _save_csv(os.path.join(out_dir, "candidate_top30_per_case.csv"), candidate_top_rows)

    summary_out = {
        "stage2_summary": stage2_summary,
        "best_alpha": float(summary["selection"]["best_alpha"]),
        "n_selected_cases": int(len(selected_cases)),
        "selected_case_keys": [str(x.case_key) for x in selected_cases],
        "outputs": {
            "twocol_main_png": twocol_png,
            "twocol_main_pdf": os.path.splitext(twocol_png)[0] + ".pdf",
            "topconf_main_png": topconf_png,
            "topconf_main_pdf": os.path.splitext(topconf_png)[0] + ".pdf",
            "detailed_png": detailed_png,
            "detailed_pdf": os.path.splitext(detailed_png)[0] + ".pdf",
            "context_png": context_png,
            "context_pdf": os.path.splitext(context_png)[0] + ".pdf",
            "selected_cases_csv": os.path.join(out_dir, "selected_cases.csv"),
            "selected_cases_json": os.path.join(out_dir, "selected_cases.json"),
            "candidate_top30_per_case_csv": os.path.join(out_dir, "candidate_top30_per_case.csv"),
        },
    }
    _save_json(os.path.join(out_dir, "run_summary.json"), summary_out)

    print(f"[case-study] stage2_summary={stage2_summary}")
    print(f"[case-study] out_dir={out_dir}")
    for row in selected_rows:
        print(
            "[case-study] "
            f"{row['case_display_name']} sample={row['sample_key']} "
            f"gt={row['gt_brand']} text={row['pred_text']} image={row['pred_image']} fusion={row['pred_fusion']}"
        )
    print(f"[case-study] twocol_figure={twocol_png}")
    print(f"[case-study] topconf_figure={topconf_png}")
    print(f"[case-study] detailed_figure={detailed_png}")
    print(f"[case-study] context_figure={context_png}")


if __name__ == "__main__":
    main()
