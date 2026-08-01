"""Print concise, comparable diagnostics for a fixed-fusion inference run."""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch

from ..components.brand.repository import BrandRepository
from ..data.detection_csv import DetectionSample


def print_runtime_format_info(
    *,
    det_csv: str,
    csv_info: Dict[str, object],
    detections: List[DetectionSample],
    repo: BrandRepository,
    args: argparse.Namespace,
    image_weight: float,
) -> None:
    """Print resolved inputs, repository shape, and fixed decision constants."""
    n_samples = len(detections)
    n_phish = sum(1 for detection in detections if bool(detection.gt_is_phishing))
    n_benign = int(n_samples - n_phish)
    n_has_logo = sum(1 for detection in detections if len(detection.bboxes) > 0)
    n_box1 = sum(1 for detection in detections if len(detection.bboxes) >= 1)
    n_box2 = sum(1 for detection in detections if len(detection.bboxes) >= 2)
    n_box3 = sum(1 for detection in detections if len(detection.bboxes) >= 3)
    n_empty_url = sum(1 for detection in detections if not str(detection.url or "").strip())

    logo_count_per_brand = repo.logo_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().tolist()
    text_count_per_brand = repo.text_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().tolist()
    avg_logo_per_brand = float(sum(int(value) for value in logo_count_per_brand) / max(1, len(logo_count_per_brand)))
    avg_text_per_brand = float(sum(int(value) for value in text_count_per_brand) / max(1, len(text_count_per_brand)))

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
        f"n={n_samples} benign={n_benign} phish={n_phish} "
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
        "[format] fixed_decision: "
        f"fusion=fixed_linear image_weight={float(image_weight):.4f} "
        f"text_weight={float(1.0 - image_weight):.4f} "
        f"threshold={float(args.threshold):.4f}"
    )
    print(
        "[format] runtime: "
        f"max_logo_per_brand={int(args.max_logo_per_brand)} "
        f"image_batch_size={int(args.image_batch_size)} "
        f"text_batch_size={int(args.text_batch_size)} "
        f"device={str(args.device)} progress={int(bool(args.progress))}"
    )
    print("[format] =================================")