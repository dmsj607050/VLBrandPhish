"""Define the stable Top-3 logo-detection CSV contract shared by both inference stages."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DETECTION_TOP3_HEADER: List[str] = [
    "image_path",
    "url",
    "sample_key",
    "gt_is_phishing",
    "gt_brand",
    "brand_split",
    "has_logo",
    "det_score_1",
    "x1_1",
    "y1_1",
    "x2_1",
    "y2_1",
    "det_score_2",
    "x1_2",
    "y1_2",
    "x2_2",
    "y2_2",
    "det_score_3",
    "x1_3",
    "y1_3",
    "x2_3",
    "y2_3",
]


@dataclass
class DetectionSample:
    """One screenshot, its label metadata, and up to three detector boxes."""
    image_path: str
    url: str
    sample_key: str
    gt_is_phishing: bool
    gt_brand: Optional[str]
    bboxes: List[Tuple[float, float, float, float]]
    det_scores: List[float]


def _parse_bool(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes"}


def _parse_box_fields(
    score_raw: str,
    x1_raw: str,
    y1_raw: str,
    x2_raw: str,
    y2_raw: str,
) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
    if not (str(x1_raw).strip() and str(y1_raw).strip() and str(x2_raw).strip() and str(y2_raw).strip()):
        return None, 0.0

    x1 = float(x1_raw)
    y1 = float(y1_raw)
    x2 = float(x2_raw)
    y2 = float(y2_raw)
    score = float(score_raw) if str(score_raw).strip() else 1.0
    return (x1, y1, x2, y2), score


def _from_dict_row(row: Dict[str, str]) -> DetectionSample:
    bboxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    for k in (1, 2, 3):
        box, score = _parse_box_fields(
            row.get(f"det_score_{k}", "") or "",
            row.get(f"x1_{k}", "") or "",
            row.get(f"y1_{k}", "") or "",
            row.get(f"x2_{k}", "") or "",
            row.get(f"y2_{k}", "") or "",
        )
        if box is None:
            continue
        bboxes.append(box)
        scores.append(float(score))

    gt_brand_raw = str(row.get("gt_brand", "") or "").strip()
    return DetectionSample(
        image_path=str(row.get("image_path", "") or ""),
        url=str(row.get("url", "") or ""),
        sample_key=str(row.get("sample_key", "") or ""),
        gt_is_phishing=_parse_bool(row.get("gt_is_phishing", "") or row.get("gt_is_phish", "") or "0"),
        gt_brand=gt_brand_raw if gt_brand_raw else None,
        bboxes=bboxes,
        det_scores=scores,
    )


def _from_list_row(cols: List[str]) -> DetectionSample:
    """Parse a headerless detection row, supporting both current and legacy column layouts."""
    if len(cols) < 20:
        raise ValueError(f"Headerless detection row must have >=20 columns, got {len(cols)}")

    # Current Test-B rows include a brand_split column; legacy rows omit it.
    #   image_path,url,sample_key,gt_is_phishing,gt_brand,brand_split,has_logo,det_score_1,x1_1,...
    # The legacy layout starts the detector fields immediately after gt_brand.
    # Legacy: image_path,url,sample_key,gt_is_phishing,gt_brand,det_score_1,x1_1,...
    # The number of columns distinguishes the legacy and current detector CSV layouts.
    if len(cols) >= 22:
        base = 7
    else:
        base = 5

    bboxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    for k in (0, 1, 2):
        off = base + k * 5
        if off + 4 >= len(cols):
            continue
        box, score = _parse_box_fields(
            cols[off],
            cols[off + 1],
            cols[off + 2],
            cols[off + 3],
            cols[off + 4],
        )
        if box is None:
            continue
        bboxes.append(box)
        scores.append(float(score))

    gt_brand_raw = cols[4].strip() if len(cols) > 4 else ""
    return DetectionSample(
        image_path=str(cols[0] if len(cols) > 0 else ""),
        url=str(cols[1] if len(cols) > 1 else ""),
        sample_key=str(cols[2] if len(cols) > 2 else ""),
        gt_is_phishing=_parse_bool(cols[3] if len(cols) > 3 else "0"),
        gt_brand=gt_brand_raw if gt_brand_raw else None,
        bboxes=bboxes,
        det_scores=scores,
    )


def read_detection_top3_csv(path: str) -> List[DetectionSample]:
    """Read the canonical headered or legacy headerless Top-3 detection CSV."""
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        first_row: List[str] = []
        for row in reader:
            if not row:
                continue
            first_row = row
            break

    if not first_row:
        return []

    has_header = str(first_row[0]).strip().lower() == "image_path"

    out: List[DetectionSample] = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                out.append(_from_dict_row(row))
        else:
            reader = csv.reader(f)
            for cols in reader:
                if not cols:
                    continue
                out.append(_from_list_row(cols))
    return out


def detection_to_csv_row(sample: DetectionSample) -> Dict[str, object]:
    """Serialize one detection sample into the stable top-k CSV row format."""
    row: Dict[str, object] = {
        "image_path": sample.image_path,
        "url": sample.url,
        "sample_key": sample.sample_key,
        "gt_is_phishing": int(bool(sample.gt_is_phishing)),
        "gt_brand": sample.gt_brand or "",
        "brand_split": "",
        "has_logo": int(1 if sample.bboxes else 0),
    }
    for k in (1, 2, 3):
        if len(sample.bboxes) >= k:
            x1, y1, x2, y2 = sample.bboxes[k - 1]
            sc = float(sample.det_scores[k - 1]) if len(sample.det_scores) >= k else 1.0
            row[f"det_score_{k}"] = sc
            row[f"x1_{k}"] = x1
            row[f"y1_{k}"] = y1
            row[f"x2_{k}"] = x2
            row[f"y2_{k}"] = y2
        else:
            row[f"det_score_{k}"] = ""
            row[f"x1_{k}"] = ""
            row[f"y1_{k}"] = ""
            row[f"x2_{k}"] = ""
            row[f"y2_{k}"] = ""
    return row


def write_detection_top3_csv(samples: List[DetectionSample], path: str) -> None:
    """Write the canonical headered Top-3 detection interchange file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTION_TOP3_HEADER)
        writer.writeheader()
        for sample in samples:
            writer.writerow(detection_to_csv_row(sample))
