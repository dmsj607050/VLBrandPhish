"""Named values passed between inference preparation, scoring, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Metrics:
    """Confusion-matrix counts and brand accuracy for one evaluated split."""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    brand_total_phish: int = 0
    brand_correct_phish: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Return confusion-matrix counts and derived metrics as a mapping."""
        tp = float(self.tp)
        tn = float(self.tn)
        fp = float(self.fp)
        fn = float(self.fn)
        accuracy = (tp + tn) / max(1.0, tp + tn + fp + fn)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
        brand_accuracy = float(self.brand_correct_phish) / max(1.0, float(self.brand_total_phish))
        return {
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "brand_acc_on_phish": float(brand_accuracy),
            "brand_total_phish": int(self.brand_total_phish),
            "brand_correct_phish": int(self.brand_correct_phish),
        }


@dataclass
class PrecomputedData:
    """Brand similarities and labels after crop features have been scored."""
    sim_img: np.ndarray
    sim_txt: np.ndarray
    router_alpha: np.ndarray
    has_logo: np.ndarray
    y_true: np.ndarray
    gt_brand_keys: List[str]
    screenshot_domains: List[str]
    resolved_image_paths: List[str]
    resolve_stats: Dict[str, int]


@dataclass
class QueryFeatureData:
    """Encoded detector crops and path-resolution diagnostics before brand scoring."""
    query_feat: np.ndarray
    route_weights: np.ndarray
    query_mask: np.ndarray
    y_true: np.ndarray
    gt_brand_keys: List[str]
    screenshot_domains: List[str]
    resolved_image_paths: List[str]
    resolve_stats: Dict[str, int]


@dataclass
class AlphaOutput:
    """Fused brand predictions plus the domain-gate evidence for every sample."""
    fusion_method: str
    alpha: float
    gate_k: float
    gate_b: float
    alpha_vec: np.ndarray
    pred_brand_idx: np.ndarray
    pred_brand_names: List[str]
    best_score: np.ndarray
    best_score_img: np.ndarray
    best_score_txt: np.ndarray
    url_would_be_phish: np.ndarray
    url_match: np.ndarray
    brand_match: np.ndarray
    url_reason_above_thr: List[str]
