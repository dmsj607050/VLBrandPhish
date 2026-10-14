"""Shared operating point reported in the paper."""

FIXED_IMAGE_WEIGHT = 0.4
FIXED_TEXT_WEIGHT = 0.6
FIXED_DECISION_THRESHOLD = 0.4
DEFAULT_TOPK = 1


def fixed_decision():
    return {"image_weight": FIXED_IMAGE_WEIGHT, "text_weight": FIXED_TEXT_WEIGHT,
            "threshold": FIXED_DECISION_THRESHOLD}
