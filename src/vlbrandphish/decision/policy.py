from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .domains import registered_domain


@dataclass
class UrlDecision:
    pred_is_phishing: bool
    url_match: bool
    screenshot_domain: str
    reason: str


def decide_with_threshold_and_url(
    score: float,
    threshold: float,
    screenshot_url: str,
    official_domains: Iterable[str],
) -> UrlDecision:
    if float(score) < float(threshold):
        return UrlDecision(
            pred_is_phishing=False,
            url_match=False,
            screenshot_domain=registered_domain(screenshot_url),
            reason="score_below_threshold",
        )

    shot_domain = registered_domain(screenshot_url)
    if not shot_domain:
        return UrlDecision(
            pred_is_phishing=True,
            url_match=False,
            screenshot_domain="",
            reason="empty_or_invalid_url",
        )

    normal_set = {str(d).strip().lower() for d in official_domains if str(d).strip()}
    is_match = shot_domain in normal_set

    if is_match:
        return UrlDecision(
            pred_is_phishing=False,
            url_match=True,
            screenshot_domain=shot_domain,
            reason="official_domain_match",
        )

    return UrlDecision(
        pred_is_phishing=True,
        url_match=False,
        screenshot_domain=shot_domain,
        reason="domain_mismatch",
    )
