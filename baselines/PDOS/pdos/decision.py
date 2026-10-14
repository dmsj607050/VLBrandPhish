"""PDOS final decision rule and human-readable explanation."""

from __future__ import annotations

from dataclasses import dataclass

from .domain_verification import DomainMatch


@dataclass(frozen=True)
class Decision:
    label: str
    reason: str


def make_decision(
    brand: str | None,
    logo_is_real: bool,
    domain: DomainMatch,
) -> Decision:
    if brand and logo_is_real and domain.matched:
        return Decision("legitimate", f"Legal {brand} logo detected and the registered domain matches a legal domain.")
    if brand and logo_is_real:
        return Decision("phishing", f"A legitimate {brand} logo was detected, but the registered domain does not match {brand}'s legal domain.")
    if domain.matched:
        return Decision("phishing", "The registered domain matches a legal domain, but the detected logo failed similarity verification.")
    return Decision("phishing", "No verified legal logo and no matching legal domain were found.")
