"""Normalize domains and apply the URL-to-brand consistency rule.

Two policy modes are supported:
  - ``phishintention`` (default): original PhishIntention three-tier domain rule
  - ``official_plus_overrides_no_whitelist``: three-tier rule plus overrides for
    short-link domains and suspicious auth/redirect patterns, matching the
    ``02_e2e_vlbrandphish_dataset`` experiment
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


_MULTI_TLDS = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.nz",
    "com.br",
    "com.cn",
    "com.hk",
    "com.tw",
    "co.jp",
    "co.kr",
    "co.in",
    "com.sg",
    "com.mx",
    "com.ar",
    "com.tr",
    "com.my",
    "com.ph",
}

_FALLBACK_COUNTRY_TLDS = {
    ".af", ".ax", ".al", ".dz", ".as", ".ad", ".ao", ".ai", ".aq", ".ag",
    ".ar", ".am", ".aw", ".ac", ".au", ".at", ".az", ".bs", ".bh", ".bd",
    ".bb", ".eus", ".by", ".be", ".bz", ".bj", ".bm", ".bt", ".bo", ".bq",
    ".an", ".nl", ".ba", ".bw", ".bv", ".br", ".io", ".vg", ".bn", ".bg",
    ".bf", ".mm", ".bi", ".kh", ".cm", ".ca", ".cv", ".cat", ".ky", ".cf",
    ".td", ".cl", ".cn", ".cx", ".cc", ".co", ".km", ".cd", ".cg", ".ck",
    ".cr", ".ci", ".hr", ".cu", ".cw", ".cy", ".cz", ".dk", ".dj", ".dm",
    ".do", ".tl", ".tp", ".ec", ".eg", ".sv", ".gq", ".er", ".ee", ".et",
    ".eu", ".fk", ".fo", ".fm", ".fj", ".fi", ".fr", ".gf", ".pf", ".tf",
    ".ga", ".gal", ".gm", ".ps", ".ge", ".de", ".gh", ".gi", ".gr", ".gl",
    ".gd", ".gp", ".gu", ".gt", ".gg", ".gn", ".gw", ".gy", ".ht", ".hm",
    ".hn", ".hk", ".hu", ".is", ".in", ".id", ".ir", ".iq", ".ie", ".im",
    ".il", ".it", ".jm", ".jp", ".je", ".jo", ".kz", ".ke", ".ki", ".kw",
    ".kg", ".la", ".lv", ".lb", ".ls", ".lr", ".ly", ".li", ".lt", ".lu",
    ".mo", ".mk", ".mg", ".mw", ".my", ".mv", ".ml", ".mt", ".mh", ".mq",
    ".mr", ".mu", ".yt", ".mx", ".md", ".mc", ".mn", ".me", ".ms", ".ma",
    ".mz", ".na", ".nr", ".np", ".nc", ".nz", ".ni", ".ne", ".ng", ".nu",
    ".nf", ".tr", ".kp", ".mp", ".no", ".om", ".pk", ".pw", ".pa", ".pg",
    ".py", ".pe", ".ph", ".pn", ".pl", ".pt", ".pr", ".qa", ".ro", ".ru",
    ".rw", ".re", ".bl", ".gp", ".sh", ".kn", ".lc", ".mf", ".pm", ".vc",
    ".ws", ".sm", ".st", ".sa", ".sn", ".rs", ".sc", ".sl", ".sg", ".sx",
    ".sk", ".si", ".sb", ".so", ".za", ".gs", ".kr", ".ss", ".es", ".lk",
    ".sd", ".sr", ".sj", ".sz", ".se", ".ch", ".sy", ".tw", ".tj", ".tz",
    ".th", ".tg", ".tk", ".to", ".tt", ".tn", ".tm", ".tc", ".tv", ".ug",
    ".ua", ".ae", ".uk", ".us", ".vi", ".uy", ".uz", ".vu", ".va", ".ve",
    ".vn", ".wf", ".eh", ".ye", ".zm",
}

# ---- constants for official_plus_overrides mode (matches url_validation.py) ----

_SUSPICIOUS_AUTH_PATH_SEGMENTS = {
    "account", "auth", "login", "secure", "signin", "update", "verify",
}

_SUSPICIOUS_REDIRECT_QUERY_KEYS = {
    "continue", "dest", "destination", "next", "redirect",
    "return", "target", "url",
}

_KNOWN_SHORTENER_DOMAINS = {
    "bit.ly", "buff.ly", "cutt.ly", "goo.gl", "is.gd", "lnkd.in",
    "ow.ly", "rebrand.ly", "shorturl.at", "surl.li", "t.co", "tinyurl.com",
}


def _host_from_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u
    try:
        host = str(urlparse(u).hostname or "").strip().lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _extract_domain_parts_fallback(url: str) -> Tuple[str, str, str]:
    host = _host_from_url(url)
    if not host:
        return "", "", ""
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", host):
        return "", "", ""

    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return "", "", ""

    last2 = ".".join(parts[-2:])
    if last2 in _MULTI_TLDS and len(parts) >= 3:
        domain_part = parts[-3]
        suffix_part = "." + last2
        return f"{domain_part}{suffix_part}", domain_part, suffix_part

    domain_part = parts[-2]
    suffix_part = "." + parts[-1]
    return f"{domain_part}{suffix_part}", domain_part, suffix_part


def extract_domain_parts(url: str) -> Tuple[str, str, str]:
    """Extract a registrable domain without fetching a public suffix list at runtime.

    Returns ``(registrable_domain, domain_part, suffix_part)`` e.g. ``("example.com",
    "example", ".com")``. Prefers ``tldextract`` with its bundled offline suffix
    snapshot for accuracy; falls back to a heuristic splitter when the dependency is
    unavailable so the pipeline still runs in minimal environments.
    """
    try:
        import tldextract  # type: ignore


        ext = extractor(str(url or ''))
        suffix = str(ext.suffix or "")
        domain = str(ext.domain or "")
        if not domain or not suffix:
            return _extract_domain_parts_fallback(url)
        suffix_part = "." + suffix
        extracted_domain = domain + suffix_part
        return extracted_domain, domain, suffix_part
    except Exception:
        return _extract_domain_parts_fallback(url)


def registered_domain(url: str) -> str:
    """Return the normalized registrable domain, or an empty string."""
    return extract_domain_parts(url)[0]


COUNTRY_TLDS = set(_FALLBACK_COUNTRY_TLDS)


@dataclass
class DomainMatchResult:
    """Result of checking a screenshot URL against the predicted brand allowlist."""
    accept: bool
    extracted_domain: str
    domain_part: str
    suffix_part: str
    reason: str


def phishintention_domain_accept(url: str, matched_domain: Iterable[str]) -> DomainMatchResult:
    """Apply the PhishIntention domain-consistency rule (matches experiment baseline).

    ``accept=True`` means the screenshot domain is *inconsistent* with the matched
    brand's official domains, i.e. the sample remains eligible for a phishing
    decision. The decision proceeds in order:

    1. Empty URL or empty matched-domain list → ``accept=False`` (Benign).
       This mirrors ``check_domain_brand_consistency`` in the experiment script,
       which returns ``True`` (consistent) when the brand is unknown or the URL
       is missing, conservatively avoiding false positives on unmatched brands.
    2. Exact registrable-domain match → ``accept=False`` (Benign).
    3. Domain-part match with a country-code TLD → ``accept=False`` (Benign).
    4. Otherwise → ``accept=True`` (Phish).
    """
    # Mirrors the experiment's check_domain_brand_consistency:
    # empty URL or no official domains → conservatively Benign, won't trigger phishing.
    if not str(url or "").strip():
        return DomainMatchResult(
            accept=False,
            extracted_domain="",
            domain_part="",
            suffix_part="",
            reason="empty_url",
        )

    extracted_domain, domain_part, suffix_part = extract_domain_parts(url)

    normalized_matched: List[str] = []
    matched_domain_parts: List[str] = []

    for item in matched_domain or []:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        md, md_part, _ = extract_domain_parts(raw)
        if not md:
            md = _host_from_url(raw)
        if md and md not in normalized_matched:
            normalized_matched.append(md)
        if md_part and md_part not in matched_domain_parts:
            matched_domain_parts.append(md_part)

    # No valid domains for brand → consistent (Benign), matching experiment's
    # behaviour when brand is absent from domain_map.
    if not normalized_matched:
        return DomainMatchResult(
            accept=False,
            extracted_domain=extracted_domain,
            domain_part=domain_part,
            suffix_part=suffix_part,
            reason="empty_matched_domain",
        )

    # Rule 1: exact registrable-domain match → consistent, not phishing.
    if extracted_domain and extracted_domain in normalized_matched:
        return DomainMatchResult(
            accept=False,
            extracted_domain=extracted_domain,
            domain_part=domain_part,
            suffix_part=suffix_part,
            reason="exact_domain_match",
        )

    # Rule 2: domain-part match with a country-code TLD (e.g. brand.co.uk vs brand.com) → consistent.
    if domain_part and (domain_part in matched_domain_parts):
        try:
            if ("." + suffix_part.split(".")[-1]) in COUNTRY_TLDS:
                return DomainMatchResult(
                    accept=False,
                    extracted_domain=extracted_domain,
                    domain_part=domain_part,
                    suffix_part=suffix_part,
                    reason="regional_domain_match",
                )
        except Exception:
            return DomainMatchResult(
                accept=True,
                extracted_domain=extracted_domain,
                domain_part=domain_part,
                suffix_part=suffix_part,
                reason="domain_parse_error_default_accept",
            )

    # URL cannot be parsed into a registrable domain → inconsistent.
    if not extracted_domain:
        return DomainMatchResult(
            accept=True,
            extracted_domain=extracted_domain,
            domain_part=domain_part,
            suffix_part=suffix_part,
            reason="invalid_or_empty_url",
        )

    # Everything else → domain mismatch.
    return DomainMatchResult(
        accept=True,
        extracted_domain=extracted_domain,
        domain_part=domain_part,
        suffix_part=suffix_part,
        reason="domain_mismatch",
    )


# ============================================================================
# official_plus_overrides_no_whitelist policy
# mirrors 02_e2e_vlbrandphish_dataset/code/url_validation.py
# ============================================================================


def _normalize_host(host_or_url: str) -> str:
    """Normalize a URL or bare hostname to lowercase, strip 'www.' prefix."""
    raw = str(host_or_url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        host = str(urlparse(raw).hostname or "").strip().lower()
    except Exception:
        return ""
    host = host.strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _registered_domain_from_host(host_or_url: str) -> str:
    """Extract registrable domain from a host or URL, handling multi-part TLDs."""
    host = _normalize_host(host_or_url)
    if not host:
        return ""
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", host):
        return ""
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return ""
    last2 = ".".join(parts[-2:])
    if last2 in _MULTI_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    last3 = ".".join(parts[-3:]) if len(parts) >= 3 else ""
    if last3 in _MULTI_TLDS and len(parts) >= 4:
        return ".".join(parts[-4:])
    return last2


def _extract_domain_parts_override(host_or_url: str) -> Tuple[str, str, str]:
    """Return (registrable_domain, domain_part, suffix_part) for the override policy."""
    reg = _registered_domain_from_host(host_or_url)
    if not reg:
        return "", "", ""
    parts = [p for p in reg.split(".") if p]
    if len(parts) < 2:
        return "", "", ""
    last2 = ".".join(parts[-2:])
    if last2 in _MULTI_TLDS and len(parts) >= 3:
        suffix = last2
        domain = parts[-3]
    else:
        suffix = parts[-1]
        domain = parts[-2]
    extracted_domain = f"{domain}.{suffix}" if domain and suffix else ""
    return extracted_domain, domain, suffix


def _looks_like_opaque_url_token(value: str) -> bool:
    """Check whether a path segment looks like an opaque URL token (len>=8, alphanumeric)."""
    token = str(value or "").strip()
    return len(token) >= 8 and token.isalnum()


def _official_host_override_reason(url: str, shot_host: str) -> str:
    """Detect suspicious patterns: known shortener domains or auth+redirect combos."""
    host = _normalize_host(shot_host)
    if not host:
        return ""
    reg_domain = _registered_domain_from_host(host)
    if host in _KNOWN_SHORTENER_DOMAINS or reg_domain in _KNOWN_SHORTENER_DOMAINS:
        return "official_host_shortener"
    raw = str(url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    path_parts = [p.strip().lower() for p in parsed.path.split("/") if p.strip()]
    if len(path_parts) < 2:
        return ""
    if path_parts[0] not in _SUSPICIOUS_AUTH_PATH_SEGMENTS:
        return ""
    query_keys = {str(k).strip().lower() for k in parse_qs(parsed.query).keys() if str(k).strip()}
    if not (query_keys & _SUSPICIOUS_REDIRECT_QUERY_KEYS):
        return ""
    if not _looks_like_opaque_url_token(path_parts[1]):
        return ""
    return "official_host_suspicious_pattern"


def official_plus_overrides_url_decision(
    url: str,
    official_hosts: Optional[List[str]] = None,
) -> DomainMatchResult:
    """``official_plus_overrides_no_whitelist`` policy (matches the experiment).

    Returns a ``DomainMatchResult`` where ``accept=True`` means domain mismatch (Phish).

    Decision order:
      1. brand has no official domain mapping → accept=False (Benign)
      2. shortener / suspicious auth-pattern override → accept=True (Phish)
      3. exact registrable-domain match → accept=False (Benign)
      4. domain-part match under a country-code TLD → accept=False (Benign)
      5. URL unparseable → accept=False (Benign)
      6. everything else → accept=True (Phish)
    """
    hosts = [_normalize_host(x) for x in list(official_hosts or [])]
    hosts = [h for h in hosts if h]
    # When brand has no official host mapping, conservatively treat as Benign.
    if not hosts:
        return DomainMatchResult(
            accept=False, extracted_domain="", domain_part="", suffix_part="",
            reason="no_official_host_mapping_benign",
        )

    shot_host = _normalize_host(url)
    if shot_host:
        override = _official_host_override_reason(url, shot_host=shot_host)
        if override:
            return DomainMatchResult(
                accept=True, extracted_domain="", domain_part="", suffix_part="",
                reason=str(override),
            )

    extracted_domain, domain_part, suffix_part = _extract_domain_parts_override(url)
    # URL cannot be resolved to a registrable domain → Benign.
    if not extracted_domain:
        return DomainMatchResult(
            accept=False, extracted_domain="", domain_part="", suffix_part="",
            reason="url_host_missing_benign",
        )

    registered_set = set()
    matched_domain_parts = set()
    for h in hosts:
        reg, mp, _ = _extract_domain_parts_override(h)
        if reg:
            registered_set.add(reg)
        if mp:
            matched_domain_parts.add(mp)

    # Rule 1: exact registrable-domain match → Benign.
    if extracted_domain in registered_set:
        return DomainMatchResult(
            accept=False, extracted_domain=extracted_domain,
            domain_part=domain_part, suffix_part=suffix_part,
            reason="registered_domain_match",
        )

    # Rule 2: domain-part match under a country-code TLD → Benign.
    if domain_part and (domain_part in matched_domain_parts):
        try:
            if ("." + suffix_part.split(".")[-1]) in COUNTRY_TLDS:
                return DomainMatchResult(
                    accept=False, extracted_domain=extracted_domain,
                    domain_part=domain_part, suffix_part=suffix_part,
                    reason="regional_domain_match",
                )
        except Exception:
            return DomainMatchResult(
                accept=True, extracted_domain=extracted_domain,
                domain_part=domain_part, suffix_part=suffix_part,
                reason="host_mismatch",
            )

    # Everything else → Phish.
    return DomainMatchResult(
        accept=True, extracted_domain=extracted_domain,
        domain_part=domain_part, suffix_part=suffix_part,
        reason="host_mismatch",
    )


def decide_url_policy(
    url: str,
    matched_domain: Iterable[str],
    url_policy_mode: str = "phishintention",
) -> DomainMatchResult:
    """Select the URL validation policy based on ``url_policy_mode``.

    - ``phishintention``: original PhishIntention three-tier domain rule
    - ``official_plus_overrides_no_whitelist``: three-tier + shortener/auth overrides
    """
    mode = str(url_policy_mode or "phishintention").strip().lower()
    if mode == "official_plus_overrides_no_whitelist":
        return official_plus_overrides_url_decision(url, list(matched_domain or []))
    return phishintention_domain_accept(url, matched_domain)
