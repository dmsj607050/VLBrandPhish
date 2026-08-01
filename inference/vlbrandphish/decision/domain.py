from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from urllib.parse import urlparse


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
    ".ac",
    ".ad",
    ".ae",
    ".af",
    ".ag",
    ".ai",
    ".al",
    ".am",
    ".ao",
    ".ar",
    ".as",
    ".at",
    ".au",
    ".az",
    ".ba",
    ".bb",
    ".bd",
    ".be",
    ".bf",
    ".bg",
    ".bh",
    ".bi",
    ".bj",
    ".bm",
    ".bn",
    ".bo",
    ".br",
    ".bs",
    ".bt",
    ".bw",
    ".by",
    ".bz",
    ".ca",
    ".cc",
    ".cd",
    ".cf",
    ".cg",
    ".ch",
    ".ci",
    ".ck",
    ".cl",
    ".cm",
    ".cn",
    ".co",
    ".cr",
    ".cu",
    ".cv",
    ".cw",
    ".cx",
    ".cy",
    ".cz",
    ".de",
    ".dj",
    ".dk",
    ".dm",
    ".do",
    ".dz",
    ".ec",
    ".ee",
    ".eg",
    ".es",
    ".et",
    ".eu",
    ".fi",
    ".fj",
    ".fm",
    ".fr",
    ".ga",
    ".gd",
    ".ge",
    ".gf",
    ".gg",
    ".gh",
    ".gi",
    ".gl",
    ".gm",
    ".gn",
    ".gp",
    ".gq",
    ".gr",
    ".gt",
    ".gu",
    ".gw",
    ".gy",
    ".hk",
    ".hn",
    ".hr",
    ".ht",
    ".hu",
    ".id",
    ".ie",
    ".il",
    ".im",
    ".in",
    ".io",
    ".iq",
    ".ir",
    ".is",
    ".it",
    ".je",
    ".jm",
    ".jo",
    ".jp",
    ".ke",
    ".kg",
    ".kh",
    ".ki",
    ".km",
    ".kn",
    ".kp",
    ".kr",
    ".kw",
    ".ky",
    ".kz",
    ".la",
    ".lb",
    ".lc",
    ".li",
    ".lk",
    ".lr",
    ".ls",
    ".lt",
    ".lu",
    ".lv",
    ".ly",
    ".ma",
    ".mc",
    ".md",
    ".me",
    ".mg",
    ".mk",
    ".ml",
    ".mm",
    ".mn",
    ".mo",
    ".mq",
    ".mr",
    ".ms",
    ".mt",
    ".mu",
    ".mv",
    ".mw",
    ".mx",
    ".my",
    ".mz",
    ".na",
    ".nc",
    ".ne",
    ".ng",
    ".ni",
    ".nl",
    ".no",
    ".np",
    ".nr",
    ".nz",
    ".om",
    ".pa",
    ".pe",
    ".pf",
    ".pg",
    ".ph",
    ".pk",
    ".pl",
    ".pm",
    ".pn",
    ".pr",
    ".ps",
    ".pt",
    ".pw",
    ".py",
    ".qa",
    ".re",
    ".ro",
    ".rs",
    ".ru",
    ".rw",
    ".sa",
    ".sb",
    ".sc",
    ".sd",
    ".se",
    ".sg",
    ".sh",
    ".si",
    ".sk",
    ".sl",
    ".sm",
    ".sn",
    ".so",
    ".sr",
    ".ss",
    ".st",
    ".sv",
    ".sx",
    ".sy",
    ".sz",
    ".tc",
    ".td",
    ".tf",
    ".tg",
    ".th",
    ".tj",
    ".tk",
    ".tl",
    ".tm",
    ".tn",
    ".to",
    ".tr",
    ".tt",
    ".tv",
    ".tw",
    ".tz",
    ".ua",
    ".ug",
    ".uk",
    ".us",
    ".uy",
    ".uz",
    ".va",
    ".vc",
    ".ve",
    ".vg",
    ".vi",
    ".vn",
    ".vu",
    ".wf",
    ".ws",
    ".ye",
    ".za",
    ".zm",
    ".zw",
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
    """Extract a registrable domain without fetching a public suffix list at runtime."""
    try:
        import tldextract  # type: ignore

        # Use tldextract's bundled suffix snapshot only. This keeps offline
        # artifact evaluation deterministic and avoids an implicit network
        # request or a machine-specific cache.
        extractor = tldextract.TLDExtract(suffix_list_urls=None)
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
    accept: bool
    extracted_domain: str
    domain_part: str
    suffix_part: str
    reason: str


def phishintention_domain_accept(url: str, matched_domain: Iterable[str]) -> DomainMatchResult:
    # Same semantics as PhishIntention _domain_accept:
    # accept=True means domain inconsistency (can be phishing).
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

    if not normalized_matched:
        return DomainMatchResult(
            accept=True,
            extracted_domain=extracted_domain,
            domain_part=domain_part,
            suffix_part=suffix_part,
            reason="empty_matched_domain",
        )

    if extracted_domain and extracted_domain in normalized_matched:
        return DomainMatchResult(
            accept=False,
            extracted_domain=extracted_domain,
            domain_part=domain_part,
            suffix_part=suffix_part,
            reason="exact_domain_match",
        )

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

    if not extracted_domain:
        return DomainMatchResult(
            accept=True,
            extracted_domain=extracted_domain,
            domain_part=domain_part,
            suffix_part=suffix_part,
            reason="invalid_or_empty_url",
        )

    return DomainMatchResult(
        accept=True,
        extracted_domain=extracted_domain,
        domain_part=domain_part,
        suffix_part=suffix_part,
        reason="domain_mismatch",
    )
