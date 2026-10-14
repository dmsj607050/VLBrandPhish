"""Dataset-specific URL validation copied from the source experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

import tldextract


_EXTRACT = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=(), fallback_to_snapshot=True)
_COUNTRY_CODES = set(
    "af ax al dz as ad ao ai aq ag ar am aw ac au at az bs bh bd bb eus by be bz bj bm bt bo bq an nl ba bw bv br io vg bn bg bf mm bi kh cm ca cv cat ky cf td cl cn cx cc co km cd cg ck cr ci hr cu cw cy cz dk dj dm do tl tp ec eg sv gq er ee et eu fk fo fm fj fi fr gf pf tf ga gal gm ge de gh gi gr gl gd gp gu gt gg gn gw gy ht hm hn hk hu is in id ir iq ie im it jm jp je jo kz ke ki kw kg la lv lb ls lr ly li lt lu mo mk mg mw my mv ml mt mh mq mr mu yt mx md mc mn me ms ma mz na nr np nc nz ni ne ng nu nf mp no om pk pw pa pg py pe ph pn pl pt pr qa ro ru rw re bl sh kn lc mf pm vc ws sm st sa sn rs sc sl sg sx sk si sb so za gs kr ss es lk sd sr sj sz se ch sy tw tj tz th tg tk to tt tn tm tc tv ug ua ae uk us vi uy uz vu va ve vn wf eh ye zm".split()
)
_COUNTRY_SUFFIXES = {"." + value for value in _COUNTRY_CODES}
_SHORTENERS = {"bit.ly", "buff.ly", "cutt.ly", "goo.gl", "is.gd", "lnkd.in", "ow.ly", "rebrand.ly", "shorturl.at", "surl.li", "t.co", "tinyurl.com"}
_AUTH_SEGMENTS = {"account", "auth", "login", "secure", "signin", "update", "verify"}
_REDIRECT_KEYS = {"continue", "dest", "destination", "next", "redirect", "return", "target", "url"}


@dataclass(frozen=True)
class URLDecision:
    accept: bool
    registered_domain: str
    reason: str


def _host(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        host = str(urlsplit(raw).hostname or "").lower().strip(".")
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _parts(value: str) -> tuple[str, str, str]:
    host = _host(value)
    ext = _EXTRACT(host)
    if not ext.domain or not ext.suffix:
        return "", "", ""
    return f"{ext.domain}.{ext.suffix}".lower(), ext.domain.lower(), "." + ext.suffix.lower()


def _official_override(url: str) -> str:
    host = _host(url)
    registered, _, _ = _parts(host)
    if host in _SHORTENERS or registered in _SHORTENERS:
        return "official_host_shortener"
    raw = str(url or "").strip()
    if not raw or "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    path = [part.lower() for part in parsed.path.split("/") if part]
    if len(path) < 2 or path[0] not in _AUTH_SEGMENTS:
        return ""
    if not ({key.lower() for key in parse_qs(parsed.query)} & _REDIRECT_KEYS):
        return ""
    if len(path[1]) < 8 or not path[1].isalnum():
        return ""
    return "official_host_suspicious_pattern"


def decide_url(url: str, official_hosts: list[str] | tuple[str, ...], mode: str) -> URLDecision:
    """Return ``accept=True`` when URL is inconsistent and should be phishing."""
    mode = str(mode or "phishintention").strip().lower()
    official = [host for host in (_host(item) for item in official_hosts) if host]
    if mode == "official_plus_overrides_no_whitelist":
        if not official:
            return URLDecision(False, "", "no_official_host_mapping_benign")
        override = _official_override(url)
        if override:
            return URLDecision(True, "", override)
        actual, actual_part, actual_suffix = _parts(url)
        if not actual:
            return URLDecision(False, "", "url_host_missing_benign")
        official_domains = {_parts(host)[0] for host in official if _parts(host)[0]}
        official_parts = {_parts(host)[1] for host in official if _parts(host)[1]}
        if actual in official_domains:
            return URLDecision(False, actual, "registered_domain_match")
        if actual_part in official_parts and actual_suffix in _COUNTRY_SUFFIXES:
            return URLDecision(False, actual, "regional_domain_match")
        return URLDecision(True, actual, "host_mismatch")

    # PhishIntention original three-tier rule. Its invalid URL and missing
    # official mapping behavior is deliberately different from a strict deny.
    if not str(url or "").strip() or not official:
        return URLDecision(False, "", "empty_url_or_official_mapping")
    actual, actual_part, actual_suffix = _parts(url)
    official_parts = []
    official_domains = set()
    for host in official:
        domain, part, _ = _parts(host)
        if domain:
            official_domains.add(domain)
        if part:
            official_parts.append(part)
    if actual and actual in official_domains:
        return URLDecision(False, actual, "exact_domain_match")
    if actual_part and actual_part in official_parts and actual_suffix in _COUNTRY_SUFFIXES:
        return URLDecision(False, actual, "regional_domain_match")
    if not actual:
        return URLDecision(True, "", "invalid_or_empty_url")
    return URLDecision(True, actual, "domain_mismatch")
