from __future__ import annotations

import csv
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
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

_SUSPICIOUS_AUTH_PATH_SEGMENTS = {
    "account",
    "auth",
    "login",
    "secure",
    "signin",
    "update",
    "verify",
}

_SUSPICIOUS_REDIRECT_QUERY_KEYS = {
    "continue",
    "dest",
    "destination",
    "next",
    "redirect",
    "return",
    "target",
    "url",
}

_KNOWN_SHORTENER_DOMAINS = {
    "bit.ly",
    "buff.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "lnkd.in",
    "ow.ly",
    "rebrand.ly",
    "shorturl.at",
    "surl.li",
    "t.co",
    "tinyurl.com",
}


def _as_bool01(value: object) -> int:
    s = str(value or "").strip().lower()
    try:
        x = float(s)
        if not math.isfinite(x):
            return 0
        return 1 if x != 0.0 else 0
    except Exception:
        return 1 if s in {"1", "true", "t", "yes", "y"} else 0


def host_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        host = str(urlparse(raw).hostname or "").strip().lower()
    except Exception:
        return ""
    return host


def _normalize_host(host_or_url: str) -> str:
    raw = str(host_or_url or "").strip()
    if not raw:
        return ""
    h = host_from_url(raw) if ("://" in raw or "/" in raw) else raw.lower()
    h = h.strip().strip(".")
    if h.startswith("www."):
        h = h[4:]
    return h


def registered_domain(host_or_url: str) -> str:
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


def extract_domain_parts(host_or_url: str) -> Tuple[str, str, str]:
    host = _normalize_host(host_or_url)
    if not host:
        return "", "", ""
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", host):
        return "", "", ""

    reg = registered_domain(host)
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
    token = str(value or "").strip()
    return len(token) >= 8 and token.isalnum()


def official_host_override_reason(url: str, *, shot_host: str) -> str:
    host = _normalize_host(shot_host)
    if not host:
        return ""

    reg_domain = registered_domain(host)
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

    path_parts = [part.strip().lower() for part in parsed.path.split("/") if part.strip()]
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


def load_brand_host_whitelist(
    csv_path: str,
    *,
    normalize_brand_key: Optional[Callable[[str], str]] = None,
) -> Set[Tuple[str, str]]:
    p = str(csv_path or "").strip()
    if not p:
        return set()

    out: Set[Tuple[str, str]] = set()
    with open(p, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand_raw = str(
                row.get("brand_norm", "")
                or row.get("brand", "")
                or row.get("brand_1", "")
                or row.get("pred_brand", "")
                or ""
            ).strip()
            host_raw = str(row.get("host", "") or row.get("url_host", "") or "").strip()
            host = _normalize_host(host_raw)
            if not host:
                continue

            if callable(normalize_brand_key):
                brand = str(normalize_brand_key(brand_raw) or "").strip()
            else:
                brand = str(brand_raw or "").strip().lower()
            if not brand:
                continue

            out.add((brand, host))
    return out


def official_url_decision(
    url: str,
    official_hosts: List[str] | Tuple[str, ...] | None,
    *,
    country_tlds: Optional[Iterable[str]] = None,
) -> Tuple[bool, bool, str]:
    hosts = [_normalize_host(x) for x in list(official_hosts or [])]
    hosts = [h for h in hosts if h]
    if not hosts:
        return True, False, "no_official_host_mapping"

    extracted_domain, domain_part, suffix_part = extract_domain_parts(url)
    if not extracted_domain:
        return True, False, "url_host_missing"

    host_set = set(hosts)
    matched_domain_parts = []
    for h in hosts:
        _, matched_domain_part, _ = extract_domain_parts(h)
        if matched_domain_part:
            matched_domain_parts.append(matched_domain_part)

    if extracted_domain in host_set:
        return False, True, "exact_domain_match"

    cctlds = {str(x or "").strip() for x in list(country_tlds or []) if str(x or "").strip()}
    if domain_part and (domain_part in matched_domain_parts):
        try:
            if ("." + suffix_part.split(".")[-1]) in cctlds:
                return False, True, "regional_domain_match"
        except Exception:
            return True, False, "host_mismatch"

    return True, False, "host_mismatch"


def official_plus_overrides_no_whitelist_url_decision(
    url: str,
    official_hosts: List[str] | Tuple[str, ...] | None,
    *,
    country_tlds: Optional[Iterable[str]] = None,
) -> Tuple[bool, bool, str]:
    hosts = [_normalize_host(x) for x in list(official_hosts or [])]
    hosts = [h for h in hosts if h]
    if not hosts:
        return False, False, "no_official_host_mapping_benign"

    shot_host = _normalize_host(url)
    if shot_host:
        override = official_host_override_reason(url, shot_host=shot_host)
        if override:
            return True, True, str(override)

    extracted_domain, domain_part, suffix_part = extract_domain_parts(url)
    if not extracted_domain:
        return False, False, "url_host_missing_benign"

    registered_set = set()
    matched_domain_parts = set()
    for h in hosts:
        reg, matched_domain_part, _ = extract_domain_parts(h)
        if reg:
            registered_set.add(reg)
        if matched_domain_part:
            matched_domain_parts.add(matched_domain_part)

    if extracted_domain in registered_set:
        return False, True, "registered_domain_match"

    cctlds = {str(x or "").strip() for x in list(country_tlds or []) if str(x or "").strip()}
    if domain_part and (domain_part in matched_domain_parts):
        try:
            if ("." + suffix_part.split(".")[-1]) in cctlds:
                return False, True, "regional_domain_match"
        except Exception:
            return True, False, "host_mismatch"

    return True, False, "host_mismatch"


def strict_host_url_decision(
    url: str,
    official_hosts: List[str] | Tuple[str, ...] | None,
    *,
    pred_brand: str = "",
    whitelist_pairs: Optional[Set[Tuple[str, str]]] = None,
    normalize_brand_key: Optional[Callable[[str], str]] = None,
) -> Tuple[bool, bool, str]:
    shot_host = _normalize_host(url)
    hosts = [_normalize_host(x) for x in list(official_hosts or [])]
    hosts = [h for h in hosts if h]

    if not hosts:
        return True, False, "no_official_host_mapping"
    if not shot_host:
        return True, False, "url_host_missing"

    shot_reg = registered_domain(shot_host)
    host_set = set(hosts)
    reg_set = {registered_domain(h) for h in hosts if registered_domain(h)}
    host_match = (shot_host in host_set) or (bool(shot_reg) and shot_reg in reg_set)

    if not host_match:
        brand_key = ""
        if pred_brand:
            if callable(normalize_brand_key):
                brand_key = str(normalize_brand_key(str(pred_brand)) or "").strip()
            else:
                brand_key = str(pred_brand or "").strip().lower()
        if whitelist_pairs and brand_key and shot_host and (brand_key, shot_host) in whitelist_pairs:
            return False, True, "whitelist_host_match"
        return True, False, "host_mismatch"

    override = official_host_override_reason(url, shot_host=shot_host)
    if override:
        return True, True, str(override)
    return False, True, "official_host_match"


def score_from_best_entry(entries: Any) -> float:
    best = 0.0
    if entries is None:
        return 0.0
    for e in list(entries):
        try:
            sim = float((e or {}).get("sim", 0.0))
        except Exception:
            sim = 0.0
        if not math.isfinite(sim):
            sim = 0.0
        accept = _as_bool01((e or {}).get("accept", 0))
        score = sim if accept else 0.0
        if score > best:
            best = score
    return float(best)


def decision_from_best_entry(entries: Any, similarity_threshold: float) -> Dict[str, object]:
    best_score = 0.0
    best_entry: Dict[str, object] = {}
    if entries is None:
        entries = []

    for e in list(entries):
        item = dict(e or {})
        try:
            sim = float(item.get("sim", 0.0))
        except Exception:
            sim = 0.0
        if not math.isfinite(sim):
            sim = 0.0
        accept = _as_bool01(item.get("accept", 0))
        score = sim if accept else 0.0
        if (not best_entry) or (score > best_score):
            best_score = float(score)
            best_entry = item

    brand = str(best_entry.get("brand", "") or "") if best_entry else ""
    pred_is_phish = bool(brand) and float(best_score) >= float(similarity_threshold)
    return {
        "pred_is_phish": bool(pred_is_phish),
        "pred_brand": brand if pred_is_phish else "",
        "top_sim": float(best_score),
        "score": float(best_score),
        "best_entry": best_entry,
    }


def ranked_brands_from_best_entry(entries: Any, normalize_fn=None) -> List[str]:
    if entries is None:
        return []

    best_by_brand: Dict[str, float] = {}
    for e in list(entries):
        item = dict(e or {})
        brand_raw = str(item.get("brand", "") or "").strip()
        if not brand_raw:
            continue
        brand = str(normalize_fn(brand_raw)) if callable(normalize_fn) else brand_raw
        if not brand:
            continue

        try:
            sim = float(item.get("sim", 0.0))
        except Exception:
            sim = 0.0
        if not math.isfinite(sim):
            sim = 0.0
        accept = _as_bool01(item.get("accept", 0))
        score = sim if accept else 0.0

        prev = best_by_brand.get(brand)
        if prev is None or score > prev:
            best_by_brand[brand] = float(score)

    ranked = sorted(best_by_brand.items(), key=lambda kv: kv[1], reverse=True)
    return [str(k) for k, _ in ranked]
