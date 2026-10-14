"""Registered-domain extraction and domain-to-brand verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import idna
import tldextract


# ``cache_dir=None`` forces tldextract to use its packaged PSL snapshot. This
# keeps inference deterministic and prevents a first-run network/cache-lock
# side effect.
_EXTRACT = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=(), fallback_to_snapshot=True)
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "і": "i", "ј": "j", "ԁ": "d", "ｍ": "m", "ｏ": "o",
})


@dataclass(frozen=True)
class DomainMatch:
    input_url: str
    hostname: str | None
    registered_domain: str | None
    expected_domains: tuple[str, ...]
    matched: bool
    homograph_skeleton: str | None


class LegalDomainDB:
    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        self._domains = {str(k).lower(): tuple(str(v) for v in values) for k, values in raw.items()}

    def domains_for(self, brand: str) -> tuple[str, ...]:
        key = brand.lower().strip()
        direct = self._domains.get(key)
        if direct is not None:
            return direct
        # Reference-library folder names may include legal suffixes, e.g.
        # ``Alibaba Group`` versus the domain DB key ``alibaba``.
        compact = "".join(character for character in key if character.isalnum())
        for candidate, domains in self._domains.items():
            candidate_compact = "".join(character for character in candidate if character.isalnum())
            if candidate_compact in compact or compact in candidate_compact:
                return domains
        return ()


def normalize_hostname(hostname: str | None) -> str | None:
    if not hostname:
        return None
    hostname = hostname.rstrip(".").strip().lower()
    try:
        return idna.encode(hostname, uts46=True).decode("ascii")
    except idna.IDNAError:
        return hostname.encode("idna").decode("ascii")


def registered_domain(hostname: str | None) -> str | None:
    normalized = normalize_hostname(hostname)
    if not normalized:
        return None
    extracted = _EXTRACT(normalized)
    if not extracted.domain or not extracted.suffix:
        return None
    return f"{extracted.domain}.{extracted.suffix}".lower()


def _parse_hostname(url: str) -> str | None:
    candidate = url.strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = "http://" + candidate
    try:
        return urlsplit(candidate).hostname
    except ValueError:
        return None


def homograph_skeleton(value: str | None) -> str | None:
    if value is None:
        return None
    return value.translate(_CONFUSABLES)


def verify_domain(url: str, brand: str, legal_db: LegalDomainDB) -> DomainMatch:
    hostname = normalize_hostname(_parse_hostname(url))
    actual = registered_domain(hostname)
    expected = tuple(registered_domain(domain) or normalize_hostname(domain) for domain in legal_db.domains_for(brand))
    expected = tuple(d for d in expected if d)
    return DomainMatch(
        input_url=url,
        hostname=hostname,
        registered_domain=actual,
        expected_domains=expected,
        matched=actual is not None and actual in expected,
        homograph_skeleton=homograph_skeleton(actual),
    )
