from __future__ import annotations

import os
import re
from typing import List
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


def normalize_brand_key(name: str | None) -> str:
    if name is None:
        return ""
    s = str(name).replace("_", " ").strip().lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = " ".join(s.split())
    return s


def read_nonempty_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    return [ln for ln in lines if ln]


def parse_semicolon_list(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).replace(",", ";").split(";")]
    return [p for p in parts if p]


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


def registered_domain(url: str) -> str:
    host = _host_from_url(url)
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


def find_first_matching_file(directory: str, suffix: str) -> str:
    files = sorted(os.listdir(directory))
    for fn in files:
        if fn.lower().endswith(suffix.lower()):
            return os.path.join(directory, fn)
    return ""
