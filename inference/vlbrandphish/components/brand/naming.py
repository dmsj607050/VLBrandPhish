"""Normalize brand labels and parse small text-file conventions used by the brand library."""

from __future__ import annotations

import os
import re
from typing import List


def normalize_brand_key(name: str | None) -> str:
    """Return a comparison key insensitive to case, punctuation, and underscores."""
    if name is None:
        return ""
    s = str(name).replace("_", " ").strip().lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = " ".join(s.split())
    return s


def read_nonempty_lines(path: str) -> List[str]:
    """Read UTF-8 text and discard blank lines while preserving source order."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    return [ln for ln in lines if ln]


def parse_semicolon_list(raw: str) -> List[str]:
    """Parse a delimiter-separated list, accepting both ';' and ',' as separators."""
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).replace(",", ";").split(";")]
    return [p for p in parts if p]


def find_first_matching_file(directory: str, suffix: str) -> str:
    """Return the first file (in sorted order) whose name ends with ``suffix``."""
    files = sorted(os.listdir(directory))
    for fn in files:
        if fn.lower().endswith(suffix.lower()):
            return os.path.join(directory, fn)
    return ""
