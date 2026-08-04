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


def brand_converter(brand_name: str) -> str:
    """Map brand names to canonical forms, matching ``url_validation.brand_converter``.

    Bridges the ``logo_brand`` / ``phish_brand`` columns in ``normal_urls.csv`` with
    the ``logo_library`` directory names.  E.g. ``"Apple Inc."`` → ``"Apple"``,
    ``"db"`` → ``"Deutsche Bank AG"``.
    """
    if brand_name in ('Adobe Inc.', 'Adobe Inc', 'Adobe Inc.'): return 'Adobe'
    if brand_name in ('ADP, LLC', 'ADP, LLC.'): return 'ADP'
    if brand_name in ('Amazon.com Inc.', 'Amazon.com Inc'): return 'Amazon'
    if brand_name == 'Americanas.com S,A Comercio Electrnico': return 'Americanas.com S'
    if brand_name in ('AOL Inc.', 'AOL Inc'): return 'AOL'
    if brand_name in ('Apple Inc.', 'Apple Inc', 'Apple Inc.'): return 'Apple'
    if brand_name in ('AT&T Inc.', 'AT&T Inc'): return 'AT&T'
    if brand_name == 'Banco do Brasil S.A.': return 'Banco do Brasil S.A'
    if brand_name == 'Credit Agricole S.A.': return 'Credit Agricole S.A'
    if brand_name == 'DGI (French Tax Authority)': return 'DGI French Tax Authority'
    if brand_name in ('DHL Airways, Inc.', 'DHL Airways, Inc', 'DHL'): return 'DHL Airways'
    if brand_name in ('Dropbox, Inc.', 'Dropbox, Inc'): return 'Dropbox'
    if brand_name in ('eBay Inc.', 'eBay Inc'): return 'eBay'
    if brand_name in ('Facebook, Inc.', 'Facebook, Inc'): return 'Facebook'
    if brand_name == 'Free (ISP)': return 'Free ISP'
    if brand_name in ('Google Inc.', 'Google Inc'): return 'Google'
    if brand_name == 'Mastercard International Incorporated': return 'Mastercard International'
    if brand_name in ('Netflix Inc.', 'Netflix Inc'): return 'Netflix'
    if brand_name in ('PayPal Inc.', 'PayPal Inc'): return 'PayPal'
    if brand_name == 'Royal KPN N.V.': return 'Royal KPN N.V'
    if brand_name == 'SF Express Co.': return 'SF Express Co'
    if brand_name == 'SNS Bank N.V.': return 'SNS Bank N.V'
    if brand_name in ('Square, Inc.', 'Square, Inc'): return 'Square'
    if brand_name == 'Webmail Providers': return 'Webmail Provider'
    if brand_name in ('Yahoo! Inc', 'Yahoo! Inc.'): return 'Yahoo!'
    if brand_name in ('Microsoft OneDrive', 'Office365', 'Outlook'): return 'Microsoft'
    if brand_name == 'Global Sources (HK)': return 'Global Sources HK'
    if brand_name == 'T-Online': return 'Deutsche Telekom'
    if brand_name == 'Airbnb, Inc': return 'Airbnb, Inc.'
    if brand_name == 'azul': return 'Azul'
    if brand_name == 'Raiffeisen Bank S.A': return 'Raiffeisen Bank S.A.'
    if brand_name in ('Twitter, Inc', 'Twitter'): return 'Twitter, Inc.'
    if brand_name == 'capital_one': return 'Capital One Financial Corporation'
    if brand_name == 'la_banque_postale': return 'La Banque postale'
    if brand_name == 'db': return 'Deutsche Bank AG'
    if brand_name in ('Swiss Post', 'PostFinance'): return 'PostFinance'
    if brand_name == 'grupo_bancolombia': return 'Bancolombia'
    if brand_name == 'barclays': return 'Barclays Bank Plc'
    if brand_name == 'gov_uk': return 'Government of the United Kingdom'
    if brand_name == 'Aruba S.p.A': return 'Aruba S.p.A.'
    if brand_name == 'TSB Bank Plc': return 'TSB Bank Limited'
    if brand_name == 'strato': return 'Strato AG'
    if brand_name == 'cogeco': return 'Cogeco'
    if brand_name == 'Canada Revenue Agency': return 'Government of Canada'
    if brand_name == 'UniCredit Bulbank': return 'UniCredit Bank Aktiengesellschaft'
    if brand_name == 'ameli_fr': return 'French Health Insurance'
    if brand_name == 'Banco de Credito del Peru': return 'bcp'
    return brand_name
