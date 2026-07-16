"""Small dependency-free primitives shared across the package."""

from .brands import normalize_brand_key, parse_semicolon_list, read_nonempty_lines

__all__ = ["normalize_brand_key", "parse_semicolon_list", "read_nonempty_lines"]
