"""Build and score the in-memory brand evidence repository used during inference."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from urllib.parse import urlparse

import torch
from PIL import Image, UnidentifiedImageError

from .naming import brand_converter, normalize_brand_key, parse_semicolon_list, read_nonempty_lines
from .encoders import CLIPEncoder


@dataclass
class BrandRepository:
    """Pre-encoded logos, prompts, and official domains indexed by brand."""
    brand_names: List[str]
    logo_emb: torch.Tensor      # [B, K, D]
    logo_mask: torch.Tensor     # [B, K]
    text_emb: torch.Tensor      # [B, T, D]
    text_mask: torch.Tensor     # [B, T]
    text_proto: torch.Tensor    # [B, D], mean text prototype per brand
    brand_domains: Dict[str, List[str]]


TRAIN_TEMPLATE_PROMPTS = [
    "{brand}",
    "{brand} logo",
    "official logo of {brand}",
    "{brand} brand logo",
    "{brand} wordmark",
    "{brand} logomark",
    "a clean logo of {brand}",
    "cropped {brand} logo",
    "small {brand} logo",
    "{brand} logo on a website header",
    "{brand} logo in a webpage screenshot",
    "{brand} logo in navigation bar",
]


# Optional cross-process closure resolver injected by ``load_brand_overlap_match277``.
# Set to None before first use so the attribute check in ``build_brand_repository``
# works without needing to force-import overlap CSV in every mode.
_OVERLAP_ALIAS_RESOLVER = None


# Text prototype mode that avoids re-encoding text prompts (which may require
# a sentencepiece tokenizer that crashes under some Windows environments).
CKPT_TEXT_PROTO_MODES = {
    "ckpt_text_proto",
    "ckpt_text_proto_mean",
    "ckpt_proto",
    "ckpt_proto_mean",
}


def _build_ckpt_brand_name_to_proto_index(encoder: CLIPEncoder) -> Dict[str, int]:
    """Return a normalized-brand-name → ckpt label index map for every known label.

    Uses exactly the same matching heuristics used for ``dbi5_dir`` / ``normal_urls``
    resolution: exact → normalize_brand_key → normalize_brand_key(brand_converter(...))
    → _brand_alias_key (unique match only).
    """
    out: Dict[str, int] = {}
    if not getattr(encoder, "_ckpt_label_to_name", None):
        return out
    l2n_items: List[Tuple[int, str]] = sorted(encoder._ckpt_label_to_name.items(), key=lambda kv: int(kv[0]))
    names = [nm for _, nm in l2n_items]
    idx_by_name = {nm: i for i, (_, nm) in enumerate(l2n_items)}

    norm_lookup = _unique_name_lookup(names, normalize_brand_key)
    alias_lookup = _unique_name_lookup(names, _brand_alias_key)
    bc_norm_lookup = _unique_name_lookup(names, lambda s: normalize_brand_key(brand_converter(s)))

    for label_idx, raw_name in l2n_items:
        primary_norm = normalize_brand_key(raw_name)
        if primary_norm and primary_norm not in out:
            out[primary_norm] = int(label_idx)
        bc_norm = normalize_brand_key(brand_converter(raw_name))
        if bc_norm and bc_norm not in out:
            out[bc_norm] = int(label_idx)
        # Also accept: raw_name string (exact, w/o spaces issues), bc() name str exact
        if raw_name and raw_name not in out:
            out[raw_name] = int(label_idx)
        bc_name = brand_converter(raw_name)
        if bc_name and bc_name not in out:
            out[bc_name] = int(label_idx)
    return out


def resolve_ckpt_proto_label_idx_for_brand(
    brand: str,
    encoder: CLIPEncoder,
    *,
    explicit_match_277_dirs: List[str] | None = None,
) -> int:
    """Resolve a logo-library brand name to a ckpt label index (>=0) or -1."""
    if not getattr(encoder, "_ckpt_label_to_name", None):
        return -1
    l2n_items: List[Tuple[int, str]] = sorted(encoder._ckpt_label_to_name.items(), key=lambda kv: int(kv[0]))
    label_name_to_idx = {str(nm): int(i) for i, nm in l2n_items}
    # Build reverse mapping: normalized/alias forms of ckpt label names → list of label_idx
    norm_form_to_idx: Dict[str, List[int]] = {}
    alias_form_to_idx: Dict[str, List[int]] = {}
    for idx, raw_name in l2n_items:
        nk = normalize_brand_key(raw_name)
        if nk: norm_form_to_idx.setdefault(nk, []).append(int(idx))
        ak = _brand_alias_key(raw_name)
        if ak: alias_form_to_idx.setdefault(ak, []).append(int(idx))
        bcn = brand_converter(raw_name)
        nk2 = normalize_brand_key(bcn)
        if nk2 and nk2 != nk: norm_form_to_idx.setdefault(nk2, []).append(int(idx))
        ak2 = _brand_alias_key(bcn)
        if ak2 and ak2 != ak: alias_form_to_idx.setdefault(ak2, []).append(int(idx))

    def _unique(d: Dict[str, List[int]], k: str) -> int:
        if not k: return -1
        lst = d.get(k)
        if lst and len(lst) == 1:
            return int(lst[0])
        return -1

    # 0) Explicit 277-dir match (from brands_overlap.csv match_277_dirs column).
    if explicit_match_277_dirs:
        for cand in explicit_match_277_dirs:
            cand = str(cand or "").strip()
            if not cand:
                continue
            if cand in label_name_to_idx:
                return int(label_name_to_idx[cand])
            # match_277_dirs may also be a snake_case short alias. Try normalization + alias.
            for k in [normalize_brand_key(cand), _brand_alias_key(cand)]:
                res = _unique(norm_form_to_idx, k)
                if res >= 0: return res
                res = _unique(alias_form_to_idx, k)
                if res >= 0: return res

    # 0b) Hard-coded crosswalks for well-known "mega" brand conglomerates that
    # share multiple phishing-visible sub-brands with distinct ckpt 277 labels.
    # These are used only as a last-resort when both the overlap-CSV hints and
    # all normalization/alias matching routes fail.
    BRAND_HARD_CROSSWALK: Dict[str, List[str]] = {
        # Overlap CSV canonical "Microsoft" has empty match_277_dirs but the
        # canonical in_277=1 sub-products are Outlook / Office365 / OneDrive.
        "Microsoft": ["Outlook", "Office365"],
        "Microsoft Corporation": ["Outlook", "Office365"],
    }
    norm_brand = normalize_brand_key(brand)
    alias_brand = _brand_alias_key(brand)
    for bform in (brand, brand_converter(brand), norm_brand, alias_brand):
        if not bform:
            continue
        if bform in BRAND_HARD_CROSSWALK:
            for cand in BRAND_HARD_CROSSWALK[bform]:
                if cand in label_name_to_idx:
                    return int(label_name_to_idx[cand])
                for k in [normalize_brand_key(cand), _brand_alias_key(cand)]:
                    res = _unique(norm_form_to_idx, k)
                    if res >= 0: return res
                    res = _unique(alias_form_to_idx, k)
                    if res >= 0: return res
    # 1) exact name
    if brand in label_name_to_idx:
        return int(label_name_to_idx[brand])
    if brand_converter(brand) in label_name_to_idx:
        return int(label_name_to_idx[brand_converter(brand)])

    names = [nm for _, nm in l2n_items]
    norm_lookup = _unique_name_lookup(names, normalize_brand_key)
    alias_lookup = _unique_name_lookup(names, _brand_alias_key)

    # 2) normalize_brand_key (unique match only — consistent with dbi5 resolver)
    for k in [normalize_brand_key(brand), normalize_brand_key(brand_converter(brand))]:
        if k and k in norm_lookup:
            matched_name = norm_lookup[k]
            if matched_name in label_name_to_idx:
                return int(label_name_to_idx[matched_name])
        if k:
            res = _unique(norm_form_to_idx, k)
            if res >= 0: return res

    # 3) _brand_alias_key (strip legal suffixes). Unique match only.
    for k in [_brand_alias_key(brand), _brand_alias_key(brand_converter(brand))]:
        if k and k in alias_lookup:
            matched_name = alias_lookup[k]
            if matched_name in label_name_to_idx:
                return int(label_name_to_idx[matched_name])
        if k:
            res = _unique(alias_form_to_idx, k)
            if res >= 0: return res

    # 4) Fallback: same approach used for dbi5_dir's _resolve_optional_brand_dir.
    names_in_dbi_semantics = list(names)
    normalized_lookup = _unique_name_lookup(names_in_dbi_semantics, normalize_brand_key)
    alias_lookup_2 = _unique_name_lookup(names_in_dbi_semantics, _brand_alias_key)
    resolved_candidates = [
        normalized_lookup.get(normalize_brand_key(brand), ""),
        alias_lookup_2.get(_brand_alias_key(brand), ""),
    ]
    for name in resolved_candidates:
        if name and name in label_name_to_idx:
            return int(label_name_to_idx[name])
    return -1


def list_brand_dirs(root: str) -> List[str]:
    """Return visible brand subdirectories in deterministic order."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Brand directory root not found: {root}")
    return [
        name for name in sorted(os.listdir(root))
        if (not name.startswith(".")) and os.path.isdir(os.path.join(root, name))
    ]


_LEGAL_SUFFIX_WORDS = {
    "ag",
    "as",
    "co",
    "com",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "international",
    "limited",
    "llc",
    "ltd",
    "nv",
    "plc",
    "sa",
}


def _brand_alias_key(name: str) -> str:
    key = normalize_brand_key(name)
    words = [w for w in key.split() if w not in _LEGAL_SUFFIX_WORDS]
    return " ".join(words) or key


def _unique_name_lookup(names: List[str], key_fn) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = {}
    for name in names:
        key = str(key_fn(name) or "").strip()
        if not key:
            continue
        grouped.setdefault(key, []).append(name)
    return {key: vals[0] for key, vals in grouped.items() if len(vals) == 1}


def _resolve_optional_brand_dir(
    root: str,
    brand: str,
    names: List[str],
    hint_names: Sequence[str] | None = None,
) -> str:
    # Try explicit hints first (e.g. match_277_dirs from brands_overlap.csv).
    # These provide authoritative crosswalks when the logo-library brand dir
    # name differs from the DBI-5 directory name (e.g. "Bancolombia" -> "grupo_bancolombia").
    if hint_names:
        for hint in hint_names:
            h = str(hint or "").strip()
            if not h:
                continue
            path = os.path.join(root, h)
            if os.path.isdir(path):
                return path
    exact = os.path.join(root, brand)
    if os.path.isdir(exact):
        return exact

    normalized_lookup = _unique_name_lookup(names, normalize_brand_key)
    alias_lookup = _unique_name_lookup(names, _brand_alias_key)
    candidates = [
        normalized_lookup.get(normalize_brand_key(brand), ""),
        alias_lookup.get(_brand_alias_key(brand), ""),
    ]
    for name in candidates:
        if name:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
    return ""


def list_logo_paths(brand_dir: str) -> List[str]:
    """Return reference images, preferring a brand-specific ``logos`` directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    logo_dir = os.path.join(brand_dir, "logos")
    scan_dir = logo_dir if os.path.isdir(logo_dir) else brand_dir

    out: List[str] = []
    for fn in sorted(os.listdir(scan_dir)):
        ext = os.path.splitext(fn)[1].lower()
        if ext in exts:
            out.append(os.path.join(scan_dir, fn))
    return out


def read_dbi_texts(brand_dbi_dir: str) -> List[str]:
    """Read the optional DBI-5 textual evidence for one reference brand."""
    candidates = [
        os.path.join(brand_dbi_dir, fn)
        for fn in sorted(os.listdir(brand_dbi_dir))
        if fn.lower().endswith("_logo_description.txt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_logo_description.txt found in: {brand_dbi_dir}")
    return read_nonempty_lines(candidates[0])


def _build_train_template_texts(brand_name: str) -> List[str]:
    brand = str(brand_name or "").strip()
    return [tpl.format(brand=brand) for tpl in TRAIN_TEMPLATE_PROMPTS if str(tpl).strip()]


def _read_custom_prompt_templates(custom_prompt_file: str) -> List[str]:
    path = os.path.abspath(str(custom_prompt_file or ""))
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Custom prompt file not found: {custom_prompt_file}")
    lines = read_nonempty_lines(path)
    if not lines:
        raise ValueError(f"Custom prompt file is empty: {custom_prompt_file}")
    return lines


def _build_custom_prompt_texts(brand_name: str, templates: Sequence[str]) -> List[str]:
    brand = str(brand_name or "").strip()
    out: List[str] = []
    for tpl in templates:
        text = str(tpl or "").strip()
        if not text:
            continue
        if "{brand}" in text:
            out.append(text.format(brand=brand))
        elif "{BRAND}" in text:
            out.append(text.replace("{BRAND}", brand))
        else:
            out.append(f"{text} {brand}".strip())
    return out


def load_brand_domains(normal_urls_csv: str) -> Dict[str, List[str]]:
    """Build brand→official-domains map, matching ``build_official_hosts_map`` + ``lookup_hosts``.

    Key alignment points with the experiment's ``url_validation.build_official_hosts_map``:
      1. reads both ``logo_brand`` and ``phish_brand`` columns and maps both;
      2. generates multiple lookup keys per brand (original, brand_converter,
         normalize_brand_key, and combinations) so logo_library directory names
         can match any brand-name variant in the CSV;
      3. stores raw URL strings (no host extraction); normalization is left to
         the URL policy functions.
    """
    out: Dict[str, List[str]] = {}

    def _add_domains(key: str, urls: List[str]) -> None:
        if not key or not urls:
            return
        if key not in out:
            out[key] = []
        exists = set(out[key])
        for u in urls:
            if u not in exists:
                out[key].append(u)
                exists.add(u)

    def _brand_keys(brand: str) -> List[str]:
        """Generate all possible lookup keys for a brand name."""
        keys = [brand]
        conv = brand_converter(brand)
        if conv != brand:
            keys.append(conv)
        nk = normalize_brand_key(brand)
        if nk and nk != brand:
            keys.append(nk)
        nk_conv = normalize_brand_key(conv)
        if nk_conv and nk_conv not in keys:
            keys.append(nk_conv)
        return keys

    with open(normal_urls_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logo_brand = str(row.get("logo_brand", "") or "").strip()
            phish_brand = str(row.get("phish_brand", "") or "").strip()
            url = str(row.get("url", "") or row.get("domain", "") or "").strip()
            if not logo_brand or not url:
                continue
            urls = [u.strip() for u in parse_semicolon_list(url) if u.strip()]
            if not urls:
                continue

            # Map logo_brand via original name + brand_converter + normalize_brand_key
            for key in _brand_keys(logo_brand):
                _add_domains(key, urls)

            # Also map phish_brand (if different from logo_brand)
            if phish_brand and phish_brand != logo_brand:
                for key in _brand_keys(phish_brand):
                    _add_domains(key, urls)

    if not out:
        raise ValueError(f"No valid brand domains loaded from: {normal_urls_csv}")
    return out


def load_brand_overlap_match277(brands_overlap_csv: str) -> Dict[str, List[str]]:
    """Load brand_dir_7299 → list of match_277_dirs candidates from brands_overlap.csv.

    Used to give ``build_brand_repository`` explicit brand-name mapping hints for
    ckpt_text_proto_mean mode, because logo-library brand dirs don't always match
    the 277 ckpt label names exactly.
    """
    out: Dict[str, List[str]] = {}
    if not brands_overlap_csv or not os.path.isfile(brands_overlap_csv):
        return out
    with open(brands_overlap_csv, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    # Build an alias index: normalized(key) → primary key (brand_dir_7299 / canonical_name_7299)
    # that lives inside overlap.csv.  The ckpt_text_proto_* resolver will use this
    # to find explicit 277-dir hints for any logo-library brand name, even when the
    # logo-library brand dir uses a different form than the CSV's keys.
    #
    # Strategy (order matters - first exact unique match wins):
    #   1. brand_dir_7299 exact
    #   2. canonical_name_7299 exact
    #   3. normalize_brand_key(brand_dir_7299) / canonical_name_7299 (unique)
    #   4. _brand_alias_key of above (unique, strip legal suffixes)
    #   5. normalize_brand_key(brand_converter(...)) variants (unique)
    def _unique(d: Dict[str, List[str]], k: str):
        vals = d.get(k)
        return vals[0] if vals and len(vals) == 1 else ""

    # First pass: build normalized forms → primary_keys (multiple)
    norm_to_prim: Dict[str, List[str]] = {}
    alias_to_prim: Dict[str, List[str]] = {}
    bc_norm_to_prim: Dict[str, List[str]] = {}
    primaries = []  # list of dicts {key, cand277}
    for row in rows:
        brand_dir = (row.get("brand_dir_7299") or row.get("brand_dir") or "").strip()
        cname = (row.get("canonical_name_7299") or "").strip()
        m277_raw = (row.get("match_277_dirs") or "").strip()
        cands: List[str] = []
        for c in m277_raw.replace('"', "").split(","):
            s = c.strip()
            if s:
                cands.append(s)
        if cname and cname not in cands:
            cands.append(cname)
        if not cands:
            continue
        primary_key = brand_dir or cname
        if not primary_key:
            continue
        primaries.append({"key": primary_key, "cand277": list(cands)})
        # register lookup forms
        def register(src: str) -> None:
            if not src:
                return
            nk = normalize_brand_key(src)
            if nk: norm_to_prim.setdefault(nk, []).append(primary_key)
            ak = _brand_alias_key(src)
            if ak: alias_to_prim.setdefault(ak, []).append(primary_key)
            bcn = brand_converter(src)
            bnk = normalize_brand_key(bcn)
            if bnk and bnk != nk:
                bc_norm_to_prim.setdefault(bnk, []).append(primary_key)
            bak = _brand_alias_key(bcn)
            if bak and bak != ak:
                alias_to_prim.setdefault(bak, []).append(primary_key)
        register(brand_dir)
        register(cname)

    primary_to_cands: Dict[str, List[str]] = {p["key"]: p["cand277"] for p in primaries}

    # Now, for any raw logo-library brand name, expose:
    #   exact in primaries (key) → cands
    #   unique match via norm / alias / bc_norm → key → cands
    def _resolve(src: str) -> List[str]:
        if not src:
            return []
        exact = primary_to_cands.get(src)
        if exact:
            return list(exact)
        nk = normalize_brand_key(src)
        bcn = brand_converter(src)
        bnk = normalize_brand_key(bcn)
        ak = _brand_alias_key(src)
        bak = _brand_alias_key(bcn)
        for k, idx in (
            (nk, norm_to_prim),
            (bnk, bc_norm_to_prim),
            (nk, alias_to_prim),  # alias may match norm (accept)
            (ak, alias_to_prim),
            (bak, alias_to_prim),
        ):
            if not k:
                continue
            prim_candidates = idx.get(k)
            if prim_candidates and len(prim_candidates) == 1:
                out_cands = primary_to_cands.get(prim_candidates[0])
                if out_cands:
                    return list(out_cands)
        return []

    # Patch _resolve into a module-level closure so resolve_ckpt_proto_label_idx_for_brand
    # can benefit, and also pre-populate `out` with resolved fallback forms (so plain
    # dict.get at the call site yields hints for most logo-library names).
    global _OVERLAP_ALIAS_RESOLVER
    _OVERLAP_ALIAS_RESOLVER = _resolve

    # Seed `out` dict with alias forms that resolve uniquely to 277 candidates,
    # letting consumers (build_brand_repository) look up via overlap_hints.get(brand).
    # Build candidate forms from all primary key variants we can cheaply enumerate.
    out_before_extra = set(out.keys())
    forms_to_try: List[str] = []
    unique_prim_keys = list({p["key"] for p in primaries})
    for key in unique_prim_keys:
        conv = brand_converter(key)
        for s in (key, conv):
            if not s: continue
            forms_to_try.append(s)
            nk = normalize_brand_key(s)
            if nk and nk != s: forms_to_try.append(nk)
            ak = _brand_alias_key(s)
            if ak and ak != s: forms_to_try.append(ak)
    # For each form, resolve candidates via _resolve; only register if the mapping
    # is uniquely consistent (all lookups resolve to the same candidate list).
    for f in forms_to_try:
        if f in out_before_extra:
            continue
        cands = _resolve(f)
        if cands:
            out[f] = list(cands)
    return out


def build_brand_repository(
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    encoder: CLIPEncoder,
    max_logo_per_brand: int = -1,
    image_batch_size: int = 64,
    text_batch_size: int = 256,
    brand_text_mode: str = "ckpt_text_proto_mean",
    custom_prompt_file: str = "",
    brands_overlap_csv: str = "",
    *,
    restrict_brands_to: Sequence[str] | None = None,
) -> BrandRepository:
    """Encode reference logos and prompts into an in-memory brand repository.

    When ``restrict_brands_to`` is provided, the repository is built only for the
    listed brand names (which must still exist as subdirectories inside
    ``logo_library_dir``). This is intended for cross-time / cross-brand ablations
    where the model only needs to cover a small subset of the full library
    (e.g. the 48 overlap brands with in_277=1), while still matching the official
    logo-library directory names exactly.
    """
    logo_library_dir = os.path.abspath(logo_library_dir)
    dbi5_dir = os.path.abspath(dbi5_dir)
    normal_urls_csv = os.path.abspath(normal_urls_csv)

    all_brand_names = list_brand_dirs(logo_library_dir)
    if restrict_brands_to:
        keep_set: set = {str(x).strip() for x in restrict_brands_to if str(x).strip()}
        missing_in_lib = sorted(keep_set - set(all_brand_names))
        if missing_in_lib:
            preview = ", ".join(missing_in_lib[:10])
            more = "" if len(missing_in_lib) <= 10 else f" ... +{len(missing_in_lib) - 10}"
            raise ValueError(
                f"restrict_brands_to references brands absent from logo_library_dir: "
                f"{len(missing_in_lib)}: {preview}{more}"
            )
        brand_names = [b for b in all_brand_names if b in keep_set]
    else:
        brand_names = all_brand_names
    text_mode = str(brand_text_mode or "ckpt_text_proto_mean").strip().lower()
    dbi_brand_names = list_brand_dirs(dbi5_dir) if text_mode == "dbi5" else []
    custom_prompt_templates = (
        _read_custom_prompt_templates(custom_prompt_file)
        if text_mode in {"custom_prompt", "custom_prompts", "prompt_file"}
        else []
    )
    domains_map = load_brand_domains(normal_urls_csv)
    overlap_hints: Dict[str, List[str]] = {}
    # Load brands_overlap.csv hints for both ckpt_text_proto modes AND dbi5 mode.
    # In dbi5 mode the logo-library brand dir name (brand_dir_7299) sometimes differs
    # from the DBI-5 directory name (e.g. "Bancolombia" -> "grupo_bancolombia");
    # the match_277_dirs column provides the authoritative crosswalk.
    if brands_overlap_csv and (text_mode in CKPT_TEXT_PROTO_MODES or text_mode == "dbi5"):
        overlap_hints = load_brand_overlap_match277(str(brands_overlap_csv))

    # For text modes that reuse ckpt text prototypes: build the name→protoidx map once.
    ckpt_name_to_proto_idx: Dict[str, int] = {}
    ckpt_text_proto_tensor: torch.Tensor | None = None
    if text_mode in CKPT_TEXT_PROTO_MODES:
        ckpt_name_to_proto_idx = _build_ckpt_brand_name_to_proto_index(encoder)
        ckpt_text_proto_tensor = getattr(encoder, "_ckpt_text_prototypes", None)
        if ckpt_text_proto_tensor is None or ckpt_text_proto_tensor.ndim != 2:
            raise RuntimeError(
                "brand_text_mode uses ckpt text prototypes but the loaded encoder checkpoint "
                "does not contain a valid 2D text_prototypes tensor."
            )

    logo_paths_by_brand: List[List[str]] = []
    text_lines_by_brand: List[List[str]] = []
    ckpt_proto_indices_by_brand: List[int] = []
    brand_domains: Dict[str, List[str]] = {}
    missing_dbi: List[str] = []
    missing_domains: List[str] = []
    missing_ckpt_proto: List[str] = []

    for brand in brand_names:
        dbi_brand_dir = ""
        if text_mode == "dbi5":
            hint_names: List[str] = list(overlap_hints.get(brand, []) or [])
            if not hint_names and _OVERLAP_ALIAS_RESOLVER is not None:
                try:
                    hint_names = list(_OVERLAP_ALIAS_RESOLVER(brand) or [])
                except Exception:
                    hint_names = []

            # Primary: exact logo-library brand name in the user-specified dbi5_dir
            # (e.g. cross_time uses 04_prompt_libraries_152/visual_all which matches
            # logo-library names exactly, so this resolves all 48 overlap brands directly).
            # Secondary: match_277_dirs hints → dbi5_dir (needed only when dbi5_dir
            # is a non-matching-name library like the 277-brand 06_prompt_library_277).
            # Final fallback: logo-library dir itself has embedded *_logo_description.txt.
            if not dbi_brand_dir:
                dbi_brand_dir = _resolve_optional_brand_dir(
                    dbi5_dir, brand, dbi_brand_names, hint_names=None
                )
            if not dbi_brand_dir:
                dbi_brand_dir = _resolve_optional_brand_dir(
                    dbi5_dir, brand, dbi_brand_names, hint_names=hint_names
                )
            if not dbi_brand_dir:
                logo_brand_dir = os.path.join(logo_library_dir, brand)
                if os.path.isdir(logo_brand_dir):
                    has_desc = any(
                        fn.lower().endswith("_logo_description.txt")
                        for fn in os.listdir(logo_brand_dir)
                    )
                    if has_desc:
                        dbi_brand_dir = logo_brand_dir
            if not dbi_brand_dir:
                missing_dbi.append(brand)
                continue

        logo_paths = list_logo_paths(os.path.join(logo_library_dir, brand))
        if int(max_logo_per_brand) > 0:
            logo_paths = logo_paths[: int(max_logo_per_brand)]
        if not logo_paths:
            raise ValueError(f"Brand '{brand}' has no logo images")

        text_lines: List[str] = []
        ckpt_proto_idx: int = -1
        if text_mode in CKPT_TEXT_PROTO_MODES:
            hints_list = list(overlap_hints.get(brand, []) or [])
            # If no hints via the exact dict key, try the closure resolver built from
            # overlap.csv (handles logo-library dirs whose normalized/alias forms match
            # canonical_name_7299/brand_dir_7299 uniquely).
            if not hints_list and _OVERLAP_ALIAS_RESOLVER is not None:
                try:
                    hints_list = list(_OVERLAP_ALIAS_RESOLVER(brand) or [])
                except Exception:
                    hints_list = []
            ckpt_proto_idx = resolve_ckpt_proto_label_idx_for_brand(
                brand, encoder, explicit_match_277_dirs=hints_list if hints_list else None
            )
            if ckpt_proto_idx < 0:
                missing_ckpt_proto.append(brand)
                continue
        elif text_mode == "dbi5":
            text_lines = read_dbi_texts(dbi_brand_dir)
        elif text_mode in {"train_templates_mean", "train12_mean", "train_templates"}:
            text_lines = _build_train_template_texts(brand)
        elif text_mode in {"custom_prompt", "custom_prompts", "prompt_file"}:
            text_lines = _build_custom_prompt_texts(brand, custom_prompt_templates)
        else:
            raise ValueError(f"Unsupported brand_text_mode: {brand_text_mode}")

        if text_mode not in CKPT_TEXT_PROTO_MODES and not text_lines:
            raise ValueError(f"Brand '{brand}' has empty text prompts under mode={brand_text_mode}")

        # Resolve domains using multiple lookup keys, matching lookup_hosts order:
        # original → brand_converter(original) → normalize_brand_key(original)
        # → normalize_brand_key(brand_converter(original))
        found_domains: List[str] = []
        for lookup_key in [brand, brand_converter(brand), normalize_brand_key(brand), normalize_brand_key(brand_converter(brand))]:
            if lookup_key and lookup_key in domains_map:
                found_domains = list(domains_map[lookup_key])
                break

        if not found_domains:
            missing_domains.append(brand)
            continue

        logo_paths_by_brand.append(logo_paths)
        text_lines_by_brand.append(text_lines)
        ckpt_proto_indices_by_brand.append(ckpt_proto_idx)
        brand_domains[brand] = list(found_domains)

    if missing_dbi:
        preview = ", ".join(missing_dbi[:20])
        more = "" if len(missing_dbi) <= 20 else f" ... +{len(missing_dbi) - 20} brands"
        raise ValueError(
            "DBI-5 text prompts are missing for brands from the logo library. "
            f"Missing {len(missing_dbi)}/{len(brand_names)}: {preview}{more}"
        )
    if missing_ckpt_proto:
        preview = ", ".join(missing_ckpt_proto[:20])
        more = "" if len(missing_ckpt_proto) <= 20 else f" ... +{len(missing_ckpt_proto) - 20} brands"
        raise ValueError(
            "Ckpt text prototypes are missing for brands from the logo library. "
            f"Missing {len(missing_ckpt_proto)}/{len(brand_names)}: {preview}{more}"
        )
    if missing_domains:
        preview = ", ".join(missing_domains[:20])
        more = "" if len(missing_domains) <= 20 else f" ... +{len(missing_domains) - 20} brands"
        raise ValueError(
            "Official domains are missing in normal URLs CSV. "
            f"Missing {len(missing_domains)}/{len(brand_names)}: {preview}{more}"
        )

    n_brand = len(brand_names)
    max_logo = max(len(x) for x in logo_paths_by_brand)
    d = int(encoder.embed_dim)

    logo_emb = torch.zeros((n_brand, max_logo, d), dtype=torch.float32, device=encoder.device)
    logo_mask = torch.zeros((n_brand, max_logo), dtype=torch.bool, device=encoder.device)

    use_ckpt_text_proto = text_mode in CKPT_TEXT_PROTO_MODES
    if use_ckpt_text_proto:
        assert ckpt_text_proto_tensor is not None
        text_proto = torch.zeros((n_brand, d), dtype=torch.float32, device=encoder.device)
        text_emb = torch.zeros((n_brand, 1, d), dtype=torch.float32, device=encoder.device)
        text_mask = torch.zeros((n_brand, 1), dtype=torch.bool, device=encoder.device)
    else:
        max_text = max(len(x) for x in text_lines_by_brand)
        text_emb = torch.zeros((n_brand, max_text, d), dtype=torch.float32, device=encoder.device)
        text_mask = torch.zeros((n_brand, max_text), dtype=torch.bool, device=encoder.device)

    flat_logo_items: List[Tuple[int, int, str]] = []
    for bid, paths in enumerate(logo_paths_by_brand):
        for j, path in enumerate(paths):
            flat_logo_items.append((bid, j, path))

    bad_logo_paths: List[str] = []
    for start in range(0, len(flat_logo_items), int(image_batch_size)):
        chunk = flat_logo_items[start : start + int(image_batch_size)]
        imgs: List[Image.Image] = []
        idxs: List[Tuple[int, int]] = []
        for bid, j, path in chunk:
            try:
                with Image.open(path) as im:
                    imgs.append(im.convert("RGB"))
                idxs.append((bid, j))
            except (UnidentifiedImageError, OSError, ValueError):
                bad_logo_paths.append(path)
                continue

        if not imgs:
            continue
        feats = encoder.encode_images(imgs, batch_size=max(1, len(imgs)))
        for k, (bid, j) in enumerate(idxs):
            logo_emb[bid, j] = feats[k]
            logo_mask[bid, j] = True

    if bad_logo_paths:
        print(f"[repo] skipped_bad_logo_images={len(bad_logo_paths)}")
        for p in bad_logo_paths[:10]:
            print(f"[repo] bad_logo_image: {p}")
        if len(bad_logo_paths) > 10:
            print(f"[repo] ... {len(bad_logo_paths) - 10} more bad logo images")

    valid_logo_per_brand = logo_mask.to(dtype=torch.int32).sum(dim=1).detach().cpu().numpy().astype(int).tolist()
    brands_without_valid_logo: List[str] = [
        brand_names[i] for i, n_valid in enumerate(valid_logo_per_brand) if int(n_valid) <= 0
    ]
    if brands_without_valid_logo:
        preview = ", ".join(brands_without_valid_logo[:10])
        more = "" if len(brands_without_valid_logo) <= 10 else f" ... +{len(brands_without_valid_logo) - 10} brands"
        raise RuntimeError(
            "Some brands have zero decodable logo images after filtering bad files: "
            f"{preview}{more}"
        )

    if use_ckpt_text_proto:
        # 1-slot embeddings (text_proto is the per-brand prototype from the ckpt; fill text_emb/text_mask
        # to keep downstream code paths intact)
        assert ckpt_text_proto_tensor is not None
        for bid, idx in enumerate(ckpt_proto_indices_by_brand):
            proto = torch.as_tensor(ckpt_text_proto_tensor[int(idx)], dtype=torch.float32, device=encoder.device).detach().clone()
            proto = proto / (proto.norm(dim=-1, keepdim=True) + 1e-6)
            text_proto[bid] = proto
            text_emb[bid, 0] = proto
            text_mask[bid, 0] = True
    else:
        flat_text_items: List[Tuple[int, int, str]] = []
        for bid, lines in enumerate(text_lines_by_brand):
            for j, line in enumerate(lines):
                flat_text_items.append((bid, j, line))

        for start in range(0, len(flat_text_items), int(text_batch_size)):
            chunk = flat_text_items[start : start + int(text_batch_size)]
            txts = [item[2] for item in chunk]
            feats = encoder.encode_texts(txts, batch_size=max(1, len(txts)))
            for k, (bid, j, _) in enumerate(chunk):
                text_emb[bid, j] = feats[k]
                text_mask[bid, j] = True

        # Average valid text prompt embeddings per brand; this matches the training-time text prototype.
        txt_valid = text_mask.to(dtype=torch.float32).unsqueeze(-1)
        txt_sum = (text_emb * txt_valid).sum(dim=1)
        txt_cnt = txt_valid.sum(dim=1).clamp_min(1.0)
        text_proto = txt_sum / txt_cnt
        text_proto = text_proto / (text_proto.norm(dim=-1, keepdim=True) + 1e-6)

    return BrandRepository(
        brand_names=brand_names,
        logo_emb=logo_emb,
        logo_mask=logo_mask,
        text_emb=text_emb,
        text_mask=text_mask,
        text_proto=text_proto,
        brand_domains=brand_domains,
    )


def score_query_features(
    query_feats: torch.Tensor,  # [Q, D]
    repo: BrandRepository,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return image and text similarities for every query crop and brand.

    The fixed 0.4/0.6 fusion is intentionally applied later in
    ``recognition.features`` after scores have been aggregated per screenshot.
    """
    if query_feats.ndim != 2:
        raise ValueError(f"query_feats must be 2D [Q,D], got shape={tuple(query_feats.shape)}")
    if int(query_feats.shape[0]) <= 0:
        b = int(len(repo.brand_names))
        empty = torch.zeros((0, b), dtype=torch.float32, device=query_feats.device)
        return empty, empty

    # Compare every crop to all logo slots, retaining each brand's best logo match.
    sim_img_full = torch.einsum("qd,bkd->qbk", query_feats, repo.logo_emb)
    sim_img_full = sim_img_full.masked_fill(~repo.logo_mask.unsqueeze(0), -1e9)
    sim_img = sim_img_full.max(dim=-1).values

    # Compare every crop to the mean text prototype of each brand.
    sim_txt = torch.einsum("qd,bd->qb", query_feats, repo.text_proto)
    return sim_img, sim_txt
