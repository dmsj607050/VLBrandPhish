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

from .naming import normalize_brand_key, parse_semicolon_list, read_nonempty_lines
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


def _resolve_optional_brand_dir(root: str, brand: str, names: List[str]) -> str:
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
    """Load official hostnames keyed by normalized brand without depending on decision policy."""
    def _host_from_url_raw(url: str) -> str:
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

    out: Dict[str, List[str]] = {}
    with open(normal_urls_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = str(row.get("logo_brand", "") or "").strip()
            domains_raw = str(row.get("url", "") or row.get("domain", "") or "").strip()
            if not brand:
                continue
            key = normalize_brand_key(brand)
            if not key:
                continue
            domains: List[str] = []
            for item in parse_semicolon_list(domains_raw):
                d = _host_from_url_raw(item)
                if d:
                    domains.append(d)
            if not domains:
                continue
            if key not in out:
                out[key] = []
            exists = set(out[key])
            for d in domains:
                if d not in exists:
                    out[key].append(d)
                    exists.add(d)
    if not out:
        raise ValueError(f"No valid brand domains loaded from: {normal_urls_csv}")
    return out


def build_brand_repository(
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    encoder: CLIPEncoder,
    max_logo_per_brand: int = -1,
    image_batch_size: int = 64,
    text_batch_size: int = 256,
    brand_text_mode: str = "dbi5",
    custom_prompt_file: str = "",
) -> BrandRepository:
    logo_library_dir = os.path.abspath(logo_library_dir)
    dbi5_dir = os.path.abspath(dbi5_dir)
    normal_urls_csv = os.path.abspath(normal_urls_csv)

    brand_names = list_brand_dirs(logo_library_dir)
    text_mode = str(brand_text_mode or "dbi5").strip().lower()
    dbi_brand_names = list_brand_dirs(dbi5_dir) if text_mode == "dbi5" else []
    custom_prompt_templates = (
        _read_custom_prompt_templates(custom_prompt_file)
        if text_mode in {"custom_prompt", "custom_prompts", "prompt_file"}
        else []
    )
    domains_map = load_brand_domains(normal_urls_csv)

    logo_paths_by_brand: List[List[str]] = []
    text_lines_by_brand: List[List[str]] = []
    brand_domains: Dict[str, List[str]] = {}
    missing_dbi: List[str] = []
    missing_domains: List[str] = []

    for brand in brand_names:
        dbi_brand_dir = ""
        if text_mode == "dbi5":
            dbi_brand_dir = _resolve_optional_brand_dir(dbi5_dir, brand, dbi_brand_names)
            if not dbi_brand_dir:
                missing_dbi.append(brand)
                continue

        logo_paths = list_logo_paths(os.path.join(logo_library_dir, brand))
        if int(max_logo_per_brand) > 0:
            logo_paths = logo_paths[: int(max_logo_per_brand)]
        if not logo_paths:
            raise ValueError(f"Brand '{brand}' has no logo images")

        if text_mode == "dbi5":
            text_lines = read_dbi_texts(dbi_brand_dir)
        elif text_mode in {"train_templates_mean", "train12_mean", "train_templates"}:
            text_lines = _build_train_template_texts(brand)
        elif text_mode in {"custom_prompt", "custom_prompts", "prompt_file"}:
            text_lines = _build_custom_prompt_texts(brand, custom_prompt_templates)
        else:
            raise ValueError(f"Unsupported brand_text_mode: {brand_text_mode}")
        if not text_lines:
            raise ValueError(f"Brand '{brand}' has empty text prompts under mode={brand_text_mode}")

        key = normalize_brand_key(brand)
        if key not in domains_map:
            missing_domains.append(brand)
            continue

        logo_paths_by_brand.append(logo_paths)
        text_lines_by_brand.append(text_lines)
        brand_domains[brand] = list(domains_map[key])

    if missing_dbi:
        preview = ", ".join(missing_dbi[:20])
        more = "" if len(missing_dbi) <= 20 else f" ... +{len(missing_dbi) - 20} brands"
        raise ValueError(
            "DBI-5 text prompts are missing for brands from the logo library. "
            f"Missing {len(missing_dbi)}/{len(brand_names)}: {preview}{more}"
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
    max_text = max(len(x) for x in text_lines_by_brand)
    d = int(encoder.embed_dim)

    logo_emb = torch.zeros((n_brand, max_logo, d), dtype=torch.float32, device=encoder.device)
    logo_mask = torch.zeros((n_brand, max_logo), dtype=torch.bool, device=encoder.device)
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
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score query crops against the repository, returning fused/image/text similarities.

    Returns ``(fused, sim_img, sim_txt)`` each of shape ``[Q, B]`` where B is the
    number of brands. ``sim_img`` takes the max over per-brand logo slots; ``sim_txt``
    uses the mean text prototype rather than per-line max to match training.
    """
    if query_feats.ndim != 2:
        raise ValueError(f"query_feats must be 2D [Q,D], got shape={tuple(query_feats.shape)}")
    if int(query_feats.shape[0]) <= 0:
        b = int(len(repo.brand_names))
        empty = torch.zeros((0, b), dtype=torch.float32, device=query_feats.device)
        return empty, empty, empty

    # Compare every query crop against every logo slot, then keep each brand's strongest logo match.
    sim_img_full = torch.einsum("qd,bkd->qbk", query_feats, repo.logo_emb)
    sim_img_full = sim_img_full.masked_fill(~repo.logo_mask.unsqueeze(0), -1e9)
    sim_img = sim_img_full.max(dim=-1).values

    # Compare each query crop against the mean text prototype of every brand.
    sim_txt = torch.einsum("qd,bd->qb", query_feats, repo.text_proto)

    fused = float(alpha) * sim_img + (1.0 - float(alpha)) * sim_txt
    return fused, sim_img, sim_txt
