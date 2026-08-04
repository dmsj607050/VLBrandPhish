"""Load the brand encoder and its cached or freshly built evidence repository."""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Tuple

from ..components.brand.encoders import CLIPEncoder
from ..components.brand.repository import BrandRepository, CKPT_TEXT_PROTO_MODES, build_brand_repository
from ..runtime.cache import build_repository_cache_key, load_brand_repository_cache, save_brand_repository_cache


def load_brand_resources(
    *,
    brand_checkpoint: str,
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    cache_dir: str,
    device: str,
    model_dir: str = "",
    max_logo_per_brand: int = -1,
    image_batch_size: int,
    text_batch_size: int,
    brand_text_mode: str,
    use_repository_cache: bool,
    rebuild_repository_cache: bool,
    brands_overlap_csv: str = "",
    restrict_brands_to: Optional[Sequence[str]] = None,
) -> Tuple[CLIPEncoder, BrandRepository, str, bool, str]:
    """Return the encoder, repository, cache key/hit flag, and cache path."""
    t0 = time.perf_counter()
    text_mode_key = str(brand_text_mode or "dbi5").strip().lower()
    # Skip text encoder load when the selected text mode does not require text encoding.
    # This avoids Windows sentencepiece/protobuf crashes for SigLIP checkpoints that only
    # have a slow SPM tokenizer saved.
    skip_text_encoder = text_mode_key in CKPT_TEXT_PROTO_MODES
    encoder = CLIPEncoder(
        os.path.abspath(brand_checkpoint),
        device=str(device),
        model_dir=str(model_dir or ""),
        skip_text_encoder=skip_text_encoder,
    )
    print(f"[inference] encoder loaded in {time.perf_counter() - t0:.2f}s")

    restrict_key_suffix = ""
    if restrict_brands_to:
        # Include a stable sorted list of restricted brand names in the cache key so
        # different ablations don't accidentally share the same repository cache.
        restrict_key_suffix = "|brands=" + ",".join(sorted({str(x).strip() for x in restrict_brands_to if str(x).strip()}))

    repo_cache_key = build_repository_cache_key(
        brand_ckpt=os.path.abspath(brand_checkpoint),
        model_dir=os.path.abspath(model_dir) if model_dir else "",
        logo_library_dir=os.path.abspath(logo_library_dir),
        dbi5_dir=os.path.abspath(dbi5_dir),
        normal_urls_csv=os.path.abspath(normal_urls_csv),
        max_logo_per_brand=int(max_logo_per_brand),
        brand_text_mode=str(brand_text_mode),
        brands_overlap_csv=os.path.abspath(brands_overlap_csv) if brands_overlap_csv else "",
        extra_suffix=restrict_key_suffix,
    )
    repo_cache_path = os.path.join(os.path.abspath(cache_dir), f"brand_repo_{repo_cache_key}.pt")
    repo_cache_hit = False
    repo: BrandRepository | None = None
    t1 = time.perf_counter()

    if use_repository_cache and not rebuild_repository_cache and os.path.isfile(repo_cache_path):
        try:
            repo = load_brand_repository_cache(repo_cache_path, encoder.device)
            repo_cache_hit = True
            print(f"[cache] brand_repo hit: {repo_cache_path}")
        except Exception as exc:
            print(f"[cache] brand_repo load failed, rebuilding: {type(exc).__name__}: {exc}")

    if repo is None:
        repo = build_brand_repository(
            logo_library_dir=os.path.abspath(logo_library_dir),
            dbi5_dir=os.path.abspath(dbi5_dir),
            normal_urls_csv=os.path.abspath(normal_urls_csv),
            encoder=encoder,
            max_logo_per_brand=int(max_logo_per_brand),
            image_batch_size=int(image_batch_size),
            text_batch_size=int(text_batch_size),
            brand_text_mode=str(brand_text_mode),
            brands_overlap_csv=os.path.abspath(brands_overlap_csv) if brands_overlap_csv else "",
            restrict_brands_to=list(restrict_brands_to) if restrict_brands_to else None,
        )
        if use_repository_cache:
            try:
                save_brand_repository_cache(repo_cache_path, repo)
                print(f"[cache] brand_repo saved: {repo_cache_path}")
            except Exception as exc:
                print(f"[cache] brand_repo save failed: {type(exc).__name__}: {exc}")

    print(
        f"[inference] brand repository ready in {time.perf_counter() - t1:.2f}s "
        f"brands={len(repo.brand_names)} cache_hit={int(repo_cache_hit)}"
    )
    return encoder, repo, repo_cache_key, repo_cache_hit, repo_cache_path
