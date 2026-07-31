"""Load the brand encoder and its cached or freshly built evidence repository."""

from __future__ import annotations

import os
import time
from typing import Tuple

from ..components.brand.encoders import CLIPEncoder
from ..components.brand.repository import BrandRepository, build_brand_repository
from .cache import _build_repo_cache_key, _load_brand_repo_cache, _save_brand_repo_cache


def load_brand_resources(
    *,
    brand_checkpoint: str,
    logo_library_dir: str,
    dbi5_dir: str,
    normal_urls_csv: str,
    cache_dir: str,
    device: str,
    max_logo_per_brand: int,
    image_batch_size: int,
    text_batch_size: int,
    brand_text_mode: str,
    use_repository_cache: bool,
    rebuild_repository_cache: bool,
) -> Tuple[CLIPEncoder, BrandRepository, str, bool]:
    """Return the encoder, brand repository, cache key, and repository-cache hit flag."""
    t0 = time.perf_counter()
    encoder = CLIPEncoder(os.path.abspath(brand_checkpoint), device=str(device))
    print(f"[inference] encoder loaded in {time.perf_counter() - t0:.2f}s")

    repo_cache_key = _build_repo_cache_key(
        brand_ckpt=os.path.abspath(brand_checkpoint),
        logo_library_dir=os.path.abspath(logo_library_dir),
        dbi5_dir=os.path.abspath(dbi5_dir),
        normal_urls_csv=os.path.abspath(normal_urls_csv),
        max_logo_per_brand=int(max_logo_per_brand),
        brand_text_mode=str(brand_text_mode),
    )
    repo_cache_path = os.path.join(os.path.abspath(cache_dir), f"brand_repo_{repo_cache_key}.pt")
    repo_cache_hit = False
    repo: BrandRepository | None = None
    t1 = time.perf_counter()

    if use_repository_cache and not rebuild_repository_cache and os.path.isfile(repo_cache_path):
        try:
            repo = _load_brand_repo_cache(repo_cache_path, encoder.device)
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
        )
        if use_repository_cache:
            try:
                _save_brand_repo_cache(repo_cache_path, repo)
                print(f"[cache] brand_repo saved: {repo_cache_path}")
            except Exception as exc:
                print(f"[cache] brand_repo save failed: {type(exc).__name__}: {exc}")

    print(
        f"[inference] brand repository ready in {time.perf_counter() - t1:.2f}s "
        f"brands={len(repo.brand_names)} cache_hit={int(repo_cache_hit)}"
    )
    return encoder, repo, repo_cache_key, repo_cache_hit
