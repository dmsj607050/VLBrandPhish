"""Brand representation, encoding, and retrieval components."""

from .encoders import CLIPEncoder
from .repository import BrandRepository, build_brand_repository

__all__ = ["BrandRepository", "CLIPEncoder", "build_brand_repository"]
