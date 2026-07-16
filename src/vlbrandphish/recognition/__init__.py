"""Brand representation, encoding, and similarity retrieval."""

from .encoders import CLIPEncoder
from .repository import BrandRepository, build_brand_repository

__all__ = ["BrandRepository", "CLIPEncoder", "build_brand_repository"]
