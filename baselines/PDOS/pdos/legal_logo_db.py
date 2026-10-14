"""Filesystem-backed legal logo reference database."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class LegalLogoDB:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def brands(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def paths_for(self, brand: str) -> list[Path]:
        directory = self.root / brand
        if not directory.exists():
            return []
        return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    def load_images(self, brand: str) -> list[tuple[Path, Image.Image]]:
        images = []
        for path in self.paths_for(brand):
            with Image.open(path) as image:
                images.append((path, image.convert("RGB")))
        return images
