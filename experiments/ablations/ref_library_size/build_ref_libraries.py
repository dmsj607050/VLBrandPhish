"""Build nested reference-logo libraries for the reference-count ablation.

For each ``M`` in ``--counts`` a directory ``ref_library_N{M}/<brand>/`` is created
whose contents (symlinks) are the first ``M`` reference images of that brand under a
single deterministic rule, identical to the training-side rule in
``model_train/Brand_Identity/TriSig/src/datasets.py``:

1. list image files (``.png/.jpg/.jpeg/.bmp/.webp``);
2. drop the auxiliary ``homepage.png`` / ``loginpage.png``;
3. sort filenames ascending;
4. keep the first ``M``.

The result is nested: ``M=1 ⊂ M=3 ⊂ M=5 ⊂ M=10``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "inference").is_dir())
DEFAULT_SOURCE_LIB = REPO_ROOT / "data" / "PhishIntention" / "277_brand_pages"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXCLUDE_FILENAMES = {"homepage.png", "loginpage.png"}


def list_reference_files(brand_dir: Path) -> list[str]:
    files = [
        fn
        for fn in os.listdir(brand_dir)
        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS and fn.lower() not in EXCLUDE_FILENAMES
    ]
    files.sort()
    return files


def _link_or_copy_file(source: Path, destination: Path) -> None:
    """Prefer a symlink, then a hard link, and copy only if both are unavailable."""
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested per-count reference-logo libraries.")
    parser.add_argument("--source-lib", default=str(DEFAULT_SOURCE_LIB))
    parser.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--counts", default="1,3,5,10")
    args = parser.parse_args()

    source = Path(args.source_lib).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source reference library not found: {source}")
    counts = [int(x) for x in str(args.counts).split(",") if str(x).strip()]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    brands = sorted(d for d in os.listdir(source) if os.path.isdir(source / d))
    manifest_rows: list[dict[str, object]] = []
    for count in counts:
        lib_dir = out_root / f"ref_library_N{count}"
        lib_dir.mkdir(parents=True, exist_ok=True)
        for brand in brands:
            brand_dir = source / brand
            files = list_reference_files(brand_dir)
            selected = files[:count]
            dst = lib_dir / brand
            dst.mkdir(parents=True, exist_ok=True)
            for old in list(dst.iterdir()):
                if old.is_symlink() or old.is_file():
                    old.unlink()
            for fn in selected:
                link = dst / fn
                if not link.exists():
                    _link_or_copy_file(brand_dir / fn, link)
            manifest_rows.append(
                {
                    "count": count,
                    "brand": brand,
                    "n_available": len(files),
                    "n_selected": len(selected),
                    "selected": ";".join(selected),
                }
            )

    manifest_path = out_root / "ref_library_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["count", "brand", "n_available", "n_selected", "selected"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[ref-lib] built {len(counts)} libraries x {len(brands)} brands under {out_root}")
    print(f"[ref-lib] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
