"""Create a deterministic SHA-256 manifest for an artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_for_include(root: Path, include: str) -> Iterable[Path]:
    path = (root / include).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"include escapes artifact root: {include}") from exc
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    else:
        raise FileNotFoundError(f"include does not exist: {include}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a SHA-256 manifest for selected artifact files.")
    parser.add_argument("--root", default=".", help="Artifact root (default: current directory).")
    parser.add_argument("--include", action="append", required=True, help="File or directory relative to --root; repeatable.")
    parser.add_argument("--output", required=True, help="Manifest path relative to --root.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        parser.error("output must be inside --root")

    files: dict[str, Path] = {}
    try:
        for include in args.include:
            for path in _files_for_include(root, include):
                if path != output:
                    files[path.relative_to(root).as_posix()] = path
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    manifest = {
        "format": "vlbrandphish-artifact-manifest-v1",
        "files": [
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for relative, path in sorted(files.items())
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['files'])} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())