"""Create a deterministic SHA-256 manifest for an artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_relative_paths(root: Path) -> set[str]:
    """Return Git-tracked paths, without reading any untracked local artifacts."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("--tracked-only requires a Git working tree") from exc
    return {item.decode("utf-8") for item in output.split(b"\0") if item}


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
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Include only Git-tracked files; prevents local data, weights, caches, and bytecode from entering the manifest.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        parser.error("output must be inside --root")

    try:
        tracked = _tracked_relative_paths(root) if args.tracked_only else None
    except RuntimeError as exc:
        parser.error(str(exc))

    files: dict[str, Path] = {}
    try:
        for include in args.include:
            for path in _files_for_include(root, include):
                relative = path.relative_to(root).as_posix()
                if path == output or (tracked is not None and relative not in tracked):
                    continue
                files[relative] = path
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    manifest = {
        "format": "vlbrandphish-artifact-manifest-v1",
        "tracked_only": bool(args.tracked_only),
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