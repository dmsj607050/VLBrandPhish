"""Record portable, reviewable metadata for one training or evaluation run.

The default record avoids Git identifiers and absolute paths so it can be reviewed
before use in a double-blind artifact. This script does not make a run
deterministic; it preserves the information needed to assess or repeat one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAMES = ("torch", "torchvision", "transformers", "numpy", "mmdet", "mmcv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_path(path: Path, root: Path) -> str:
    """Return a reproducible path without storing an absolute user-directory path."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _file_record(raw_path: str, root: Path) -> dict[str, Any]:
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input is missing or not a file: {raw_path}")
    return {
        "path": _public_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # Optional runtime information must never block logging.
        return {"available": False, "error": type(exc).__name__}

    runtime: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        runtime["device_count"] = torch.cuda.device_count()
        runtime["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return runtime


def _git_metadata(root: Path) -> dict[str, Any]:
    """Collect minimal revision state only when the caller explicitly opts in."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    return {"available": True, "commit": commit, "dirty_tracked_tree": dirty}


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")
    record: dict[str, Any] = {
        "format": "vlbrandphish-run-metadata-v1",
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": args.seed,
        "command": args.command,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "config_files": [_file_record(path, root) for path in args.config],
        "input_files": [_file_record(path, root) for path in args.input],
    }
    if args.collect_torch_runtime:
        record["torch_runtime"] = _torch_runtime()
    if args.include_git:
        record["git"] = _git_metadata(root)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Record hashes and environment metadata for one experiment run.")
    parser.add_argument("--root", default=".", help="Repository root used to redact absolute paths (default: current directory).")
    parser.add_argument("--output", required=True, help="Output JSON path; typically inside an ignored run-output directory.")
    parser.add_argument("--config", action="append", required=True, help="Frozen config file to hash; repeatable.")
    parser.add_argument("--input", action="append", default=[], help="Input manifest, checkpoint, or result file to hash; repeatable.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed used for this run.")
    parser.add_argument("--command", default=None, help="Exact run command after removing paths or identifiers that would break anonymity.")
    parser.add_argument("--collect-torch-runtime", action="store_true", help="Record PyTorch/CUDA/GPU runtime details when PyTorch is installed.")
    parser.add_argument("--include-git", action="store_true", help="Include commit and dirty state. Do not use in a blind-review artifact.")
    args = parser.parse_args()

    try:
        record = build_record(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run metadata to {output}")
    if args.include_git:
        print("Warning: Git metadata can deanonymize a blind-review artifact; review before sharing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())