"""Check a tracked research-artifact tree for basic release and anonymity hazards.

The checker is intentionally conservative: it reports paths and rule names, never
prints suspicious file contents. It does not replace a human anonymity, license,
or ethics review.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "docs/REPRODUCIBILITY.md",
    "docs/EXPERIMENTS.md",
    "docs/INFERENCE_WORKFLOW.md",
    "docs/DATA_AND_MODELS.md",
    "docs/USENIX_SECURITY_2027_READINESS.md",
)
REQUIRED_PATHS = (
    "cli.py",
    "inference",
    "training",
    "experiments",
    "training/vlbrandphish/brand_recognition/configs/stage1_siglip.yaml",
    "training/vlbrandphish/brand_recognition/configs/stage2_trisig_ot_safe.yaml",
)
LOCAL_ONLY_ALLOWLIST = {
    "data/.gitkeep",
    "data/README.md",
    "models/.gitkeep",
    "models/README.md",
    "outputs/.gitkeep",
}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".yaml", ".yml", ".json", ".cff", ".rst", ".sh"}
SENSITIVE_PATTERNS = {
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b|github_pat_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(rb"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "home-directory path": re.compile(rb"(?:[A-Za-z]:\\[Uu]sers\\|/(?:home|Users)/)"),
}


def _tracked_files(root: Path) -> list[Path]:
    """Return Git-tracked paths, falling back to a deliberately small public set."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        candidates = [
            root / name
            for name in ("README.md", "LICENSE", "pyproject.toml", "requirements.txt")
            if (root / name).is_file()
        ]
        for directory in ("inference", "training", "experiments", "scripts", "docs"):
            base = root / directory
            if base.is_dir():
                candidates.extend(path for path in base.rglob("*") if path.is_file())
        return sorted(candidates)

    return sorted(root / item.decode("utf-8") for item in output.split(b"\0") if item)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README", "LICENSE"}


def _scan_bytes(path: Path) -> Iterable[str]:
    if not _is_text_file(path):
        return ()
    try:
        payload = path.read_bytes()
    except OSError:
        return ("unreadable tracked file",)
    return tuple(name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(payload))


def _is_allowed_under_local_only(relative: str) -> bool:
    """Allow only documentation and explicit placeholders under local-artifact roots."""
    name = relative.rsplit("/", 1)[-1]
    if name in {".gitkeep", "README.md"}:
        return True
    if relative.startswith("data/") and name.endswith(".md"):
        return True
    return relative in LOCAL_ONLY_ALLOWLIST


def verify(root: Path) -> list[str]:
    failures: list[str] = []
    for required in REQUIRED_FILES:
        if not (root / required).is_file():
            failures.append(f"missing required release document: {required}")

    for required in REQUIRED_PATHS:
        if not (root / required).exists():
            failures.append(f"missing required source path: {required}")

    for path in _tracked_files(root):
        if not path.is_file():
            continue
        relative = _relative(path, root)
        top_level = relative.split("/", 1)[0]
        if top_level in {"data", "models", "outputs"} and not _is_allowed_under_local_only(relative):
            failures.append(f"tracked local-only artifact: {relative}")
        for issue in _scan_bytes(path):
            failures.append(f"{issue}: {relative}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a research artifact for basic release hazards.")
    parser.add_argument("--root", default=".", help="Artifact root (default: current directory).")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"artifact root is not a directory: {root}")

    failures = verify(root)
    if failures:
        print("Artifact check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Artifact check passed: required documents and source layout exist with no basic tracked-file hazards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())