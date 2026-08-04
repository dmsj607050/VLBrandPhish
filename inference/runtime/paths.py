"""Resolve portable repository-relative defaults and timestamped output directories."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict


@dataclass(frozen=True)
class DefaultPaths:
    """Repository-relative defaults used only when a CLI option is omitted."""
    phishkiller_root: str
    testb_root: str
    benign_root: str
    phish_root: str
    logo_library_dir: str
    dbi5_dir: str
    det_csv: str
    normal_urls_csv: str
    brand_ckpt: str
    detector_ckpt: str

    def to_dict(self) -> Dict[str, str]:
        """Return a serializable mapping of resolved default paths."""
        return dict(asdict(self))


REL_PATHS = {
    "testb_root": "data/inputs",
    "benign_root": "data/inputs/benign",
    "phish_root": "data/inputs/phish",
    "logo_library_dir": "data/brand_library/images",
    "dbi5_dir": "data/brand_library/DBI-5",
    "det_csv": "outputs/detection/logo_detection_top3.csv",
    "normal_urls_csv": "data/brand_library/normal_urls.csv",
    "brand_ckpt": "models/vlbrandphish/brand_identify.pt",
    "detector_ckpt": "models/vlbrandphish/detector.pth",
}


def _abs_from_root(root: str, rel_path: str) -> str:
    return os.path.abspath(os.path.join(root, rel_path))


def resolve_default_paths(this_file: str) -> DefaultPaths:
    """Resolve defaults from the repository containing ``this_file``.

    Paths are resolved relative to the repository root (three levels up from this
    file: runtime/ -> vlbrandphish/ -> inference/ -> repository root) so that
    """
    # runtime/ is three levels below the repository root.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(this_file)), "..", "..", ".."))
    testb_root = _abs_from_root(repo_root, REL_PATHS["testb_root"])

    return DefaultPaths(
        # Kept as a stable field name for configuration compatibility; its
        # value is the repository root, never an external project directory.
        phishkiller_root=repo_root,
        testb_root=testb_root,
        benign_root=_abs_from_root(repo_root, REL_PATHS["benign_root"]),
        phish_root=_abs_from_root(repo_root, REL_PATHS["phish_root"]),
        logo_library_dir=_abs_from_root(repo_root, REL_PATHS["logo_library_dir"]),
        dbi5_dir=_abs_from_root(repo_root, REL_PATHS["dbi5_dir"]),
        det_csv=_abs_from_root(repo_root, REL_PATHS["det_csv"]),
        normal_urls_csv=_abs_from_root(repo_root, REL_PATHS["normal_urls_csv"]),
        brand_ckpt=_abs_from_root(repo_root, REL_PATHS["brand_ckpt"]),
        detector_ckpt=_abs_from_root(repo_root, REL_PATHS["detector_ckpt"]),
    )


def default_run_dir(base_out_dir: str, prefix: str) -> str:
    """Return a timestamped output path without creating it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.abspath(base_out_dir), f"{prefix}_{stamp}")
