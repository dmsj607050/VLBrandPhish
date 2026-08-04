"""Shared helpers for experiment runners.

The repository distinguishes implemented analysis tools from paper-study scaffolds.
The latter deliberately fail before they create output folders: a successful exit code
must always mean that predictions and metrics were actually produced.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

# experiments/_shared/ -> experiments/ -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

METHODS = ("vlbrandphish", "phishintention", "phishpedia")


class ExperimentImplementationRequired(RuntimeError):
    """Raised when a paper-study runner has not yet been ported."""


def method_weights_dir(method: str) -> Path:
    """Return the weights directory for a method."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}; expected one of {METHODS}")
    return MODELS_DIR / method


def ensure_dir(path: str | Path) -> str:
    p = str(path)
    os.makedirs(p, exist_ok=True)
    return p


def _unavailable(method: str) -> str:
    raise ExperimentImplementationRequired(
        f"The {method} paper-study runner is not yet ported. Do not treat this "
        "scaffold as an evaluation result; see README.md#reproducing-experiments."
    )


def run_vlbrandphish(screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Reserved for the complete VLBrandPhish paper-study evaluator."""
    return _unavailable("VLBrandPhish")


def run_phishintention(screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Reserved for the complete PhishIntention paper-study evaluator."""
    return _unavailable("PhishIntention")


def run_phishpedia(screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Reserved for the complete PhishPedia paper-study evaluator."""
    return _unavailable("PhishPedia")


_DISPATCH = {
    "vlbrandphish": run_vlbrandphish,
    "phishintention": run_phishintention,
    "phishpedia": run_phishpedia,
}


def run_method(method: str, screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Dispatch to one method, rejecting unknown names before doing any work."""
    if method not in _DISPATCH:
        raise ValueError(f"unknown method: {method}; expected one of {METHODS}")
    return _DISPATCH[method](screenshot_dir, url_list, out_dir)


def run_all_methods(
    screenshot_dir: str,
    url_list: str,
    out_dir: str,
    methods: Sequence[str] = METHODS,
) -> None:
    """Reject an incomplete study runner without producing partial output.

    The current public study runners are protocol declarations, not reproducible table
    pipelines. Failing explicitly is safer than hiding missing implementations behind
    log messages and returning success.
    """
    selected = tuple(methods)
    unknown = [method for method in selected if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; expected a subset of {METHODS}")
    if not selected:
        raise ValueError("at least one method must be selected")
    raise ExperimentImplementationRequired(
        "This study runner is a scaffold, not a table-reproduction command. "
        "Implemented cache-based analyses live under experiments/analysis and "
        "experiments/ablations/prompt_ablation; see README.md#reproducing-experiments."
    )
