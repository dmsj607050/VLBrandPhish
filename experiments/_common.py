"""Shared helpers for experiment runners.

Each experiment under ``experiments/<name>/`` is a thin runner that calls one of the
method inference packages (``inference/<method>/``) and loads weights from
``models/<method>/``. This module centralises path resolution and method dispatch so the
per-experiment runners stay small.

The actual full runners (with metric computation, CSV output, and plotting) live in the
project-root ``01_*`` .. ``09_*`` directories and can be ported here incrementally. These
templates establish the structure and the call pattern.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

# experiments/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

METHODS = ("vlbrandphish", "phishintention", "phishpedia")


def method_weights_dir(method: str) -> Path:
    """Return the weights directory for a method."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}; expected one of {METHODS}")
    return MODELS_DIR / method


def ensure_dir(path: str | Path) -> str:
    p = str(path)
    os.makedirs(p, exist_ok=True)
    return p


def run_vlbrandphish(screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Run VLBrandPhish end-to-end on a sample set.

    Loads the logo detector and brand recognizer from ``models/vlbrandphish/`` and calls
    the inference pipeline. Returns the path to the produced prediction CSV.
    """
    from inference.vlbrandphish.pipeline import main as pipeline_main  # noqa: F401
    # TODO: port the exact invocation from 01_*/code/experiment_runners/VLBrandPhish/.
    # The pipeline CLI accepts --det-csv, --brand-ckpt, and explicit dataset roots.
    raise NotImplementedError(
        "Port the VLBrandPhish runner from 01_*/code/experiment_runners/VLBrandPhish/. "
        "Weights: models/vlbrandphish/{logo_detector.pth, brand_recognizer.pt}."
    )


def run_phishintention(screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Run PhishIntention on a sample set via its Wrapper.

    Loads weights from ``models/phishintention/`` (see inference/phishintention/configs.py).
    Requires detectron2 at runtime.
    """
    from inference.phishintention.phishintention import PhishIntentionWrapper  # noqa: F401
    # TODO: iterate samples, call wrapper.test_orig_phishintention(url, screenshot_path),
    # and write predictions to out_dir. See 01_*/code/experiment_runners/PhishIntention/.
    raise NotImplementedError(
        "Port the PhishIntention runner from 01_*/code/experiment_runners/PhishIntention/. "
        "Entry: PhishIntentionWrapper.test_orig_phishintention(url, screenshot_path)."
    )


def run_phishpedia(screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Run PhishPedia on a sample set via its Wrapper.

    Loads weights from ``models/phishpedia/`` (see inference/phishpedia/configs.py).
    Requires detectron2 at runtime.
    """
    from inference.phishpedia.phishpedia import PhishpediaWrapper  # noqa: F401
    # TODO: iterate samples, call wrapper.test_orig_phishpedia(url, screenshot_path, html_path),
    # and write predictions to out_dir. See 01_*/code/experiment_runners/Phishpedia/.
    raise NotImplementedError(
        "Port the PhishPedia runner from 01_*/code/experiment_runners/Phishpedia/. "
        "Entry: PhishpediaWrapper.test_orig_phishpedia(url, screenshot_path, html_path)."
    )


_DISPATCH = {
    "vlbrandphish": run_vlbrandphish,
    "phishintention": run_phishintention,
    "phishpedia": run_phishpedia,
}


def run_method(method: str, screenshot_dir: str, url_list: str, out_dir: str) -> str:
    """Dispatch to a method's runner. Raises NotImplementedError until ported."""
    return _DISPATCH[method](screenshot_dir, url_list, out_dir)


def run_all_methods(screenshot_dir: str, url_list: str, out_dir: str, methods: Sequence[str] = METHODS) -> None:
    ensure_dir(out_dir)
    for method in methods:
        try:
            run_method(method, screenshot_dir, url_list, ensure_dir(os.path.join(out_dir, method)))
        except NotImplementedError as exc:
            print(f"[{method}] not ported yet: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{method}] failed: {type(exc).__name__}: {exc}")
