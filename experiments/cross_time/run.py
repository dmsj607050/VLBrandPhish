"""Experiment 2: cross-time phishing detection on brands seen during training."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._shared.e2e import (
    add_e2e_arguments,
    execute_experiment,
    execute_from_existing_predictions,
    official_records,
)


def _seen_training_brands(dataset_root: str) -> set[str]:
    """Use the versioned in_277 split, not a heuristic name match."""
    split_path = Path(dataset_root) / "08_brands_overlap.csv"
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    brands = {
        str(row.get("brand_dir_7299", "") or "").strip()
        for row in rows
        if str(row.get("in_277", "")).strip() == "1"
    }
    if len(brands) != 48:
        raise ValueError(f"Expected 48 seen-training brands in {split_path}, found {len(brands)}")
    return brands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: cross-time VLBrandPhish evaluation.")
    add_e2e_arguments(parser)
    # Cross-time experiment MUST use 04_prompt_libraries_152/visual_all as the text
    # prompt library. This library uses logo-library native brand_dir_7299 names
    # (matching 03_logo_reference_library exactly), so all 48 overlap brands are
    # exact-name matches with no match_277_dirs redirect needed.
    dataset_root_default = parser.get_default("dataset_root")
    dbi5_default = str(Path(dataset_root_default) / "04_prompt_libraries_152" / "visual_all")
    for action in parser._actions:
        if getattr(action, "dest", "") == "dbi5_dir":
            action.default = dbi5_default
            break
    # Cross-time restrict_brands MUST be the 48 overlap brands (in_277==1) from
    # 03_logo_reference_library — same file that defines the selection split.
    restrict_default = str(Path(dataset_root_default) / "08_brands_overlap.csv")
    for action in parser._actions:
        if getattr(action, "dest", "") == "restrict_brands_csv":
            action.default = restrict_default
            break
    parser.add_argument("--from-overall-run", default="", help="Reuse predictions from a completed full-dataset run.")
    parser.add_argument("--max-benign", type=int, default=-1, help="Deterministic benign limit; -1 uses all readable benign samples.")
    parser.add_argument("--max-phish", type=int, default=-1, help="Deterministic phishing limit; -1 uses all 6,400.")
    args = parser.parse_args(argv)

    seen_brands = _seen_training_brands(args.dataset_root)
    records = official_records(
        args.dataset_root,
        selected_phish_brands=seen_brands,
        phish_split="seen_training_brands_recent",
        max_benign=int(args.max_benign),
        max_phish=int(args.max_phish),
    )
    protocol = {
        "name": "VLBP-Overlap / cross-time",
        "selection_file": str(Path(args.dataset_root) / "08_brands_overlap.csv"),
        "selection_rule": "in_277 == 1 (48 brands seen during brand-model training)",
        "expected_full_counts": {"phish": 6400, "benign": 12187, "brands": 48},
        "time_metadata": "collection_timestamp is frozen per sample in input_manifest.csv",
    }
    kwargs = {
        "experiment_name": "cross_time",
        "experiment_dir": Path(__file__).parent,
        "records": records,
        "args": args,
        "group_fields": ("dataset_split",),
        "protocol": protocol,
    }
    if args.from_overall_run:
        execute_from_existing_predictions(source_overall_run=args.from_overall_run, **kwargs)
    else:
        execute_experiment(**kwargs)


if __name__ == "__main__":
    main()