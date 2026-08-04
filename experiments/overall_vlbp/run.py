"""Experiment 1: phishing detection on the complete official VLBP dataset.

All 152 brands, no sample or brand filtering -- full end-to-end evaluation.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1: full-dataset VLBrandPhish evaluation (all 152 brands)."
    )
    add_e2e_arguments(parser)
    # 04_prompt_libraries_152/visual_all is the DBI-5 text-prompt library for
    # all 152 VLBP Official brands (logo-library-native brand_dir_7299 names).
    dataset_root_default = parser.get_default("dataset_root")
    dbi5_default = str(Path(dataset_root_default) / "04_prompt_libraries_152" / "visual_all")
    for action in parser._actions:
        if getattr(action, "dest", "") == "dbi5_dir":
            action.default = dbi5_default
            break
    parser.add_argument(
        "--from-overall-run",
        default="",
        help="Reuse predictions from a completed full-dataset run.",
    )
    parser.add_argument(
        "--max-benign",
        type=int,
        default=-1,
        help="Deterministic benign limit; -1 uses all (12,187).",
    )
    parser.add_argument(
        "--max-phish",
        type=int,
        default=-1,
        help="Deterministic phishing limit; -1 uses all (12,801).",
    )
    args = parser.parse_args(argv)

    records = official_records(
        args.dataset_root,
        phish_split="full",
        max_benign=int(args.max_benign),
        max_phish=int(args.max_phish),
    )
    protocol = {
        "name": "VLBP-Full / overall",
        "selection": "all samples under 01_phish_12K and 02_bengin_12K (152 brands, no filtering)",
        "expected_full_counts": {"phish": 12801, "benign": 12164, "brands": 152},
    }
    kwargs = {
        "experiment_name": "overall_vlbp",
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
