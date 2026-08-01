"""Logo localization analysis.

Research question: RQ2
Dataset: Three test sets: Default (901), Merged (2,480), New (12,801)
Metrics: COCO bbox mAP@[.50:.95]
Paper table(s): Table: cn-logo-det

This is a protocol scaffold. Its complete multi-method evaluator has not yet been ported; running it exits explicitly before any partial result directory is created. See `docs/EXPERIMENTS.md`.\r\n"""

from __future__ import annotations

import argparse
import os

from experiments._common import METHODS, run_all_methods


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Logo localization analysis")
    parser.add_argument("--screenshot-dir", required=True, help="Directory of webpage screenshots.")
    parser.add_argument("--url-list", required=True, help="URL list / manifest for the samples.")
    parser.add_argument("--out-dir", default="outputs/logo_detection", help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), help="Methods to run.")
    args = parser.parse_args(argv)

    run_all_methods(args.screenshot_dir, args.url_list, args.out_dir, methods=tuple(args.methods))


# Expected local data: data/phishintention/Logo_train, data/vlbp_set/01_phish_12K
if __name__ == "__main__":
    main()
