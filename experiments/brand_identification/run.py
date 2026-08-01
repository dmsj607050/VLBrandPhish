"""Brand identification on the unified cropped-logo set.

Research question: RQ2
Dataset: Cropped-logo eval set (2,700 crops: 1,352 phish + 1,348 benign, 110 brands)
Metrics: Match Acc
Paper table(s): Table: cn-brand-cls

This is a thin runner. It calls the three method inference packages under
``inference/<method>/`` and loads weights from ``models/<method>/`` via the shared
dispatch in ``experiments/_common.py``. The full runner (metric computation, CSV output,
plotting) is ported incrementally from the project-root ``07_*`` directory.
"""

from __future__ import annotations

import argparse
import os

from experiments._common import METHODS, ensure_dir, run_all_methods


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Brand identification on the unified cropped-logo set")
    parser.add_argument("--screenshot-dir", required=True, help="Directory of webpage screenshots.")
    parser.add_argument("--url-list", required=True, help="URL list / manifest for the samples.")
    parser.add_argument("--out-dir", default="outputs/brand_identification", help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), help="Methods to run.")
    args = parser.parse_args(argv)

    out_dir = ensure_dir(args.out_dir)
    run_all_methods(args.screenshot_dir, args.url_list, out_dir, methods=tuple(args.methods))


# Expected local data: data/vlbp_set/08_open_set_logo_crops_1352_seed20260501
if __name__ == "__main__":
    main()
