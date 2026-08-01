"""Cross-brand stability.

Research question: RQ3
Dataset: VLBP-Novel (6,401 phishing, 104 brands unseen in training) + 12,187 benign
Metrics: Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc
Paper table(s): Table: cn-cross-brand

This is a protocol scaffold. Its complete multi-method evaluator has not yet been ported; running it exits explicitly before any partial result directory is created. See `docs/EXPERIMENTS.md`.\r\n"""

from __future__ import annotations

import argparse
import os

from experiments._common import METHODS, run_all_methods


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-brand stability")
    parser.add_argument("--screenshot-dir", required=True, help="Directory of webpage screenshots.")
    parser.add_argument("--url-list", required=True, help="URL list / manifest for the samples.")
    parser.add_argument("--out-dir", default="outputs/cross_brand", help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), help="Methods to run.")
    args = parser.parse_args(argv)

    run_all_methods(args.screenshot_dir, args.url_list, args.out_dir, methods=tuple(args.methods))


# Expected local data: data/vlbp_set/01_phish_12K (novel split), data/vlbp_set/02_bengin_12K
if __name__ == "__main__":
    main()
