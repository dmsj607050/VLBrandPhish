"""Overall detection on the PhishIntention dataset.

Research question: RQ1
Dataset: PhishIntention dataset (24,532 phishing + 21,974 benign, 277 brands)
Metrics: Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc
Paper table(s): Table: cn-e2e-testa

This is a protocol scaffold. Its complete multi-method evaluator has not yet been ported; running it exits explicitly before any partial result directory is created. See `docs/EXPERIMENTS.md`.\r\n"""

from __future__ import annotations

import argparse
import os

from experiments._common import METHODS, run_all_methods


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Overall detection on the PhishIntention dataset")
    parser.add_argument("--screenshot-dir", required=True, help="Directory of webpage screenshots.")
    parser.add_argument("--url-list", required=True, help="URL list / manifest for the samples.")
    parser.add_argument("--out-dir", default="outputs/overall_phishintention", help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), help="Methods to run.")
    args = parser.parse_args(argv)

    run_all_methods(args.screenshot_dir, args.url_list, args.out_dir, methods=tuple(args.methods))


# Expected local data: data/phishintention/phish_sample_30k, data/phishintention/benign_25k
if __name__ == "__main__":
    main()
