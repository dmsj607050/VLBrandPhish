"""Adversarial robustness under logo manipulation and perturbation.

Research question: RQ4
Dataset: PhishEval: visible manipulation (13 types, 88,152 phish) + perturbation (6 types, 6,528 phish) + original (1,320)
Metrics: Phishing detection rate on squatted-domain samples; benign alert rate; embedding intra-brand distance & separation ratio (113-sample subset)
Paper table(s): Tables: cn-adversarial-summary, cn-adversarial-embedding-distance

This is a thin runner. It calls the three method inference packages under
``inference/<method>/`` and loads weights from ``models/<method>/`` via the shared
dispatch in ``experiments/_common.py``. The full runner (metric computation, CSV output,
plotting) is ported incrementally from the project-root ``09_*`` directory.
"""

from __future__ import annotations

import argparse
import os

from experiments._common import METHODS, ensure_dir, run_all_methods


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Adversarial robustness under logo manipulation and perturbation")
    parser.add_argument("--screenshot-dir", required=True, help="Directory of webpage screenshots.")
    parser.add_argument("--url-list", required=True, help="URL list / manifest for the samples.")
    parser.add_argument("--out-dir", default="outputs/adversarial_robustness", help="Output directory.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), help="Methods to run.")
    args = parser.parse_args(argv)

    out_dir = ensure_dir(args.out_dir)
    run_all_methods(args.screenshot_dir, args.url_list, out_dir, methods=tuple(args.methods))


# Expected local data: data/phisheval/
if __name__ == "__main__":
    main()
