# Adversarial robustness under logo manipulation and perturbation

| | |
| --- | --- |
| Research question | RQ4 |
| Dataset | PhishEval: visible manipulation (13 types, 88,152 phish) + perturbation (6 types, 6,528 phish) + original (1,320) |
| Metrics | Phishing detection rate on squatted-domain samples; benign alert rate; embedding intra-brand distance & separation ratio (113-sample subset) |
| Paper table(s) | Tables: cn-adversarial-summary, cn-adversarial-embedding-distance |
| Methods | vlbrandphish, phishintention, phishpedia |

## Status

This directory records the intended protocol and required data, but its `run.py` is not
yet the complete multi-method evaluator required to reproduce the associated paper table.
It exits with an explicit error before writing partial outputs. See
[`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the implementation boundary and
reproduction requirements.
## Expected data

data/phisheval/

See [data/](../../data/) for dataset sources, sizes, and download links.
