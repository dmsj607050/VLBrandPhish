# Overall detection on the PhishIntention dataset

| | |
| --- | --- |
| Research question | RQ1 |
| Dataset | PhishIntention dataset (24,532 phishing + 21,974 benign, 277 brands) |
| Metrics | Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc |
| Paper table(s) | Table: cn-e2e-testa |
| Methods | vlbrandphish, phishintention, phishpedia |

## Status

This directory records the intended protocol and required data, but its `run.py` is not
yet the complete multi-method evaluator required to reproduce the associated paper table.
It exits with an explicit error before writing partial outputs. See
[`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the implementation boundary and
reproduction requirements.
## Expected data

data/phishintention/phish_sample_30k, data/phishintention/benign_25k

See [data/](../../data/) for dataset sources, sizes, and download links.
