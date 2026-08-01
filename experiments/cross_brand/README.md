# Cross-brand stability

| | |
| --- | --- |
| Research question | RQ3 |
| Dataset | VLBP-Novel (6,401 phishing, 104 brands unseen in training) + 12,187 benign |
| Metrics | Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc |
| Paper table(s) | Table: cn-cross-brand |
| Methods | vlbrandphish, phishintention, phishpedia |

## Status

This directory records the intended protocol and required data, but its `run.py` is not
yet the complete multi-method evaluator required to reproduce the associated paper table.
It exits with an explicit error before writing partial outputs. See
[`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the implementation boundary and
reproduction requirements.
## Expected data

data/vlbp_set/01_phish_12K (novel split), data/vlbp_set/02_bengin_12K

See [data/](../../data/) for dataset sources, sizes, and download links.
