# Logo localization analysis

| | |
| --- | --- |
| Research question | RQ2 |
| Dataset | Three test sets: Default (901), Merged (2,480), New (12,801) |
| Metrics | COCO bbox mAP@[.50:.95] |
| Paper table(s) | Table: cn-logo-det |
| Methods | vlbrandphish, phishintention, phishpedia |

## Status

This directory records the intended protocol and required data, but its `run.py` is not
yet the complete multi-method evaluator required to reproduce the associated paper table.
It exits with an explicit error before writing partial outputs. See
[`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the implementation boundary and
reproduction requirements.
## Expected data

data/phishintention/Logo_train, data/vlbp_set/01_phish_12K

See [data/](../../data/) for dataset sources, sizes, and download links.
