# Brand identification on the unified cropped-logo set

| | |
| --- | --- |
| Research question | RQ2 |
| Dataset | Cropped-logo eval set (2,700 crops: 1,352 phish + 1,348 benign, 110 brands) |
| Metrics | Match Acc |
| Paper table(s) | Table: cn-brand-cls |
| Methods | vlbrandphish, phishintention, phishpedia |

## Status

This directory records the intended protocol and required data, but its `run.py` is not
yet the complete multi-method evaluator required to reproduce the associated paper table.
It exits with an explicit error before writing partial outputs. See
[`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for the implementation boundary and
reproduction requirements.
## Expected data

data/vlbp_set/08_open_set_logo_crops_1352_seed20260501

See [data/](../../data/) for dataset sources, sizes, and download links.
