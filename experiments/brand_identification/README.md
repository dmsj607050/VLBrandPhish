# Brand identification on the unified cropped-logo set

| | |
| --- | --- |
| Research question | RQ2 |
| Dataset | Cropped-logo eval set (2,700 crops: 1,352 phish + 1,348 benign, 110 brands) |
| Metrics | Match Acc |
| Paper table(s) | Table: cn-brand-cls |
| Methods | vlbrandphish, phishintention, phishpedia |

## Run

```bash
python -m experiments.brand_identification.run \
  --screenshot-dir <screenshots> \
  --url-list <url manifest> \
  --out-dir outputs/brand_identification
```

Each method's predictions land in ``outputs/brand_identification/<method>/``. The runner dispatches to
``inference/<method>/`` and loads weights from ``models/<method>/`` (see
``experiments/_common.py``). Baselines require ``detectron2`` at runtime.

## Expected data

data/vlbp_set/08_open_set_logo_crops_1352_seed20260501

See [data/](../../data/) for dataset sources, sizes, and download links.
