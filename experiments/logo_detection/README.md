# Logo localization analysis

| | |
| --- | --- |
| Research question | RQ2 |
| Dataset | Three test sets: Default (901), Merged (2,480), New (12,801) |
| Metrics | COCO bbox mAP@[.50:.95] |
| Paper table(s) | Table: cn-logo-det |
| Methods | vlbrandphish, phishintention, phishpedia |

## Run

```bash
python -m experiments.logo_detection.run \
  --screenshot-dir <screenshots> \
  --url-list <url manifest> \
  --out-dir outputs/logo_detection
```

Each method's predictions land in ``outputs/logo_detection/<method>/``. The runner dispatches to
``inference/<method>/`` and loads weights from ``models/<method>/`` (see
``experiments/_common.py``). Baselines require ``detectron2`` at runtime.

## Expected data

data/phishintention/Logo_train, data/vlbp_set/01_phish_12K

See [data/](../../data/) for dataset sources, sizes, and download links.
