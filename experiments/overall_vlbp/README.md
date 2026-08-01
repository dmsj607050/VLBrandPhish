# Overall detection on VLBP-Full

| | |
| --- | --- |
| Research question | RQ1 |
| Dataset | VLBP-Full (12,801 phishing + 12,187 benign, 152 brands) |
| Metrics | Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc |
| Paper table(s) | Table: cn-e2e-main |
| Methods | vlbrandphish, phishintention, phishpedia |

## Run

```bash
python -m experiments.overall_vlbp.run \
  --screenshot-dir <screenshots> \
  --url-list <url manifest> \
  --out-dir outputs/overall_vlbp
```

Each method's predictions land in ``outputs/overall_vlbp/<method>/``. The runner dispatches to
``inference/<method>/`` and loads weights from ``models/<method>/`` (see
``experiments/_common.py``). Baselines require ``detectron2`` at runtime.

## Expected data

data/vlbp_set/01_phish_12K, data/vlbp_set/02_bengin_12K

See [data/](../../data/) for dataset sources, sizes, and download links.
