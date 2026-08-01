# Cross-brand stability

| | |
| --- | --- |
| Research question | RQ3 |
| Dataset | VLBP-Novel (6,401 phishing, 104 brands unseen in training) + 12,187 benign |
| Metrics | Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc |
| Paper table(s) | Table: cn-cross-brand |
| Methods | vlbrandphish, phishintention, phishpedia |

## Run

```bash
python -m experiments.cross_brand.run \
  --screenshot-dir <screenshots> \
  --url-list <url manifest> \
  --out-dir outputs/cross_brand
```

Each method's predictions land in ``outputs/cross_brand/<method>/``. The runner dispatches to
``inference/<method>/`` and loads weights from ``models/<method>/`` (see
``experiments/_common.py``). Baselines require ``detectron2`` at runtime.

## Expected data

data/vlbp_set/01_phish_12K (novel split), data/vlbp_set/02_bengin_12K

See [data/](../../data/) for dataset sources, sizes, and download links.
