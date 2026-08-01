# Overall detection on the PhishIntention dataset

| | |
| --- | --- |
| Research question | RQ1 |
| Dataset | PhishIntention dataset (24,532 phishing + 21,974 benign, 277 brands) |
| Metrics | Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc |
| Paper table(s) | Table: cn-e2e-testa |
| Methods | vlbrandphish, phishintention, phishpedia |

## Run

```bash
python -m experiments.overall_phishintention.run \
  --screenshot-dir <screenshots> \
  --url-list <url manifest> \
  --out-dir outputs/overall_phishintention
```

Each method's predictions land in ``outputs/overall_phishintention/<method>/``. The runner dispatches to
``inference/<method>/`` and loads weights from ``models/<method>/`` (see
``experiments/_common.py``). Baselines require ``detectron2`` at runtime.

## Expected data

data/phishintention/phish_sample_30k, data/phishintention/benign_25k

See [data/](../../data/) for dataset sources, sizes, and download links.
