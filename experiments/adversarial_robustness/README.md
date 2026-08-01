# Adversarial robustness under logo manipulation and perturbation

| | |
| --- | --- |
| Research question | RQ4 |
| Dataset | PhishEval: visible manipulation (13 types, 88,152 phish) + perturbation (6 types, 6,528 phish) + original (1,320) |
| Metrics | Phishing detection rate on squatted-domain samples; benign alert rate; embedding intra-brand distance & separation ratio (113-sample subset) |
| Paper table(s) | Tables: cn-adversarial-summary, cn-adversarial-embedding-distance |
| Methods | vlbrandphish, phishintention, phishpedia |

## Run

```bash
python -m experiments.adversarial_robustness.run \
  --screenshot-dir <screenshots> \
  --url-list <url manifest> \
  --out-dir outputs/adversarial_robustness
```

Each method's predictions land in ``outputs/adversarial_robustness/<method>/``. The runner dispatches to
``inference/<method>/`` and loads weights from ``models/<method>/`` (see
``experiments/_common.py``). Baselines require ``detectron2`` at runtime.

## Expected data

data/phisheval/

See [data/](../../data/) for dataset sources, sizes, and download links.
