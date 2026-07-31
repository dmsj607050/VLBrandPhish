# Datasets

This directory documents the datasets used in the paper. The datasets themselves are
**not committed** to this repository (too large, and some require a redistribution or
privacy review). Each entry below records its name, source, size, role, and where to
obtain it. Place downloaded data under `data/` following the local layout in
[docs/DATA_AND_MODELS.md](../docs/DATA_AND_MODELS.md) before running experiments.

See the release policy before publishing any screenshot, URL, or brand asset.

## Overview

| Dataset | Source | Approx. size | Role | Link |
| --- | --- | --- | --- | --- |
| PhishIntention | Public (PhishIntention + Phishpedia) | ~49 GB | End-to-end phishing detection; logo-detector training/test; brand reference library (277 brands) | [phishintention.md](phishintention.md) |
| VLBP-Set | Constructed in this work (OpenPhish + Tranco) | ~15 GB | Recent-webpage and unseen-brand evaluation (Full / Overlap / Novel); cropped-logo eval set | [vlbp_set.md](vlbp_set.md) |
| PhishEval | Public (PhishEval 2024) | ~21 GB | Adversarial robustness: visible logo manipulation (13 types) + perturbation attacks (6 types) | [phisheval.md](phisheval.md) |
| Phishpedia logo set | Public (Phishpedia) | ~46 GB | Supplementary logo-detection train/test combined with PhishIntention's logo split | [phishintention.md](phishintention.md#phishpedia-logo-detection-set) |

Sizes are measured on the local working copy and are approximate. Sample counts and
splits are authoritative in the paper and in each per-dataset file below.

## Local layout (after download)

```text
data/
├── phishintention/        # PhishIntention dataset + logo resources
├── vlbp_set/              # VLBP-Set (phishing, benign, reference library, crops)
├── phisheval/             # PhishEval adversarial samples
└── phishpedia_logo/       # Phishpedia logo-detection set
```

Each experiment under `experiments/` points at the specific split it consumes; nothing
here is used for training VLBrandPhish except the PhishIntention logo resources.
