# Overall detection on the PhishIntention dataset (RQ1)

| | |
| --- | --- |
| Research question | RQ1 -- overall VLBrandPhish end-to-end detection on the PhishIntention public dataset |
| Dataset | Raw: 29,496 phishing + 25,400 benign directories. After filtering: **24,532 phishing + 21,973 benign = 46,505** samples, 183 targeted brands |
| Metrics | Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc |
| Pipeline | Full Stage 1 (logo detection) + Stage 2 (brand recognition + URL decision) |

## Asset defaults (baked into `run.py`)

| Asset | Path |
| --- | --- |
| Raw screenshots | `data/PhishIntention/test/{phish_sample_30k,benign_25k}` |
| 277-brand logo library | `data/PhishIntention/277_brand_pages` |
| Brand-to-domain map | `data/PhishIntention/normal_urls.csv` |
| CRP exclusion list | `data/PhishIntention/phish30k_nonCRP.txt` |
| Brand-text mode | `ckpt_text_proto_mean` (checkpoint-shipped prototypes, no DBI-5 required) |

## Filtering chain (exact replica of PhishIntention `evaluation_30k/cli.py` + `dataset.py`)

1. **Basic load** -- keep directories with `shot.png` + `info.txt` + non-empty URL.
2. **`phish30k_nonCRP.txt` exclusion** -- remove 4,093 non-credential-requiring phishing
   sample keys.
3. **`aligned_phish_brands_only`** -- via `_normalize_brand_key` + `normal_urls.csv`
   `phish_brand -> canonical` lookup; drop 871 samples whose brand does not map to a
   277 reference brand with a non-empty domain list.

Final: 24,532 phishing + 21,973 benign = **46,505**.

*(Note: the PhishIntention paper reports 46,506 total; the 1-sample difference is `hotmailiniciarsesionentrada.com`, a benign entry whose `info.txt` contains only NUL bytes and is therefore unusable.)*

## Run

**Smoke test (manifest generation only; no GPU):**

```
python experiments/overall_phishintention/run.py --prepare-only --run-name smoke
```

**Full experiment (Stage 1 detection + Stage 2 inference + metrics):**

```
python experiments/overall_phishintention/run.py --run-name overall_pi_full
```

Outputs under `experiments/overall_phishintention/results/overall_phishintention_<run-name>/`:

| File | Content |
| --- | --- |
| `input_manifest.csv` | Frozen sample list with ground-truth labels |
| `stage1/logo_detection_top3.csv` | Logo detection results (DINO detector) |
| `stage2/predictions.csv` | Brand recognition + phishing decisions |
| `experiment_summary.json` | Metrics, protocol, and asset hashes |
| `tables/*.csv` | Per-split metric tables |
