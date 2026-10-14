# Experiment 1: Full-dataset phishing detection

| | |
| --- | --- |
| Research question | RQ1 -- overall end-to-end phishing detection on the VLBP official dataset |
| Dataset | All samples under `01_phish_12K` and `02_bengin_12K` (152 brands, no filtering) |
| Pipeline | Stage 1 (DINO logo detection) + Stage 2 (SigLIP brand retrieval + URL decision) |
| Fixed decision | image_weight=0.4, text_weight=0.6, threshold=0.4 |
| Expected counts | **12,801 phishing + 12,164 benign = 24,965** (23 benign dirs lack `shot.png`) |

## Defaults (baked into `run.py`)

| Parameter | Value |
| --- | --- |
| `dataset_root` | `data/VLBrandPhish` (or set `VLBRANDPHISH_DATASET_ROOT` env var) |
| `dbi5_dir` | `<dataset_root>/04_prompt_libraries_152/visual_all` |
| `brand_text_mode` | `dbi5` |

## Run

**Smoke test (manifest only, no GPU):**

```
python experiments/overall_vlbp/run.py --prepare-only --run-name smoke
```

**Full experiment:**

```
python experiments/overall_vlbp/run.py --run-name overall_full
```

**Reuse predictions from another full run (e.g. for cross_brand subsets):**

```
python experiments/cross_brand/run.py --from-overall-run results/overall_vlbp_<name>
```

Outputs under `experiments/overall_vlbp/results/overall_vlbp_<run-name>/`.
