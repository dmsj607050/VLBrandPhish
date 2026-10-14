# Experiment 3: Cross-Brand (Novel Brands, in_277 == 0)

Cross-brand phishing detection on **unseen** brands (104 brands with `in_277 == 0` in
`08_brands_overlap.csv`).

## Datasets

| Asset | Path |
|---|---|
| Phishing screenshots | `data/VLBrandPhish/01_phish_12K` -- novel 104 brands, 6,401 images |
| Benign screenshots | `data/VLBrandPhish/02_bengin_12K` -- 12,164 images |
| Logo reference library | `data/VLBrandPhish/03_logo_reference_library/logo_reference_library` -- restricted to 104 novel brands |
| Text prompt library | `data/VLBrandPhish/04_prompt_libraries_152/visual_all` -- exact match with logo reference, 104/104 |
| Normal URL mapping | `data/VLBrandPhish/07_normal_urls_testb.csv` |
| Brand split table | `data/VLBrandPhish/08_brands_overlap.csv` -- `in_277 == 0` |

## Hyperparameters

Identical to cross-time, fixed:

| Parameter | Value |
|---|---|
| brand_text_mode | dbi5 |
| image_weight (alpha) | 0.4 |
| text_weight | 0.6 |
| threshold | 0.4 |
| topk | 1 (highest-confidence detector box) |
| device | cuda:0 |

## Run

```bash
# Prepare manifest (validate config only, no inference)
python experiments/cross_brand/run.py --prepare-only --run-name smoke_manifest

# Full run (Stage 1 DINO detection + Stage 2 VLBrandPhish inference)
python experiments/cross_brand/run.py --run-name cross_brand_full
```

Output under `experiments/cross_brand/results/cross_brand_<run-name>/`.
