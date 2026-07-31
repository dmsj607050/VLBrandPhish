# VLBP-Set

Dataset constructed in this work to extend the temporal and brand coverage of existing
phishing-detection benchmarks. Enables evaluation on recent webpages and on brands unseen
during training. **Used exclusively for evaluation, never for training.**

- **Source:**
  - Phishing webpages: OpenPhish (https://openphish.com), collected continuously from
    September 2025 to February 2026.
  - Benign webpages: Tranco list generated 8 February 2026
    (https://tranco-list.eu/list/QW3J4), visited over the following ~10 days.
  - Brand reference library: official brand websites + public webpages, with manually
    compiled DBI-4 text descriptions.
- **Local copy:** `data/vlbp_set/` (~15 GB, 81k files)
- **Redistribution:** screenshots and URLs require a security/privacy review before public
  release. See [release policy](../docs/DATA_AND_MODELS.md). A versioned release is
  planned alongside the paper (host TBD).

## Statistics

| Split | Brands | Phishing | Benign |
| --- | ---:| ---:| ---:|
| VLBP-Full | 152 | 12,801 | 12,187 |
| VLBP-Overlap | 48 | 6,400 | 12,187 |
| VLBP-Novel | 104 | 6,401 | 12,187 |

The brand distribution is long-tailed: the top brand (Roblox) has 1,742 phishing pages
while the median across brands is 13; 47 brands have ≤5 samples.

## Evaluation splits

- **VLBP-Full:** all 12,801 phishing + 12,187 benign — overall detection (RQ1).
- **VLBP-Overlap:** 48 brands shared with PhishIntention (seen during training), recent
  webpages — cross-time stability (RQ3).
- **VLBP-Novel:** 104 brands unseen during training; newly added brands are inserted into
  the protected-brand pool using their text descriptions and reference logos — cross-brand
  stability (RQ3).

## Cropped-logo evaluation set

To isolate brand identification from logo-localization errors, a unified cropped-logo set
was built from manual bounding-box annotations: **2,700 logo crops** (1,352 phishing +
1,348 benign, 110 brands). Used for RQ2 brand-identification analysis and the prompt
ablation.

## Local layout

```text
data/vlbp_set/
├── 01_phish_12K/                          # 12,801 phishing webpages (by brand)
├── 02_bengin_12K/                         # 12,187 benign webpages
├── 03_logo_reference_library/             # 152-brand reference library (logos + DBI-4 text)
├── 04_normal_urls_testb.csv               # brand -> official domain map
├── 05_brands_overlap.csv                  # 48 overlap brands (PhishIntention ∩ VLBP)
├── 06_generated_prompt_libraries/         # text-prototype prompt libraries
├── 07_bengin_12K_logo_xml/                # manual logo bounding-box annotations (VOC XML)
└── 08_open_set_logo_crops_1352_seed20260501/  # cropped-logo evaluation set
```
