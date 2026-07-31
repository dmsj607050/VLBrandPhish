# PhishIntention dataset

Public dataset for logo-based visual phishing detection. Combines benign webpages from
Phishpedia with credential-requiring-page (CRP) phishing webpages from PhishIntention.

- **Source:** PhishIntention and Phishpedia official releases
  - Code/data: https://github.com/lindsey98/PhishIntention
  - Code/data: https://github.com/lindsey98/phishpedia
- **Local copy:** `data/phishintention/` (~49 GB, 313k files)
- **License / redistribution:** Verify upstream terms before redistributing screenshots
  or URLs. See [release policy](../docs/DATA_AND_MODELS.md).

## Statistics (after cleaning)

| Split | Brands | Phishing / squatted | Benign |
| --- | ---:| ---:| ---:|
| Full | 277 | 24,532 | 21,974 |

Cleaning removed 871 phishing samples whose brand labels did not match page content and
3,426 benign samples missing URL information (the domain-verification stage requires a URL).

## Contents

- Webpage screenshots + URLs (used for logo detection and domain verification).
- Target-brand labels for phishing samples (used to verify brand identification).
- **Logo target list:** 3,061 logo samples across 277 brands — used to build the brand
  reference library and to train/evaluate the brand-identification module.
- **Logo detection split:** 8,109 training images and 901 test images with logo bounding
  boxes (derived from 9,010 annotated phishing screenshots).

## Role in the paper

- **RQ1 (Overall):** end-to-end phishing detection on the official PhishIntention test set
  (Table in §6.2).
- **RQ2 (Logo detection):** the 901-image test set is the "Default" split; combined with
  the Phishpedia test set it forms the "Merged" split (2,480 images).
- **Training:** all VLBrandPhish model components are trained using PhishIntention's logo
  and brand resources. VLBP-Set is never used for training.

## Phishpedia logo-detection set

A separate public logo-detection release from Phishpedia (`coco_train.json`,
`coco_test.json`, screenshots). Locally `data/phishpedia_logo/` (~46 GB). In this work it
is only used as an additional logo-detection test set, merged with PhishIntention's
901-image test set to form the 2,480-image "Merged" split for RQ2. It is not used for
phishing-detection evaluation on its own.

## Local layout

```text
data/phishintention/
├── 277_brand_pages/        # brand reference library (logos per brand)
├── Logo_train/             # logo-detection train/test annotations + images
├── benign_25k/             # benign webpage screenshots + URLs
├── phish_sample_30k/       # phishing webpage screenshots + URLs
├── normal_urls.csv         # brand -> official domain map
└── phish30k_nonCRP.txt
```
