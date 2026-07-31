# Data and model release policy

## Not included in this repository

The working directory contains captured phishing pages, benign pages, reference-logo images, experimental caches, and large checkpoints. They are excluded from this release because they may be large, redistributable only under third-party terms, or contain URLs and screenshots that need a security/privacy review.

## Before publishing a dataset

For each dataset or brand-library asset, record:

1. Source URL, collection date, and version.
2. License or permission for redistribution.
3. Any takedown, access-control, or malware-handling requirements.
4. Annotation schema and train/validation/test split procedure.
5. A checksum and permanent download location.

Do not upload browser profiles, cookies, tokens, raw crawler logs, credentials, or private URLs. If an original screenshot cannot be redistributed, publish a stable sample identifier and a script that retrieves it from an allowed source instead.

## Expected files for inference

Datasets are documented under [`data/`](../data/) (name, source, size, role, link) and are
not committed. Model weights live under `models/<method>/` (see
[`models/README.md`](../models/README.md)).

| Purpose | Local path | Distribution recommendation |
| --- | --- | --- |
| VLBrandPhish logo detector | `models/vlbrandphish/logo_detector.pth` | Model host or Zenodo release asset |
| VLBrandPhish brand recognizer | `models/vlbrandphish/brand_recognizer.pt` | Model host or Zenodo release asset |
| PhishIntention weights | `models/phishintention/*` | Official release + provenance |
| Phishpedia weights | `models/phishpedia/*` | Official release + provenance |
| Brand logo library | `data/phishintention/277_brand_pages/` | Archived dataset with license/provenance |
| Domain reference table | `data/phishintention/normal_urls.csv` | Versioned CSV with source/provenance |
| Evaluation screenshots | `data/phishintention/`, `data/vlbp_set/` | Access-controlled archive or retrieval script |
| Adversarial samples | `data/phisheval/` | Access-controlled archive |

When public links are available, replace this table's recommendation cells with the exact URL, version, SHA-256 checksum, and expected filename.

## Anonymous-review access

For a double-blind submission, do not use this public GitHub repository as the review artifact. The artifact mirror must have no Git history, usernames, organization names, tracked analytics, or public URLs that identify the authors. Provide an anonymous access link in the paper's Open Science appendix, enumerate what is accessible, and explain every legal, ethical, privacy, or security restriction on data or checkpoints. See [USENIX Security '27 readiness](USENIX_SECURITY_2027_READINESS.md).
