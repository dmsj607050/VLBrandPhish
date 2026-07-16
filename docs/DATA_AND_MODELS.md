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

| Purpose | Local path | Distribution recommendation |
| --- | --- | --- |
| Logo-detector checkpoint | `models/logo_detector.pth` | Model host or Zenodo release asset |
| Brand-recognizer checkpoint | `models/brand_recognizer.pt` | Model host or Zenodo release asset |
| Brand logo library | `data/brand_library/images/` | Archived dataset with license/provenance |
| Domain reference table | `data/brand_library/normal_urls.csv` | Versioned CSV with source/provenance |
| Evaluation screenshots | `data/inputs/` | Access-controlled archive or retrieval script |

When public links are available, replace this table's recommendation cells with the exact URL, version, SHA-256 checksum, and expected filename.
