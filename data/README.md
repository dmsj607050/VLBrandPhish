# Local data layout

This directory is ignored by Git. Do not commit phishing screenshots, captured URLs, browser artifacts, annotations with personal data, or third-party brand assets without verifying that redistribution is permitted.

Expected local layout for the default inference commands:

```text
data/
├── inputs/
│   ├── benign/<sample_id>/<sample_id>.png and <sample_id>.json
│   └── phish/<brand>/{screens,jsons}/
└── brand_library/
    ├── images/<brand>/*.png
    ├── DBI-5/
    └── normal_urls.csv
```

See [the release policy](../docs/DATA_AND_MODELS.md) for the required access and provenance notes.
