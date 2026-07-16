# Project structure

The package is organized by responsibility rather than by experiment order:

| Area | Responsibility | Public entry point |
| --- | --- | --- |
| `config/` | Repository-relative defaults and portable path resolution | imported by commands |
| `data/` | Test-B records and the detection CSV contract | imported by detection/inference |
| `detection/` | Top-k detector runtime and its MMDetection training adapter | `vlbrandphish detect` / `train-detector` |
| `recognition/` | SigLIP encoders, shared fusion layers, and brand evidence repository | used by inference and brand training |
| `decision/` | URL-domain extraction and final decision policy | used by inference |
| `inference/` | End-to-end brand matching, caching, grid evaluation, and reports | `vlbrandphish infer` |
| `training/brand/` | Brand-recognizer datasets, model, transforms, and trainer | `vlbrandphish train-brand` |
| `experiments/` | Ablations, analyses, paper figures, and data-preparation studies | module-specific commands |
| `tools/annotation/` | VOC/XML annotation maintenance utilities | module-specific commands |

`scripts/train_brand.py` is deliberately a thin compatibility wrapper, not a second implementation. The detector wrappers live inside `src/vlbrandphish/detection/training/`; they configure MMDetection/DINO but do not vendor either upstream project.

All paper-specific outputs, model weights, screenshots, and caches stay outside Git. This keeps source history compact and lets a tagged source revision map cleanly to a separately archived artifact bundle.
