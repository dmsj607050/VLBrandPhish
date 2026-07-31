# VLBrandPhish

Official code for **VLBrandPhish**, a visual-language, brand-aware phishing-detection system. The repository is prepared for the accompanying paper; the final title, author list, venue, and citation will be added when publication metadata is available.

## What is included

- A detection-and-verification workflow: logo detection, visual-language brand identification, and URL-domain verification.
- SigLIP-based brand-recognizer training code with image-to-image and image-to-text evidence fusion.
- Logo-detector training wrappers built on MMDetection/DINO.
- Reproducible command-line entry points, configuration files, unit tests, and a GitHub Actions syntax check.

Datasets, screenshots, trained weights, caches, and experimental run outputs are deliberately **not** tracked. See [Data and model release policy](docs/DATA_AND_MODELS.md) before adding any of them.

## Repository layout

```text
VLBrandPhish-main/
├── configs/                 # Training configurations and prompt templates
├── data/                    # Local-only datasets and brand library (ignored)
├── docs/                    # Reproducibility and release documentation
├── models/                  # Local-only checkpoints (ignored)
├── scripts/                 # Thin backwards-compatible command wrappers
├── src/vlbrandphish/
│   ├── config/              # Portable repository paths and configuration
│   ├── data/                # Dataset records and CSV contracts
│   ├── detection/           # Logo detection runtime and MMDetection training
│   ├── recognition/         # Encoders, fusion modules, and brand repository
│   ├── decision/            # Domain parsing and phishing decision policy
│   ├── inference/           # End-to-end prediction runtime
│   ├── training/brand/      # SigLIP brand-recognizer training
│   ├── experiments/         # Ablations, analysis, figures, and preparation
│   └── tools/annotation/    # Dataset annotation utilities
└── tests/                   # Lightweight, data-free checks
```

## Installation

Python 3.10 or 3.11 is recommended. Install a PyTorch build matching your GPU/CPU first, then:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VLBrandPhish-main
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For logo detection or detector training, install a compatible MMDetection stack following the OpenMMLab instructions, then run `pip install -r requirements-detector.txt`.

## Quick start

Place data and checkpoints according to [Data and model release policy](docs/DATA_AND_MODELS.md). All default paths are relative to the repository root.

```bash
# Generate top-k logo detections.
vlbrandphish detect \\
  --det-config /path/to/mmdetection_config.py \\
  --det-checkpoint models/logo_detector.pth \\
  --progress

# Identify the brand and make the phishing decision.
vlbrandphish infer \\
  --det-csv outputs/detection/logo_detection_top3.csv \\
  --brand-ckpt models/brand_recognizer.pt \\
  --progress

# Train the brand recognizer.
vlbrandphish train-brand --config configs/training/stage1_siglip.yaml
```

The `infer` command accepts explicit paths for every dataset and model input. Run `vlbrandphish infer --help` for the complete interface. The legacy wrapper `python scripts/train_brand.py ...` remains available for existing training jobs.

## Reproducibility and release

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data and model release policy](docs/DATA_AND_MODELS.md)
- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Architecture review and design decisions](docs/ARCHITECTURE.md)
- [Pre-publication release checklist](docs/RELEASE_CHECKLIST.md)
- [USENIX Security '27 readiness and anonymous-artifact guide](docs/USENIX_SECURITY_2027_READINESS.md)

## License

Released under the [MIT License](LICENSE). Please cite the paper once the final bibliographic information is added.
