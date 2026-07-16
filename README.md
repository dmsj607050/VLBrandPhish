# VLBrandPhish

Official code for **VLBrandPhish**, a visual-language, brand-aware phishing detection pipeline. The repository is prepared for the accompanying paper; the final title, author list, venue, and citation will be added when publication metadata is available.

## What is included

- A two-stage inference pipeline: logo detection followed by visual-language brand identification and URL-domain verification.
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
├── scripts/                 # Training entry points
├── src/vlbrandphish/
│   ├── pipeline/            # Stage-1 and Stage-2 inference implementation
│   └── training/            # SigLIP brand-recognizer implementation
├── training/logo_detector/  # MMDetection/DINO detector training wrappers
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

For Stage-1 or logo-detector training, install a compatible MMDetection stack following the OpenMMLab instructions, then run `pip install -r requirements-detector.txt`.

## Quick start

Place data and checkpoints according to [Data and model release policy](docs/DATA_AND_MODELS.md). All default paths are relative to the repository root.

```bash
# Stage 1: generate top-3 logo detections.
python -m vlbrandphish.pipeline.stage1_detect_top3 \\
  --det-config /path/to/mmdetection_config.py \\
  --det-checkpoint models/logo_detector.pth \\
  --progress

# Stage 2: identify the brand and make a phishing decision.
python -m vlbrandphish.pipeline.stage2_infer \\
  --det-csv outputs/stage1/logo_detection_top3.csv \\
  --brand-ckpt models/brand_recognizer.pt \\
  --progress

# Train the brand recognizer.
python scripts/train.py --config configs/training/stage1_siglip.yaml
```

The Stage-2 command accepts explicit paths for every dataset and model input. Run `python -m vlbrandphish.pipeline.stage2_infer --help` for the complete interface.

## Reproducibility and release

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data and model release policy](docs/DATA_AND_MODELS.md)
- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Pre-publication release checklist](docs/RELEASE_CHECKLIST.md)

## License

Released under the [MIT License](LICENSE). Please cite the paper once the final bibliographic information is added.
