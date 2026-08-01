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
|-- data/                            # Dataset documentation (actual data hosted externally)
|-- docs/                            # Reproducibility and release documentation
|-- models/                          # Model checkpoints, one subdir per method (ignored)
|-- scripts/                         # Manifest, verification, and metadata utilities
|-- inference/                       # Per-method inference pipelines
|   `-- vlbrandphish/                # VLBrandPhish end-to-end pipeline
|       |-- components/brand/        # Shared encoder, fusion, brand repository, naming
|       |-- logo_detection.py        # Logo detection command
|       |-- model_loading.py         # Encoder and brand-repository loading
|       |-- brand_recognition.py     # Logo-crop encoding and evidence fusion
|       |-- domain_verification.py   # Domain-brand consistency checks
|       |-- evaluation.py            # Metrics and reports
|       `-- pipeline.py              # End-to-end inference command
|-- training/                        # Training code (VLBrandPhish only)
|   `-- vlbrandphish/
|       |-- logo_detection/          # MMDetection/DINO logo-detector training
|       `-- brand_recognition/       # Brand-recognizer training
|           |-- common/              # Shared data/model/loss/optimizer code
|           |-- configs/             # Training configurations and prompt templates
|           |-- stage1/              # Pairwise SigLIP stage
|           `-- stage2/              # Fusion/TriSig stage
|-- experiments/                     # Per-experiment thin runners (call inference + models)
|   |-- overall_phishintention/      # RQ1: overall detection on PhishIntention dataset
|   |-- overall_vlbp/                # RQ1: overall detection on VLBP-Full
|   |-- logo_detection/              # RQ2: logo localization (Default/Merged/New)
|   |-- brand_identification/        # RQ2: brand matching on cropped-logo set
|   |-- cross_time/                  # RQ3: cross-time (VLBP-Overlap)
|   |-- cross_brand/                 # RQ3: cross-brand (VLBP-Novel)
|   |-- adversarial_robustness/      # RQ4: logo manipulation + perturbation attacks
|   |-- ablations/                   # Branch / prompt / ref-library-size / fusion-weight
|   |-- analysis/                    # Inference diagnostics and protocols (supporting)
|   |-- figures/                     # Case-study and figure generation (supporting)
|   `-- preparation/                 # Dataset/reference-library preparation (incl. VOC annotation tools)
`-- tests/                           # Lightweight, data-free checks
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
```

The repository uses a flat layout: run commands from the repository root so that
`inference/`, `training/`, and `experiments/` are importable. There is
no installable package; `pip install -e .` is not required.

For logo detection or detector training, install a compatible MMDetection stack following the OpenMMLab instructions, then run `pip install -r requirements-detector.txt`.

## Quick start

Place data and checkpoints according to [Data and model release policy](docs/DATA_AND_MODELS.md). All default paths are relative to the repository root.

```bash
# Generate top-k logo detections.
python cli.py detect \\
  --det-config /path/to/mmdetection_config.py \\
  --det-checkpoint models/vlbrandphish/logo_detector.pth \\
  --progress

# Identify the brand and make the phishing decision.
python cli.py infer \\
  --det-csv outputs/detection/logo_detection_top3.csv \\
  --brand-ckpt models/vlbrandphish/brand_recognizer.pt \\
  --progress

# Train the brand recognizer.
python cli.py train-brand --config training/vlbrandphish/brand_recognition/configs/stage1_siglip.yaml
```

The `infer` command accepts explicit paths for every dataset and model input. Run `python cli.py infer --help` for the complete interface. The legacy wrapper `python scripts/train_brand.py ...` remains available for existing training jobs.

## Reproducibility and release

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data and model release policy](docs/DATA_AND_MODELS.md)
- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Architecture review and design decisions](docs/ARCHITECTURE.md)
- [Pre-publication release checklist](docs/RELEASE_CHECKLIST.md)
- [USENIX Security '27 readiness and anonymous-artifact guide](docs/USENIX_SECURITY_2027_READINESS.md)

## License

Released under the [MIT License](LICENSE). Please cite the paper once the final bibliographic information is added.
