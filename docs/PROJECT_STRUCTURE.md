# Project structure

`src/vlbrandphish/pipeline/` is the canonical inference implementation:

- `stage1_detect_top3.py` runs logo detection and emits a portable top-k CSV.
- `stage2_infer.py` crops detected logos, retrieves brand evidence, fuses image/text scores, and applies URL-domain verification.
- `contracts.py` defines the Stage-1/Stage-2 CSV contract.
- `brand_repo.py`, `models.py`, and `decision.py` contain brand-library, encoder, and decision logic.

`src/vlbrandphish/training/` is the canonical SigLIP brand-recognizer implementation. Its YAML configurations live in `configs/training/`, and `scripts/train.py` is the public entry point.

`training/logo_detector/` contains the project-specific wrappers used to configure and train the MMDetection/DINO logo detector. It intentionally does not vendor MMDetection or DINO: install those dependencies separately and cite their upstream projects.

All paper-specific results belong in a separate release archive or an `outputs/` artifact, not in the source repository. This keeps source history small and makes the exact version used for a paper easy to tag.
