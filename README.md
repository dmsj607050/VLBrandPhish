# VLBrandPhish

Official code for **VLBrandPhish**, a visual-language, brand-aware phishing-detection system.

The method works in two stages: (1) a DINO-based logo detector localizes brand logos in webpage screenshots, and (2) the current SigLIP brand recognizer combines image and text evidence with weights 0.4 and 0.6, then checks the predicted brand against the page URL using a fixed 0.4 decision threshold. The current recognizer is trained with the pairwise objective, four unfrozen text blocks, two unfrozen vision blocks, and checkpoint-saved brand text prototypes.

---

## Repository structure

```
VLBrandPhish-main/
|-- .github/workflows/ci.yml       # Syntax, path, and artifact-hygiene checks
|-- inference/                     # Flat inference package
|   |-- components/                # Brand encoders and repositories
|   |-- data/                      # Detection CSV and image-path helpers
|   |-- decision/                  # Domain-aware phishing decision policy
|   |-- detection/                 # Stage 1 DINO detector command
|   |-- evaluation/                # Metrics
|   |-- recognition/               # Stage 2 feature extraction and fusion
|   |-- runtime/                   # Default paths, caching, and diagnostics
|   `-- run.py                     # Stage 2 end-to-end CLI
|-- model_train/
|   |-- Brand/                     # Current SigLIP brand-recognition training source
|   `-- Logo/                      # Current DINO/Swin localization training source
|-- baselines/PDOS/                # Project's PDOS reproduction; not official author code
|-- training/
|   |-- brand_recognition/         # Earlier/general training implementation
|   `-- logo_detection/            # Earlier detector-training implementation
|-- experiments/                   # Paper experiment runners
|-- tests/test_smoke.py            # Reproducibility environment smoke test
|-- data/                          # External datasets only; contents are excluded from Git
|-- models/                        # External model files only; contents are excluded from Git
|-- requirements.txt
|-- requirements-detector.txt
|-- pyproject.toml
|-- README.md
`-- LICENSE
```
---

## Installation

Requires **Python 3.10+**. Install PyTorch first (matching your GPU/CUDA), then:

```bash
git clone <REPO_URL>
cd VLBrandPhish-main
pip install -r requirements.txt
```

For logo detection (Stage 1) or detector training, additionally install a compatible MMDetection stack, then:

```bash
pip install -r requirements-detector.txt
```

The repository uses a flat source layout — no `pip install -e .` needed. Run all commands from the repository root.

---

## Datasets

Datasets are **not included** in this repository (too large; some require redistribution review). Download each dataset separately and place it in the expected path.

| Dataset | Size | Content | Expected path |
|---|---|---|---|
| **VLBP-Set** | ~15 GB | 12,801 phishing + 12,187 benign screenshots (152 brands); logo reference library; DBI text prompts; cropped-logo eval set | `data/VLBrandPhish/` |
| **PhishIntention** | ~49 GB | 24,532 phishing + 21,974 benign; 277-brand logo library; logo-detection annotations | `data/PhishIntention/` |
| **PhishEval** | ~21 GB | Adversarial samples: 13 visible manipulations + 6 perturbation attacks | `data/PhishEval/` |
| **PhishPedia logo** | ~46 GB | Additional logo-detection test set (merged with PhishIntention's) | `data/Phishpedia/` |

### Expected VLBP-Set local layout

The following is the expected layout after downloading the external dataset. Dataset contents are intentionally not stored in this Git repository.

```
data/VLBrandPhish/
|-- 01_phish_12K/                  # Phishing screenshots organized by brand
|-- 02_bengin_12K/                 # Benign screenshots
|-- 03_logo_reference_library/     # Reference logo images per brand
|-- 04_prompt_libraries_152/       # Earlier prompt collection
|-- 05_logo_crops/crops/           # Cropped-logo evaluation set
|-- 06_prompt_library_277/         # 277-brand prompt library
|-- 07_normal_urls_testb.csv       # Brand-to-official-domain mapping
`-- 08_brands_overlap.csv          # Seen/novel brand split table
```
**Download:** TODO — add public download URL, version, and SHA-256 checksum.

### PhishIntention local layout

```
data/PhishIntention/
├── 277_brand_pages/                   # Brand reference library
├── logo_train/                        # Logo-detection train/test annotations (some releases use Logo_train)
├── benign_25k/                        # Benign webpage screenshots + URLs
├── phish_sample_30k/                  # Phishing webpage screenshots + URLs
├── normal_urls.csv                    # Brand → official domain map
└── phish30k_nonCRP.txt
```

**Download:** TODO — add public download URL, version, and SHA-256 checksum.

### PhishEval local layout

```
data/PhishEval/
├── perturbated_dataset/               # 6 perturbation attack types
│   ├── SRNet/
│   ├── Swin/
│   ├── ViT/
│   ├── FGSM/
│   ├── PGD/
│   └── CW/
├── crawl_benign/                      # Benign samples
└── ...                                # Visible manipulation samples
```

**Download:** TODO — add public download URL, version, and SHA-256 checksum.

---

## Models

Model weights are **not included** in this repository. Place them under `models/` as shown below.

### VLBrandPhish

| File | Purpose | Expected path |
|---|---|---|
| `detector.pth` | DINO logo detector (MMDetection) | `models/vlbrandphish/detector.pth` |
| `brand_recognition_siglip_text4_vision2_full277.pt` | Current SigLIP brand recognizer | `models/vlbrandphish/brand_recognition_siglip_text4_vision2_full277.pt` |
| `patched_config.py` | DINO detector configuration | `models/vlbrandphish/patched_config.py` |
| SigLIP base model | Hugging Face SigLIP files (config.json, etc.) | `models/vlbrandphish/siglip-base-patch16-224/` |

Model weights are not stored in Git. The current brand-recognition checkpoint named above must be supplied from the current training run; the older `brand_identify.pt` release does not represent the current text4/vision2 training configuration. The detector and its config can be placed under `models/vlbrandphish/`:

```bash
mkdir -p models/vlbrandphish
curl -L https://huggingface.co/VLBrandPhish/vlbrandphish/resolve/main/detector.pth?download=true -o models/vlbrandphish/detector.pth
curl -L https://huggingface.co/VLBrandPhish/vlbrandphish/resolve/main/patched_config.py?download=true -o models/vlbrandphish/patched_config.py
```

Direct downloads: [`detector.pth`](https://huggingface.co/VLBrandPhish/vlbrandphish/resolve/main/detector.pth?download=true) and [`patched_config.py`](https://huggingface.co/VLBrandPhish/vlbrandphish/resolve/main/patched_config.py?download=true). The local SigLIP base model is required for feature encoding.
### PhishIntention (baseline)

Official release at https://github.com/lindsey98/PhishIntention. Expected under `models/phishintention/`:

| File | Purpose |
|---|---|
| `layout_detector.pth` | AWL layout/logo detector |
| `crp_classifier.pth.tar` | Credential-requiring page classifier |
| `crp_locator.pth` | CRP element locator |
| `ocr_siamese.pth.tar` | Siamese logo matcher |
| `ocr_pretrained.pth.tar` | OCR encoder pretrained weights |
| `domain_map.pkl` | Brand → official domain map |

### PhishPedia (baseline)

Official release at https://github.com/lindsey98/phishpedia. Expected under `models/phishpedia/`:

| File | Purpose |
|---|---|
| `rcnn_bet365.pth` | Faster R-CNN logo detector |
| `resnetv2_rgb_new.pth.tar` | Siamese logo matcher |
| `domain_map.pkl` | Brand → official domain map |

**Download:** TODO — add public download URLs, versions, and SHA-256 checksums.

---

## Inference workflow

The system operates in two explicit stages. All commands are run from the repository root using the dedicated `run.py` scripts.

### Stage 1: Logo detection

Runs the DINO detector on webpage screenshots, retaining the top-3 logo bounding boxes per image. Entry point: `inference/detection/command.py`.

```bash
python inference/detection/command.py \
  --det-config models/vlbrandphish/patched_config.py \
  --det-checkpoint models/vlbrandphish/detector.pth \
  --benign-root data/VLBrandPhish/02_bengin_12K \
  --phish-root data/VLBrandPhish/01_phish_12K \
  --out-csv outputs/detection/logo_detection_top3.csv \
  --device cuda:0
```

**Output:** `outputs/detection/logo_detection_top3.csv` — one row per sample, with up to 3 bounding boxes.

### Stage 2: Brand recognition + phishing decision

Consumes the detection CSV, identifies brands using the SigLIP encoder, and makes the phishing decision with domain verification. Entry point: `inference/run.py`.

```bash
python inference/run.py \
  --det-csv outputs/detection/logo_detection_top3.csv \
  --brand-ckpt models/vlbrandphish/brand_recognition_siglip_text4_vision2_full277.pt \
  --model-dir models/vlbrandphish/siglip-base-patch16-224 \
  --logo-library-dir data/VLBrandPhish/03_logo_reference_library/logo_reference_library \
  --dbi5-dir data/VLBrandPhish/06_prompt_library_277 \
  --normal-urls-csv data/VLBrandPhish/07_normal_urls_testb.csv \
  --benign-root data/VLBrandPhish/02_bengin_12K \
  --phish-root data/VLBrandPhish/01_phish_12K \
  --brand-text-mode ckpt_text_proto_mean \
  --device cuda:0 \
  --out-dir outputs/inference_run
```

**Output:** `predictions.csv`, `metrics.json`, `summary.json`, `brand_level_diagnosis.csv`, and cache files.

### Fixed decision rule

The fusion and decision are **fixed** (no parameter search):

| Parameter | Value |
|---|---|
| Image weight | 0.4 |
| Text weight | 0.6 |
| Decision threshold | 0.4 |
| Detector candidates | Top-1 (highest-confidence box) |

A sample is predicted as phishing only when a logo is detected, the fused score is at least 0.4, and the page domain does not match the predicted brand's official domain.

### Data flow

```
webpage screenshots + URLs
        │
        ▼
Stage 1: detection/command.py    ← DINO detector
        │
        ▼
logo_detection_top3.csv
        │
        ▼
Stage 2: run.py                  ← SigLIP encoder + brand library + domain verification
        │
        ▼
predictions.csv + metrics.json + summary.json
```

---

## Reproducing experiments

The overall and robustness experiments are manifest-driven and use the current fixed image/text weights (0.4/0.6) and threshold (0.4). Baseline-specific thresholds are documented in their experiment READMEs.

### RQ1 — Overall phishing detection

**VLBP-Full (all 152 brands):**

```bash
python experiments/overall_vlbp/run.py
```

**PhishIntention dataset (277 brands):**

```bash
python experiments/overall_phishintention/run.py
```

### RQ2 — Logo detection & brand identification

**Logo detection evaluation:**

```bash
python experiments/logo_detection/run.py
```

**Brand identification (open-set cropped logos):**

```bash
python experiments/brand_identification/run.py
```

### RQ3 — Cross-time & cross-brand stability

**Cross-time (48 seen brands, recent webpages):**

```bash
python experiments/cross_time/run.py
```

**Cross-brand (104 novel brands, never seen in training):**

```bash
python experiments/cross_brand/run.py
```

### RQ4 — Adversarial robustness

```bash
python experiments/adversarial_robustness/run.py
```

### Ablations

```bash
# Branch ablation (image-only / fusion / text-only)
python experiments/ablations/branch_ablation/run.py

# Prompt configuration ablation (6 prompt modes)
python experiments/ablations/prompt_ablation/run.py

# Reference-library size: 4x4 VLBrandPhish training/inference caps + PDOS
python experiments/ablations/ref_library_size/train_variants.py --run-id refcap_text4_vision2_run01
python experiments/ablations/ref_library_size/run.py --run-id refcap_eval_run01 --train-run-id refcap_text4_vision2_run01
```

### Common options

All experiment runners accept:

| Option | Description |
|---|---|
| `--dataset-root` | Path to dataset root |
| `--out-dir` | Explicit output directory |
| `--run-name` | Suffix for timestamped output directory |
| `--prepare-only` | Write & validate manifest without running inference |
| `--resume` | Resume from a completed run directory |
| `--device` | GPU device (default: `cuda:0`) |
| `--brand-text-mode` | Set by each experiment protocol: PhishIntention uses checkpoint-saved prototypes; unseen-brand runs use their external descriptions |

Results are saved under `experiments/<name>/results/<run_name>/` and include the frozen input manifest, detection CSV, predictions, metrics, and logs.

---

## Quick smoke test

Verify that your environment, datasets, models, and inference pipeline are all working:

```bash
python tests/test_smoke.py
```

This checks:
1. Python version and required packages
2. Dataset directories and file counts
3. Model files and SigLIP configuration
4. A minimal 2-sample inference run (1 benign + 1 phishing)

Use `--skip-inference` to check only environment/data/models without running the GPU pipeline.

---

## License

MIT License — see [LICENSE](LICENSE).

**Citation:** TODO — add paper title, authors, venue, year, and DOI/arXiv link.
