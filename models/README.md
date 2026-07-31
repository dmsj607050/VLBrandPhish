# Model checkpoints

This directory is ignored by Git (see `.gitignore`). Each method's weights live in its own
subdirectory. The inference code under `inference/<method>/` resolves weight paths here
automatically (see each method's `configs.py`).

Place the files below before running inference. Release them through a versioned archive,
Hugging Face, Zenodo, or another durable host, then record the public download URL and
SHA-256 checksum in `docs/DATA_AND_MODELS.md`. Do not commit large checkpoints to GitHub.

## models/vlbrandphish/

VLBrandPhish's own trained models.

| File | Purpose |
| --- | --- |
| `logo_detector.pth` | DINO logo detector (MMDetection) |
| `brand_recognizer.pt` | SigLIP brand recognizer (stage1 + stage2) |

## models/phishintention/

Official PhishIntention release weights (https://github.com/lindsey98/PhishIntention).

| File | Purpose |
| --- | --- |
| `layout_detector.pth` | AWL layout/logo detector (Detectron2) |
| `crp_classifier.pth.tar` | Credential-requiring page classifier |
| `crp_locator.pth` | CRP element locator (Detectron2) |
| `ocr_siamese.pth.tar` | Siamese logo matcher |
| `ocr_pretrained.pth.tar` | OCR encoder pretrained weights |
| `domain_map.pkl` | Brand -> official domain map |
| `expand_targetlist.zip` | Protected logo reference library |

`LOGO_FEATS.npy` / `LOGO_FILES.npy` are regenerated automatically on first run and cached here.

## models/phishpedia/

Official PhishPedia release weights (https://github.com/lindsey98/phishpedia).

| File | Purpose |
| --- | --- |
| `rcnn_bet365.pth` | Faster R-CNN logo detector (Detectron2) |
| `resnetv2_rgb_new.pth.tar` | Siamese logo matcher |
| `domain_map.pkl` | Brand -> official domain map |
| `expand_targetlist.zip` | Protected logo reference library |

The `faster_rcnn.yaml` detectron2 config ships with the code at
`inference/phishpedia/configs/faster_rcnn.yaml` (not here).

`LOGO_FEATS.npy` / `LOGO_FILES.npy` are regenerated automatically on first run and cached here.
