# Logo Detection Evaluation

VLBrandPhish DINO+SwinL logo detector evaluation on two public test sets.

| | |
| --- | --- |
| Research question | RQ2 — logo localisation performance |
| Datasets | PhishIntention val (901 images) / PhishPedia test (1,579 images) |
| Metrics | COCO bbox mAP@[.50:.95], mAP@.50, mAP@.75 (logo class only) |
| Paper table(s) | cn-logo-det |

## Datasets (expected under `data/`)

| Dataset | Images | GT logos | Format |
| --- | --- | --- | --- |
| phishintention | 901 | 934 | COCO JSON (`data/phishintention/val_coco.json` + `val_imgs/`) |
| phishpedia | 1,579 | 1,464 | COCO JSON (`data/phishpedia/coco_test.json`) |

Paths are relative to the repo root. Override via `--pi-coco-json` / `--pp-coco-json` / `--pp-images-root`.

## Run

**Smoke test (1 image):**

```
python experiments/logo_detection/run.py \
  --det-config models/vlbrandphish/patched_config.py \
  --det-checkpoint models/vlbrandphish/logo_detector.pth \
  --max-images 1
```

**Full evaluation (both datasets):**

```
python experiments/logo_detection/run.py \
  --det-config models/vlbrandphish/patched_config.py \
  --det-checkpoint models/vlbrandphish/logo_detector.pth \
  --dataset both
```

Outputs under `experiments/logo_detection/results/`.
