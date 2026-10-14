# Phase 2: train and evaluate the Logo detector

## Dataset interpretation

The three datasets in the paper's detector analysis are evaluation sets, not three separate training runs:

| Split | Source | Images | Use |
|---|---|---:|---|
| Train | PhishIntention `Logo_train/train_coco.json` | 8,109 | train YOLOv5s+SE |
| Default | PhishIntention `Logo_train/val_coco.json` | 901 | test |
| Merged | Default + Phishpedia `coco_test.json` | 2,480 | test |
| New | VLBrandPhish `01_phish_12K` XML labels | 12,801 | test |

Only the `logo` category is retained from PhishIntention. Phishpedia's `box` category is excluded and its `logo` category is retained. Every XML object in the VLBrandPhish brand folders is converted to the same one-class `logo` target.

This matches the figure's statement that the comparison evaluates logo localization only; input boxes, buttons, labels, blocks, and Phishpedia `box` annotations are not detection targets.

## Commands

Run from the repository root:

```powershell
python baselines/PDOS/scripts/prepare_logo_detection_data.py --link-mode hardlink
```

This creates `baselines/PDOS/data/prepared_logo_detection/`. It uses hard links to avoid duplicating the source images; use `--link-mode copy` if hard links are unavailable.

Train the algorithm from the supplied YOLOv5 v7.0 initialization:

```powershell
python baselines/PDOS/scripts/train_logo_detector.py `
  --epochs 200 `
  --batch-size 32 `
  --device 0 `
  --workers 4
```

The expected checkpoint is:

`baselines/PDOS/results/training/pdos_yolov5s_se_logo/weights/best.pt`

If GPU memory is insufficient, use `--batch-size 16` or `--batch-size 8` and record the difference from the paper. The implementation uses SGD, learning rate 0.01, momentum 0.937, weight decay 0.0005, seed 42, input size 640, SE reduction 16, first 10 actual backbone Conv2d layers frozen, and P3 objectness loss weight 1.5.

Evaluate all three sets:

```powershell
python baselines/PDOS/scripts/evaluate_logo_detector.py `
  --weights baselines/PDOS/results/training/pdos_yolov5s_se_logo/weights/best.pt `
  --device 0 `
  --save-predictions
```

The output is `results/tables/table_logo_detection.csv`, including mAP50, mAP50:95, precision/recall/F1 at IoU 0.50, box counts, and average inference time. `mAP50:95` is the requested Table 6-style metric.

For a quick smoke test before committing to a full evaluation:

```powershell
python .\scripts\evaluate_logo_detector.py `
  --weights .\models\yolo\yolov5s.pt `
  --device 0 `
  --max-images 3 `
  --output baselines/PDOS/results/tables/smoke_logo_detection.csv
```

That smoke test uses the COCO initialization with an untrained one-class head and is only a pipeline check, not a paper result.
