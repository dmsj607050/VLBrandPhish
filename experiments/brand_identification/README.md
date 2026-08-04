# Brand identification on the unified cropped-logo set

| | |
| --- | --- |
| Research question | RQ2 -- brand identification accuracy |
| Dataset | Cropped-logo eval set (~2,700 crops, ~100 brands) |
| Metrics | Match Acc (benign→benign, phish→correct target brand) |
| Methods | VLBrandPhish (0.8 img + 0.2 txt, thr=0.62), PhishIntention (BiT+OCR, thr=0.87), PhishPedia (BiT, thr=0.87) |

## Run

```
python experiments/brand_identification/run.py
```

Results written to `experiments/brand_identification/results/`.
