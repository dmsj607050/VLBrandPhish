# Brand identification on the unified cropped-logo set

| | |
| --- | --- |
| Research question | RQ1 -- brand identification accuracy |
| Dataset | Cropped-logo eval set (~2,700 crops, ~100 brands) |
| Metrics | Match Acc (benign to benign, phishing to the correct target brand) |
| Text source | Checkpoint-saved text prototypes from the current brand recognizer |
| Methods | VLBrandPhish (0.4 img + 0.6 txt, thr=0.4), PhishIntention (BiT+OCR, thr=0.87), PhishPedia (BiT, thr=0.87) |

## Run

```
python experiments/brand_identification/run.py
```

Results written to `experiments/brand_identification/results/`.
