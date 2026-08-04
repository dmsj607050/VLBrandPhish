# Image-text branch ablation

| | |
| --- | --- |
| Section | Appendix -- branch contribution study |
| Setting | Image-only (w=1.0) / Fusion (w=0.8) / Text-only (w=0.0) on the cropped-logo eval set (~2,700 crops, ~100 brands) |
| Metrics | Match Acc, Brand Top-1, Precision, Recall, F1, AUROC |
| Paper table(s) | Branch ablation table |

## Run

```bash
# Full sweep: image_weight = 1.0, 0.8, 0.0
python experiments/ablations/branch_ablation/run.py

# Single weight
python experiments/ablations/branch_ablation/run.py --image-weights 0.8

# Custom output
python experiments/ablations/branch_ablation/run.py --out-dir results/branch_ablation_test
```

Output under `experiments/ablations/branch_ablation/results/`: `summary.json`, `results.csv`, `results.md`.
