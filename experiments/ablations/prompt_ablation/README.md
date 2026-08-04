# Text prompt configuration ablation

| | |
| --- | --- |
| Section | Appendix -- prompt configuration study |
| Setting | Cropped-logo eval set (~2,700 crops), 6 prompt configurations at fixed 0.8 img + 0.2 txt fusion |
| Metrics | AUROC, Brand Top-1, F1 (plus Match Acc, Precision, Recall) |
| Paper table(s) | Prompt ablation table |

## Prompt configurations

| Directory | Description |
|---|---|
| `brand_only` | Brand name only |
| `brand_plus_color` | Brand name + color |
| `brand_plus_content` | Brand name + content composition |
| `brand_plus_style` | Brand name + style |
| `brand_plus_industry` | Brand name + industry |
| `visual_all` | Brand name + color + content + style (default) |

All configs live under `data/VLBrandPhish/04_prompt_libraries_152/`.

## Run

```bash
# Full sweep (all 6 configs)
python experiments/ablations/prompt_ablation/run.py

# Specific configs only
python experiments/ablations/prompt_ablation/run.py --configs brand_only,visual_all
```

Output under `experiments/ablations/prompt_ablation/results/`: `summary.json`, `results.csv`, `results.md`, `results_table13.md`.
