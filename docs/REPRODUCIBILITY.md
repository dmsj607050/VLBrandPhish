# Reproducibility guide

## Environment

1. Create a clean Python 3.10+ environment.
2. Install a PyTorch build suitable for the available CPU/GPU.
3. Install `requirements.txt` and the project in editable mode.
4. For Stage-1 or detector training, install a PyTorch/CUDA-compatible MMDetection stack as described by OpenMMLab.

Record `python --version`, `pip freeze`, GPU model, CUDA version, and checkpoint SHA-256 for every reported run.

## Brand recognizer training

The training code expects two local dataset roots below `data/` when `data_root` is not set explicitly:

```text
data/
├── Stage1/Logo-2K+/
└── Stage2/277_brand_pages/
```

Update `data_root`, `output_dir`, and any resume checkpoint in a copied YAML file before a new run. Avoid editing the published defaults when reporting a separate experiment.

```bash
vlbrandphish train-brand --config configs/training/stage1_siglip.yaml
vlbrandphish train-brand --config configs/training/stage2_trisig_ot_safe.yaml
```

## End-to-end evaluation

1. Run Stage 1 to create a top-k detection CSV.
2. Run Stage 2 with that CSV, the brand recognizer checkpoint, logo library, domain table, and the explicit benign/phishing roots.
3. Save the command line, input-manifest checksums, configuration, code commit, and output metrics together.

Use `--out-dir` to make each evaluation output location explicit. Generated CSV files, caches, predictions, figures, and checkpoints are ignored by default.
