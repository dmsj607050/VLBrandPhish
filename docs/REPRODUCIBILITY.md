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

## Artifact preflight

Before releasing an evaluation artifact, validate the tracked tree without loading a dataset or checkpoint:

```bash
python scripts/verify_artifact.py
```

Generate a deterministic SHA-256 manifest for the exact files that reviewers can access. Do not include local data, private URLs, model weights, or Git metadata unless their release terms permit it:

```bash
python scripts/create_artifact_manifest.py \
  --include README.md --include LICENSE --include pyproject.toml \
  --include configs --include docs --include scripts --include src \
  --output artifact_manifest.json
```

For every reported table, archive the command line, the frozen YAML file, the exact checkpoint and data checksums, random seeds, package versions, CUDA/runtime version, GPU model, and the resulting metrics. The repository alone cannot reproduce a paper result without this run-level evidence.

Create the run-level record immediately after each run. It hashes the frozen configuration and declared inputs, records the seed and package versions, and redacts absolute paths outside the repository root. The output is ignored by Git so it cannot be published by accident:

```bash
python scripts/record_run_metadata.py \
  --config configs/training/stage1_siglip.yaml \
  --input data/manifests/stage1_test.json \
  --input models/brand_recognizer.pt \
  --seed 42 --collect-torch-runtime \
  --output outputs/table_2/run_metadata.json
```

Use `--include-git` only for a non-anonymous archival copy. A commit identifier can link a submission to the public repository, so it must not appear in a double-blind artifact. The current training configurations fix `seed: 42`, enable deterministic cuDNN behavior, and request PyTorch deterministic algorithms in warn-only mode; the training engine also seeds Python, NumPy, PyTorch, and data-loader workers. Exact cross-platform equality is still not guaranteed, so record any emitted deterministic-operation warnings and the CUDA/runtime details.
