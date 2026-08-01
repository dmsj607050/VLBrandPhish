# Reproducibility guide

## Runtime environment

Use Python 3.10 or 3.11 and run every command from the repository root. This is a flat
source layout: `inference/`, `training/`, and `experiments/` are imported directly, so
there is no installed `vlbrandphish` command.

1. Create a clean Python environment.
2. Install a PyTorch build matched to the operating system, GPU/CPU, and CUDA runtime.
3. Install the Python requirements:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. For logo detection or detector training, install a PyTorch/CUDA-compatible
   MMDetection stack, then run `pip install -r requirements-detector.txt`.

For every reported run, archive `python --version`, `pip freeze`, GPU model, CUDA
version, checkpoint SHA-256, data/split SHA-256, and the exact command. The repository
currently records version ranges only; publication requires freezing the exact working
environment for each table.

## Brand-recognizer training

If `data_root` is not set in the YAML file, the training code searches for the following
local dataset layout:

```text
data/
|-- Stage1/Logo-2K+/
`-- Stage2/277_brand_pages/
```

Copy a YAML configuration before changing `data_root`, `output_dir`, or `resume` for a
new experiment. Keep the copied YAML with the resulting metrics.

```bash
python cli.py train-brand \
  --config training/vlbrandphish/brand_recognition/configs/stage1_siglip.yaml

python cli.py train-brand \
  --config training/vlbrandphish/brand_recognition/configs/stage2_trisig_ot_safe.yaml
```

## VLBrandPhish inference

The intended two-stage workflow is explicit: first write top-k detections, then provide
that CSV, a brand-recognizer checkpoint, reference library, URL-domain table, and the
benign/phishing evaluation roots to inference.

```bash
python cli.py detect --help
python cli.py infer --help
```

Use explicit paths for all data and model artifacts. Outputs, feature caches, and
checkpoints are ignored by Git; save their hashes and metadata beside the metrics.

## Experiments and paper tables

See [experiment status and reproduction contract](EXPERIMENTS.md). Several analysis,
prompt-ablation, figure, and preparation tools are executable once their cached or local
inputs are supplied. The high-level study directories (`overall_*`, logo detection,
cross-time, cross-brand, adversarial robustness) are protocol scaffolds, not yet
one-command paper-table evaluators. They deliberately fail rather than returning a
misleading successful run.

## Artifact preflight

Validate the tracked source tree without loading a dataset or checkpoint:

```bash
python scripts/verify_artifact.py
```

Generate a deterministic SHA-256 manifest for exactly the files shared with reviewers:

```bash
python scripts/create_artifact_manifest.py --tracked-only \
  --include README.md --include LICENSE --include pyproject.toml --include cli.py \
  --include data --include models --include docs --include experiments \
  --include inference --include training --include scripts \
  --output artifact_manifest.json
```

Create a run-level record immediately after each completed run. It hashes the frozen
configuration and declared inputs, records the seed and package versions, and redacts
absolute paths outside the repository root:

```bash
python scripts/record_run_metadata.py \
  --config training/vlbrandphish/brand_recognition/configs/stage1_siglip.yaml \
  --input data/manifests/stage1_test.json \
  --input models/vlbrandphish/brand_recognizer.pt \
  --seed 42 --collect-torch-runtime \
  --output outputs/table_2/run_metadata.json
```

Use `--include-git` only for a non-anonymous archival copy. A commit identifier can link
a submission to the public repository, so it must not appear in a double-blind artifact.