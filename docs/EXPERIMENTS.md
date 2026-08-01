# Experiment status and reproduction contract

This repository separates reusable analysis tools from paper-study directories. A
directory name alone is not evidence that its paper table can be reproduced. The status
below is the source of truth for the public repository.

| Area | Status | What it can do today |
| --- | --- | --- |
| `experiments/analysis/` | Implemented with cached Stage-2 artifacts | Recompute fixed-fusion brand-mode and diagnostic summaries from an existing inference cache and split files. |
| `experiments/ablations/prompt_ablation/` | Implemented with required model/data artifacts | Evaluate prompt modes or prompt-library sweeps after the detector CSV, checkpoint, logo library, and query cache are supplied. |
| `experiments/preparation/` | Implemented utilities | Build prompt libraries and prepare/edit VOC annotations from explicit local inputs. |
| `experiments/figures/` | Implemented with cached Stage-2 artifacts | Render case-study figures from a saved inference summary, caches, and images. |
| `overall_*`, `logo_detection`, `brand_identification`, `cross_*`, `adversarial_robustness` | Study scaffolds | Describe the paper protocol and expected assets, but do **not** yet implement the complete multi-method table pipeline. Their `run.py` exits with an error instead of claiming success. |

## What a table-reproduction command must provide

Before a scaffold is promoted to an implemented study runner, it must accept or fix all
of the following inputs:

1. A versioned sample manifest, including sample identifiers, screenshot paths, URLs,
   labels, and the train/validation/test split.
2. The exact checkpoint, logo/reference library, and baseline release, each with a
   version and SHA-256 hash.
3. A frozen configuration with the published fixed decision rule: image weight 0.8,
   text weight 0.2, and threshold 0.62. These values are never searched on the
   reported test split.
4. A deterministic output directory containing predictions, metrics, a confusion matrix,
   and the command/environment record.
5. The expected output schema and a check that fails if any selected method did not
   produce its predictions.

## Current runnable entry points

Run commands from the repository root. These commands require the explicitly supplied
local artifacts documented in `docs/DATA_AND_MODELS.md`:

```bash
python cli.py detect --help
python cli.py infer --help
python cli.py train-brand --help
python -m experiments.ablations.prompt_ablation.prompt_modes --help
```

For every paper result, save its command and hashes with:

```bash
python scripts/record_run_metadata.py --help
```

The study scaffolds are retained to make the intended paper protocol visible, but they
must not be cited as completed evaluations until their runners meet the contract above.