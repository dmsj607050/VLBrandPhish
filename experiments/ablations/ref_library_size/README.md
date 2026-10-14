# Reference-logo-count ablation

This experiment crosses four VLBrandPhish training caps (1, 3, 5, and 10 reference logos per brand) with four inference caps (1, 3, 5, and 10). It uses the current brand model: SigLIP pairwise training, four unfrozen text blocks, two unfrozen vision blocks, image/text fusion weights 0.4/0.6, and a 0.4 decision threshold.

The experiment freezes one PhishIntention manifest and one Top-3 Stage-1 detection CSV. VLBrandPhish and PDOS reuse that manifest and detection CSV. PDOS is evaluated with its reproduced cosine threshold of 0.60. Add PhishIntention and Phishpedia curves through a CSV with columns `method,infer_ref,f1`; accepted method names are `PhishIntention` and `Phishpedia`.

## Prepare inputs

Supply the external datasets, model weights, and runtimes described in `model_train/Brand/README.md`, `model_train/Logo/README.md`, and `baselines/PDOS/README.md`. The current 1/3/5/10 training variants have to be trained from the current 4-text/2-vision configuration; older checkpoint results are not reused by this runner.

```bash
python experiments/ablations/ref_library_size/train_variants.py --run-id refcap_text4_vision2_run01
```

## Evaluate and aggregate

```bash
python experiments/ablations/ref_library_size/run.py \
  --run-id refcap_eval_run01 \
  --train-run-id refcap_text4_vision2_run01
```

The default stages are `manifest,stage1,infer,pdos,aggregate`. Use `--steps` to run or resume selected stages only after checking that the existing manifest and detection CSV belong to this run. Pass `--pdos-weights` and `--pdos-resnet-weights` if the PDOS weights are outside their documented default paths. Use `--baseline-f1-csv path/to/baselines_f1.csv` to add the two already evaluated baseline curves.

Each run writes a frozen manifest, Stage-1 CSV, nested reference-library manifests, per-setting predictions and caches, `ablation_table.csv`, and `ablation_table.md` under `results/<run-id>/`. Generated data and model weights are excluded from Git.
