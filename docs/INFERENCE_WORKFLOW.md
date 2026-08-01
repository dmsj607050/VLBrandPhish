# VLBrandPhish inference workflow

The code for the VLBrandPhish method is intentionally organized around the execution
path. Run every command from the repository root through `python cli.py`.

```text
webpage screenshots + URLs
        |
        v
Stage 1: detection/command.py
  MMDetection/DINO logo detector
        |
        v
outputs/.../logo_detection_top3.csv
        |
        v
Stage 2: pipeline/cli.py
  data/ -> recognition/ -> decision/ -> evaluation/
        |
        v
predictions.csv + metrics.json + summary.json + run metadata
```

## Module responsibilities

| Package | Responsibility | Reads | Produces |
| --- | --- | --- | --- |
| `data/` | Stable detection-CSV schema, test-set loading, and screenshot resolution | sample roots and Stage-1 CSV | `DetectionSample` records and image crops |
| `detection/` | Runs the logo detector and keeps the top-k boxes per screenshot | detector config, detector checkpoint, benign/phishing roots | Stage-1 detection CSV |
| `recognition/` | Loads the brand recognizer and reference-logo/text repository; encodes detected crops | brand checkpoint, optional local SigLIP directory, logo library, DBI-5 prompts | image/text similarities and candidate brands |
| `decision/` | Checks predicted brand against URL/domain evidence | predicted brand and screenshot URL | brand/domain acceptance signal |
| `evaluation/` | Applies thresholds and writes sample-, brand-, and aggregate metrics | similarities, labels, decisions | predictions and metric reports |
| `runtime/` | Owns only paths, cache keys/files, and typed intermediate records | command arguments | reusable caches and deterministic paths |
| `pipeline/` | Wires the modules above without reimplementing them | all explicit Stage-2 inputs | a complete inference run directory |

## Commands

Stage 1 writes detector outputs. Use explicit dataset roots for a reproducible run:

```bash
python cli.py detect \
  --det-config <mmdetection-config.py> \
  --det-checkpoint <logo-detector.pth> \
  --benign-root <benign-screenshots> \
  --phish-root <phishing-screenshots> \
  --out-csv outputs/detection/logo_detection_top3.csv
```

Stage 2 consumes that CSV and does brand recognition, domain verification, and metric
reporting. `--model-dir` is the portable way to supply the local Hugging Face SigLIP
files when a checkpoint records a training-machine path in `model_name`:

```bash
python cli.py infer \
  --det-csv outputs/detection/logo_detection_top3.csv \
  --brand-ckpt <brand-recognizer.pt> \
  --model-dir <local-siglip-directory> \
  --logo-library-dir <reference-logo-library> \
  --dbi5-dir <brand-prompts> \
  --normal-urls-csv <brand-domain-table.csv> \
  --benign-root <benign-screenshots> \
  --phish-root <phishing-screenshots> \
  --out-dir outputs/inference_smoke
```

For a small smoke run, use `--max-benign 1 --max-phish 1` on Stage 1, then pass the
resulting CSV to Stage 2 with `--max-samples 2`. Stage 1 requires a compatible
MMDetection environment; Stage 2 requires PyTorch, Transformers, and SentencePiece.
Do not enable `--grid-search` on a reported test split unless the threshold-selection
protocol is explicitly documented.

## Output contract

A successful Stage-2 run writes `predictions.csv`, `metrics.json`, `metrics_raw.json`,
`summary.json`, `brand_repository_meta.json`, and a brand-level diagnosis under the
chosen output directory. A `summary.json` records the resolved model/data paths, cache
paths, cache hits, timing, and path-resolution counts. Use
`scripts/record_run_metadata.py` immediately afterwards to bind the outputs to the exact
input/checkpoint hashes and environment.