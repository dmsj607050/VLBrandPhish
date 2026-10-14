# PDOS baseline: project reproduction

This directory contains this project's reproduction of PDOS. It is not the authors' official source release. The included implementation, experiment scripts, configuration files, and tests are the locally maintained reproduction.

Model weights, datasets, and run outputs are excluded from Git. The reference-count ablation can reuse the experiment's frozen Top-3 detection CSV and invoke scripts/run_pdos_staged_experiment.py with --stage recognize and --stage decide.

## External assets

Provide the trained PDOS logo detector, ImageNet ResNet-101 weights, YOLOv5 v7.0 source, and reference logo library separately. The staged evaluator accepts explicit --weights, --resnet-weights, and --logo-library-root paths. Do not use the COCO-initialized detector from run_pdos.py as a reported PDOS result; it is intended only for smoke testing.

Install requirements.txt and the required PyTorch/CUDA dependencies in a compatible environment. Dataset paths and model weights are never committed.

## Reproducibility notes

The existing PDOS evaluation uses a cosine-similarity threshold of 0.60. The method is described as a project reproduction, with its assumptions documented in IMPLEMENTATION_DECISIONS.md.