"""Command-line entry point for training the brand recognizer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common.config import load_config, resolve_stage_paths, split_pipeline_config
from .engine import run_training
from .common.utils import print_config


def main(argv: Sequence[str] | None = None) -> None:
    """Train a single recognizer stage or the configured two-stage schedule."""
    parser = argparse.ArgumentParser(description="Train the SigLIP brand recognizer.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/stage1_siglip.yaml",
        help="Path to the YAML training configuration.",
    )
    args = parser.parse_args(argv)

    cfg = resolve_stage_paths(load_config(args.config), project_root=Path.cwd())
    print_config(cfg)
    stage = str(cfg.get("stage", "pipeline")).lower()

    if stage == "pipeline":
        first_stage, second_stage = split_pipeline_config(cfg)
        first_checkpoint = run_training(first_stage)
        if bool(cfg.get("resume_from_stage1", True)):
            second_stage["resume"] = str(first_checkpoint)
        second_checkpoint = run_training(second_stage)
        print("\nTraining schedule finished.")
        print(f"First-stage checkpoint: {first_checkpoint}")
        print(f"Second-stage checkpoint: {second_checkpoint}")
        return

    if stage in {"stage1", "stage2"}:
        checkpoint = run_training(cfg)
        print(f"\nDone. Best checkpoint: {checkpoint}")
        return

    raise ValueError(f"Unknown training stage: {stage}")


if __name__ == "__main__":
    main()
