import argparse
from pathlib import Path

from vlbrandphish.training.config import load_config, resolve_stage_paths, split_pipeline_config
from vlbrandphish.training.train_eval import run_training
from vlbrandphish.training.utils import print_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SigLIP brand identity training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/stage1_siglip.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = resolve_stage_paths(cfg, project_root=Path.cwd())
    print_config(cfg)

    stage = str(cfg.get("stage", "pipeline")).lower()
    if stage == "pipeline":
        cfg_stage1, cfg_stage2 = split_pipeline_config(cfg)
        best_stage1 = run_training(cfg_stage1)

        resume_from_stage1 = cfg.get("resume_from_stage1")
        if resume_from_stage1 is None:
            resume_from_stage1 = True
        if bool(resume_from_stage1):
            cfg_stage2["resume"] = str(best_stage1)

        best_stage2 = run_training(cfg_stage2)
        print("\nPipeline finished.")
        print(f"Stage1 best: {best_stage1}")
        print(f"Stage2 best: {best_stage2}")
        return

    if stage in {"stage1", "stage2"}:
        best_path = run_training(cfg)
        print(f"\nDone. Best checkpoint: {best_path}")
        return

    raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
