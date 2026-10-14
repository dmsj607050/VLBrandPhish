#!/usr/bin/env python3
"""Train reference-count variants for the current text4/vision2 brand model."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "inference").is_dir())
TRAIN = ROOT / "model_train" / "Brand"
CONFIGS = TRAIN / "configs" / "active"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one current-brand checkpoint per reference-logo cap.")
    parser.add_argument("--counts", default="1,3,5,10")
    parser.add_argument("--run-id", required=True, help="Unique output namespace; never reuses prior checkpoints.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", default="", help="Override the 277-brand training image directory.")
    parser.add_argument("--prompt-dir", default="", help="Override the DBI-4 brand prompt directory.")
    parser.add_argument("--model-dir", default="", help="Override the local SigLIP model directory.")
    parser.add_argument("--init-from", default="", help="Override the Logo-2K+ Stage-1 initialization checkpoint.")
    args = parser.parse_args()

    config_out = Path(__file__).parent / "results" / "training_configs" / args.run_id
    config_out.mkdir(parents=True, exist_ok=False)
    for count in [int(x) for x in args.counts.split(",") if x.strip()]:
        src = CONFIGS / f"ablation_ref{count}_textlast4_vision2.yaml"
        if not src.is_file():
            raise FileNotFoundError(src)
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        cfg["ref_images_per_brand_max"] = count
        cfg["unfreeze_text_last_n"] = 4
        cfg["unfreeze_vision_last_n"] = 2
        cfg["output_dir"] = str(TRAIN / "outputs" / f"refcount_{args.run_id}" / f"train{count}")
        if args.data_root:
            cfg["data_root"] = str(Path(args.data_root).resolve())
        if args.prompt_dir:
            cfg.setdefault("prompts", {})["brand_prompt_txt_dir"] = str(Path(args.prompt_dir).resolve())
        if args.model_dir:
            cfg["model_name"] = str(Path(args.model_dir).resolve())
        if args.init_from:
            cfg["init_from"] = str(Path(args.init_from).resolve())

        dest = config_out / f"train_ref{count}.yaml"
        dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [args.python, str(TRAIN / "train.py"), "--config", str(dest)]
        print("COMMAND:", " ".join(command), flush=True)
        subprocess.run(command, cwd=str(TRAIN), check=True)


if __name__ == "__main__":
    main()
