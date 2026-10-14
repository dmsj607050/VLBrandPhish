"""Launch the pinned YOLOv5 v7.0 trainer with the custom SE module."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = ROOT / "third_party" / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    args = parse_args()
    data_yaml = args.data_root / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Prepared dataset missing: {data_yaml}. Run prepare_logo_detection_data.py first.")
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(f"Resume checkpoint missing: {args.resume}")
    if args.resume is None and not args.weights.exists():
        raise FileNotFoundError(f"Initial weights missing: {args.weights}")

    # YOLOv5 v7.0 imports a few optional notebook/plotting packages at module
    # import time. These stubs are local and do not install or download them.
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    from pdos.detector import _install_ipython_compat
    _install_ipython_compat()
    sys.path.insert(0, str(YOLO_ROOT))

    import models.yolo as yolo_module  # type: ignore
    # Use a real importable module path because torch.save pickles the class
    # path when saving the complete YOLO model object.
    from pdos.se import SEBlock
    yolo_module.SEBlock = SEBlock

    sys.argv = [
        str(YOLO_ROOT / "train.py"),
        "--data", str(data_yaml),
        "--cfg", str(ROOT / "configs" / "pdos_yolov5s_se_logo.yaml"),
        "--hyp", str(ROOT / "configs" / "hyp.pdos.yaml"),
        "--weights", str(args.weights),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img", str(args.img),
        "--optimizer", "SGD",
        "--device", args.device,
        "--workers", str(args.workers),
        "--project", str(args.project),
        "--name", args.name,
        "--exist-ok",
        "--freeze-convs", str(args.freeze_convs),
        "--seed", str(args.seed),
    ]
    if args.resume is not None:
        sys.argv.extend(["--resume", str(args.resume)])
    runpy.run_path(str(YOLO_ROOT / "train.py"), run_name="__main__")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "prepared_logo_detection")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo" / "yolov5s.pt")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a YOLOv5 last.pt checkpoint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=ROOT / "results" / "training")
    parser.add_argument("--name", default="pdos_yolov5s_se_logo")
    parser.add_argument("--freeze-convs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
