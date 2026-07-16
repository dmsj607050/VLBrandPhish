from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

from mmengine.runner import Runner

from .config import TrainConfig
from .model import build_mmdet_cfg


class _SuppressStartupNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if not msg:
            return True

        if msg.startswith("Config:"):
            return False
        if msg.startswith("------------------------------------------------------------"):
            return False
        if msg.startswith("System environment:"):
            return False
        if msg.startswith("Runtime environment:"):
            return False
        if "paramwise_options" in msg:
            return False
        return True


def _install_startup_log_filter() -> None:
    for logger_name in ("mmengine", "mmdet"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(_SuppressStartupNoiseFilter())


def _count_coco(ann_file: str):
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annos = coco.get("annotations", [])
    imgid_has_anno = set()
    for a in annos:
        img_id = a.get("image_id")
        if img_id is not None:
            imgid_has_anno.add(int(img_id))

    n_img = len(images)
    n_ann = len(annos)
    n_pos = len(imgid_has_anno)
    n_neg = n_img - n_pos
    return n_img, n_ann, n_pos, n_neg


def _print_summary(cfg_obj: TrainConfig, base_config: str, cfg) -> None:
    train_ann = cfg.train_dataloader["dataset"]["ann_file"]
    val_ann = cfg.val_dataloader["dataset"]["ann_file"]

    tr_n_img, tr_n_ann, tr_n_pos, tr_n_neg = _count_coco(train_ann)
    va_n_img, va_n_ann, va_n_pos, va_n_neg = _count_coco(val_ann)

    approx_iters = int(math.ceil(tr_n_img / max(1, int(cfg_obj.batch_size))))

    actual_base = None
    meta = cfg.get("__logo_detector_dino__")
    if isinstance(meta, dict):
        actual_base = meta.get("base_config")
    print("\n=== Training Summary (logo_detector_dino) ===")
    print(f"base_config: {actual_base or base_config}")
    print(f"work_dir   : {cfg_obj.work_dir}")
    print(f"images_root: {cfg_obj.images_root}")
    print(f"train_ann  : {train_ann}")
    print(f"val_ann    : {val_ann}")
    print(f"classes    : {cfg_obj.classes}")
    print("-- dataset --")
    print(f"train: images={tr_n_img} (pos={tr_n_pos}, neg={tr_n_neg}), annos={tr_n_ann}, approx_iters/epoch={approx_iters}")
    print(f"val  : images={va_n_img} (pos={va_n_pos}, neg={va_n_neg}), annos={va_n_ann}")
    print("-- hparams --")
    print(f"batch_size={cfg_obj.batch_size} workers={cfg_obj.num_workers} epochs={cfg_obj.max_epochs}")
    print(f"lr={cfg_obj.lr} weight_decay={cfg_obj.weight_decay} amp={cfg_obj.amp} grad_clip_norm={cfg_obj.grad_clip_norm}")
    print(f"multiscale_short_edges={cfg_obj.multiscale_short_edges} max_size={cfg_obj.max_size}")
    print(f"swin_pretrained={cfg_obj.swin_pretrained}")
    print(f"detector_load_from={cfg.get('load_from')}")
    print("===========================================\n")


def _dump_patched_config(cfg, work_dir: str) -> str:
    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, "patched_config.py")
    dump_fn = getattr(cfg, "dump", None)
    if callable(dump_fn):
        dump_fn(out_path)
    else:
        text = getattr(cfg, "pretty_text", None)
        if text is None:
            raise RuntimeError("Config object has no dump/pretty_text")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(text))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--verbose-startup", action="store_true")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (before auto-scaling by batch size).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs.",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Load model weights from a checkpoint and start a new training run (does NOT restore optimizer state).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Override train dataset filter_cfg.min_size (smaller keeps more tiny boxes).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in work_dir (uses work_dir/last_checkpoint).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a specific checkpoint path (overrides --resume).",
    )
    parser.add_argument(
        "--dump-config-only",
        action="store_true",
        help="Only dump final patched MMDet config to work_dir/patched_config.py and exit.",
    )
    args = parser.parse_args()

    if not args.verbose_startup:
        _install_startup_log_filter()

    cfg_obj = TrainConfig().resolve_paths()

    work_dir = args.work_dir or cfg_obj.work_dir
    base_config = args.base_config or cfg_obj.base_config_path

    lr = float(args.lr) if args.lr is not None else float(cfg_obj.lr)
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(cfg_obj.max_epochs)
    dataset_min_size = int(args.min_size) if args.min_size is not None else 32

    use_val_split = not bool(getattr(cfg_obj, "val_is_test", False))
    cfg = build_mmdet_cfg(
        images_root=cfg_obj.images_root,
        coco_json=cfg_obj.coco_json,
        val_images_root=(getattr(cfg_obj, "val_images_root", None) if use_val_split else None),
        val_coco_json=(getattr(cfg_obj, "val_coco_json", None) if use_val_split else None),
        save_best_only=bool(getattr(cfg_obj, "save_best_only", True)),
        work_dir=work_dir,
        swin_pretrained_path=cfg_obj.swin_pretrained,
        use_official_detector_pretrained=cfg_obj.use_official_detector_pretrained,
        val_ratio=cfg_obj.val_ratio,
        seed=cfg_obj.seed,
        batch_size=cfg_obj.batch_size,
        num_workers=cfg_obj.num_workers,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=cfg_obj.weight_decay,
        amp=cfg_obj.amp,
        grad_clip_norm=cfg_obj.grad_clip_norm,
        log_interval=cfg_obj.log_interval,
        val_interval=cfg_obj.val_interval,
        classes=cfg_obj.classes,
        multiscale_short_edges=cfg_obj.multiscale_short_edges,
        max_size=cfg_obj.max_size,
        dataset_min_size=dataset_min_size,
        base_config_path=base_config,
        prefer_swin_b=cfg_obj.prefer_swin_b,
        force_patch_swin_b=False,
    )

    if args.resume_from:
        cfg.resume = True
        cfg.resume_from = os.path.abspath(args.resume_from)
        cfg.load_from = None
    elif args.resume:
        cfg.resume = True
        cfg.load_from = None
    elif args.finetune_from:
        cfg.resume = False
        cfg.resume_from = None
        cfg.load_from = os.path.abspath(args.finetune_from)

    _print_summary(cfg_obj, base_config or "<auto>", cfg)

    dumped_cfg = _dump_patched_config(cfg, work_dir=work_dir)
    print(f"patched config dumped to: {dumped_cfg}\n")

    if args.dump_config_only:
        return

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
