from __future__ import annotations

import argparse
import json
import os
import glob
import sys
import tempfile
from typing import Any, Optional

try:
    from mmengine.config import Config
    from mmengine.runner import Runner
except ModuleNotFoundError as e:
    missing = getattr(e, "name", None) or "unknown"
    print(
        "Missing dependency: {}\n\n"
        "This script requires MMDetection runtime packages (at least: mmengine, mmcv, mmdet).\n"
        "On Windows, please activate your conda env and install them first.\n\n"
        "Example (CPU-only):\n"
        "  pip install -U openmim\n"
        "  mim install mmengine\n"
        "  mim install 'mmcv>=2.0.0'\n"
        "  pip install -U mmdet pycocotools opencv-python\n\n"
        "Example (PowerShell line continuation uses backtick `, not ^).\n"
        .format(missing)
    )
    sys.exit(1)


def _set_ann_file(obj: Any, ann_file: str) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        obj["ann_file"] = ann_file
        return
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if isinstance(it, dict):
                it["ann_file"] = ann_file


def _set_img_prefix(dataset_cfg: Any, images_root: Optional[str]) -> None:
    if not images_root:
        return
    if not isinstance(dataset_cfg, dict):
        return
    dataset_cfg["data_root"] = ""
    dataset_cfg["data_prefix"] = {"img": images_root}


def _ensure_test_section(cfg: Config) -> None:
    if "test_dataloader" not in cfg and "val_dataloader" in cfg:
        cfg.test_dataloader = cfg.val_dataloader
    if "test_evaluator" not in cfg and "val_evaluator" in cfg:
        cfg.test_evaluator = cfg.val_evaluator


def _load_coco(ann_file: str) -> Any:
    with open(ann_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_coco_category_stats(ann_file: str) -> None:
    try:
        coco = _load_coco(ann_file)
        cats = coco.get("categories", []) if isinstance(coco, dict) else []
        anns = coco.get("annotations", []) if isinstance(coco, dict) else []
        if not isinstance(cats, list) or not isinstance(anns, list):
            return
        id2name = {}
        for c in cats:
            if isinstance(c, dict) and isinstance(c.get("id"), int):
                id2name[int(c["id"])] = str(c.get("name"))
        counts = {}
        for a in anns:
            if not isinstance(a, dict):
                continue
            cid = a.get("category_id")
            if isinstance(cid, int):
                counts[int(cid)] = counts.get(int(cid), 0) + 1

        print("[eval_test] ann_file:", ann_file)
        print("[eval_test] categories:", [{"id": k, "name": v} for k, v in sorted(id2name.items())])
        print(
            "[eval_test] ann_count_by_category:",
            {id2name.get(k, str(k)): int(v) for k, v in sorted(counts.items())},
        )
    except Exception:
        return


def _filter_coco_by_category_name(
    *,
    ann_file: str,
    category_name: str,
    out_dir: Optional[str],
) -> str:
    coco = _load_coco(ann_file)
    if not isinstance(coco, dict):
        raise RuntimeError("Invalid COCO json")

    cats = coco.get("categories", [])
    if not isinstance(cats, list):
        raise RuntimeError("COCO json missing categories")

    matched = [c for c in cats if isinstance(c, dict) and str(c.get("name")) == str(category_name)]
    if not matched:
        raise RuntimeError(f"Category '{category_name}' not found in ann file")

    keep_id = matched[0].get("id")
    if not isinstance(keep_id, int):
        raise RuntimeError("Invalid category id")

    anns = coco.get("annotations", [])
    if not isinstance(anns, list):
        raise RuntimeError("COCO json missing annotations")

    keep_anns = [a for a in anns if isinstance(a, dict) and int(a.get("category_id")) == int(keep_id)]
    for a in keep_anns:
        a["category_id"] = 1

    out = {
        "images": coco.get("images", []),
        "annotations": keep_anns,
        "categories": [{"id": 1, "name": str(category_name)}],
    }

    os.makedirs(out_dir or tempfile.gettempdir(), exist_ok=True)
    base = os.path.splitext(os.path.basename(ann_file))[0]
    out_path = os.path.join(out_dir or tempfile.gettempdir(), f"{base}__{category_name}_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out_path


def _guess_project_root(here: str) -> str:
    return os.path.abspath(here)


def _auto_find_checkpoint(work_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(work_dir, "best_*.pth")))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return os.path.abspath(candidates[0])

    last_ckpt_file = os.path.join(work_dir, "last_checkpoint")
    if os.path.isfile(last_ckpt_file):
        with open(last_ckpt_file, "r", encoding="utf-8") as f:
            p = f.read().strip()
        if p:
            if not os.path.isabs(p):
                p = os.path.join(work_dir, p)
            if os.path.isfile(p):
                return os.path.abspath(p)

    epoch_candidates = sorted(glob.glob(os.path.join(work_dir, "epoch_*.pth")))
    if epoch_candidates:
        epoch_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return os.path.abspath(epoch_candidates[0])

    raise FileNotFoundError(f"No checkpoint found under: {work_dir}")


def _filter_coco_by_category_id(
    *,
    ann_file: str,
    category_id: int,
    out_dir: Optional[str],
) -> str:
    coco = _load_coco(ann_file)
    if not isinstance(coco, dict):
        raise RuntimeError("Invalid COCO json")

    cats = coco.get("categories", [])
    if not isinstance(cats, list):
        raise RuntimeError("COCO json missing categories")

    matched = [c for c in cats if isinstance(c, dict) and int(c.get("id")) == int(category_id)]
    if not matched:
        raise RuntimeError(f"Category id '{category_id}' not found in ann file")

    keep_name = matched[0].get("name")
    keep_name = str(keep_name) if keep_name is not None else str(category_id)

    anns = coco.get("annotations", [])
    if not isinstance(anns, list):
        raise RuntimeError("COCO json missing annotations")

    keep_anns = [a for a in anns if isinstance(a, dict) and int(a.get("category_id")) == int(category_id)]
    for a in keep_anns:
        a["category_id"] = 1

    out = {
        "images": coco.get("images", []),
        "annotations": keep_anns,
        "categories": [{"id": 1, "name": keep_name}],
    }

    os.makedirs(out_dir or tempfile.gettempdir(), exist_ok=True)
    base = os.path.splitext(os.path.basename(ann_file))[0]
    out_path = os.path.join(out_dir or tempfile.gettempdir(), f"{base}__catid{int(category_id)}_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out_path


def _patch_for_test(
    cfg: Config,
    *,
    ann_file: str,
    images_root: Optional[str],
    checkpoint: str,
    work_dir: Optional[str],
    outfile_prefix: Optional[str],
    log_interval: int,
) -> None:
    _ensure_test_section(cfg)

    if work_dir:
        cfg.work_dir = work_dir

    cfg.load_from = checkpoint

    test_dl = cfg.get("test_dataloader")
    if isinstance(test_dl, dict):
        ds = test_dl.get("dataset")
        if isinstance(ds, dict):
            ds["ann_file"] = ann_file
            _set_img_prefix(ds, images_root)

    _set_ann_file(cfg.get("test_evaluator"), ann_file)

    if outfile_prefix and isinstance(cfg.get("test_evaluator"), dict):
        cfg.test_evaluator["outfile_prefix"] = outfile_prefix

    cfg.setdefault("default_hooks", {})
    if isinstance(cfg.default_hooks, dict):
        logger_hook = cfg.default_hooks.get("logger")
        if not isinstance(logger_hook, dict):
            logger_hook = {"type": "LoggerHook"}
            cfg.default_hooks["logger"] = logger_hook
        logger_hook["interval"] = int(log_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="MMDet config (recommend: work_dir/patched_config.py)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ann-file", type=str, default=None)
    parser.add_argument(
        "--category-name",
        type=str,
        default=None,
        help="Optional: keep only this category from COCO ann-file (useful when ann-file contains extra categories)",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=None,
        help="Optional: keep only this category_id from COCO ann-file (e.g., 1 for logo in coco_test.json)",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Optional images root to set as data_prefix.img for test dataset",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Optional work_dir for logs/eval artifacts",
    )
    parser.add_argument(
        "--outfile-prefix",
        type=str,
        default=None,
        help="Optional outfile prefix for evaluator outputs",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Print progress every N iterations (via LoggerHook)",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable ProgressBarHook (if available)",
    )

    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = _guess_project_root(here)

    work_dir = os.path.abspath(args.work_dir) if args.work_dir else os.path.join(here, "work_dir")

    config_path = os.path.abspath(args.config) if args.config else os.path.join(work_dir, "patched_config.py")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    ann_file_in = (
        os.path.abspath(args.ann_file)
        if args.ann_file
        else os.path.join(project_root, "datasets", "PhishIntention", "val_coco.json")
    )
    if not os.path.isfile(ann_file_in):
        raise FileNotFoundError(f"Ann file not found: {ann_file_in}")

    images_root = (
        os.path.abspath(args.images_root)
        if args.images_root
        else os.path.join(project_root, "datasets", "PhishIntention", "val_imgs")
    )

    checkpoint_path = os.path.abspath(args.checkpoint) if args.checkpoint else _auto_find_checkpoint(work_dir)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ann_file = ann_file_in
    _print_coco_category_stats(ann_file)

    if args.category_name and args.category_id is not None:
        raise ValueError("Please specify only one of --category-name or --category-id")

    if args.category_id is not None:
        ann_file = _filter_coco_by_category_id(
            ann_file=ann_file,
            category_id=int(args.category_id),
            out_dir=os.path.abspath(args.work_dir) if args.work_dir else None,
        )
        _print_coco_category_stats(ann_file)
    elif args.category_name:
        ann_file = _filter_coco_by_category_name(
            ann_file=ann_file,
            category_name=str(args.category_name),
            out_dir=os.path.abspath(args.work_dir) if args.work_dir else None,
        )
        _print_coco_category_stats(ann_file)

    cfg = Config.fromfile(config_path)

    _patch_for_test(
        cfg,
        ann_file=ann_file,
        images_root=images_root,
        checkpoint=checkpoint_path,
        work_dir=work_dir,
        outfile_prefix=os.path.abspath(args.outfile_prefix) if args.outfile_prefix else None,
        log_interval=int(args.log_interval),
    )

    runner = Runner.from_cfg(cfg)
    if not args.no_progress_bar:
        try:
            from mmengine.hooks import ProgressBarHook  # type: ignore

            runner.register_hook(ProgressBarHook(), priority="LOWEST")
        except Exception:
            pass
    runner.test()


if __name__ == "__main__":
    main()
