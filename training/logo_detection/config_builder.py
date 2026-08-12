from __future__ import annotations

import glob
import os
import re
from typing import Any, Optional, Sequence

from mmengine.config import Config

from .dataset import create_train_val_coco_split, patch_dataset_cfg
from .loss import patch_loss_cfg


def _set_evaluator_ann_file(evaluator: Any, ann_file: str) -> None:
    if evaluator is None:
        return
    if isinstance(evaluator, dict):
        evaluator["ann_file"] = ann_file
        return
    if isinstance(evaluator, (list, tuple)):
        for ev in evaluator:
            if isinstance(ev, dict):
                ev["ann_file"] = ann_file


def _find_mmdet_configs_root() -> str:
    import mmdet

    pkg_dir = os.path.dirname(os.path.abspath(mmdet.__file__))
    candidates = [
        os.path.join(pkg_dir, ".mim", "configs"),
        os.path.join(os.path.dirname(pkg_dir), "configs"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError("Could not find mmdetection configs directory; verify mmdet is installed correctly")


def locate_base_config(prefer_swin_b: bool = True) -> str:
    root = _find_mmdet_configs_root()

    if prefer_swin_b:
        patterns = [
            os.path.join(root, "**", "dino*5scale*swin*b*.py"),
            os.path.join(root, "**", "dino*5scale*swin*base*.py"),
            os.path.join(root, "**", "dino*5scale*swin*l*.py"),
            os.path.join(root, "**", "dino*5scale*swin*large*.py"),
            os.path.join(root, "**", "deformable_detr*two_stage*.py"),
        ]
    else:
        patterns = [
            os.path.join(root, "**", "dino*5scale*swin*l*.py"),
            os.path.join(root, "**", "dino*5scale*swin*large*.py"),
            os.path.join(root, "**", "dino*5scale*swin*b*.py"),
            os.path.join(root, "**", "dino*5scale*swin*base*.py"),
            os.path.join(root, "**", "deformable_detr*two_stage*.py"),
        ]

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))

    matches = [m for m in matches if os.path.isfile(m)]
    if not matches:
        raise RuntimeError(
            "No DINO/Deformable-DETR config with Swin backbone found in mmdet configs. "
            "Set TrainConfig.base_config_path to an explicit mmdet config file path."
        )

    if prefer_swin_b:
        for m in matches:
            low = os.path.basename(m).lower()
            if "swin" in low and ("swin-b" in low or "swin_b" in low or "base" in low):
                return m

    return matches[0]


def _find_nearest_metafile(base_cfg_path: str, configs_root: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(base_cfg_path))
    stop = os.path.abspath(configs_root)
    while True:
        for name in ("metafile.yml", "metafile.yaml"):
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
        if os.path.abspath(d) == stop:
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _find_official_weights_for_base_config(base_cfg_path: str) -> Optional[str]:
    configs_root = _find_mmdet_configs_root()
    metafile = _find_nearest_metafile(base_cfg_path, configs_root=configs_root)
    if metafile is None:
        return None

    rel_cfg = os.path.relpath(os.path.abspath(base_cfg_path), os.path.abspath(configs_root))
    rel_cfg = rel_cfg.replace(os.sep, "/")

    def _norm_cfg_path(p: str) -> str:
        p = str(p).strip().strip("'\"")
        p = p.replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        if p.startswith("configs/"):
            p = p[len("configs/") :]
        return p

    rel_cfg_norm = _norm_cfg_path(rel_cfg)
    rel_cfg_candidates = {
        rel_cfg_norm,
        f"configs/{rel_cfg_norm}",
        f"./{rel_cfg_norm}",
        f"./configs/{rel_cfg_norm}",
    }

    # Prefer parsing with PyYAML when available.
    try:
        import yaml  # type: ignore

        with open(metafile, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict):
            collections = data.get("Collections", [])
            if isinstance(collections, list):
                exact_matches = []
                basename_matches = []
                for coll in collections:
                    if not isinstance(coll, dict):
                        continue
                    models = coll.get("Models", [])
                    if not isinstance(models, list):
                        continue
                    for m in models:
                        if not isinstance(m, dict):
                            continue
                        cfg = m.get("Config")
                        if not isinstance(cfg, str):
                            continue
                        cfg_norm = _norm_cfg_path(cfg)
                        if cfg_norm in rel_cfg_candidates:
                            exact_matches.append(m)
                        elif os.path.basename(cfg_norm) == os.path.basename(rel_cfg_norm):
                            basename_matches.append(m)

                if exact_matches:
                    w = exact_matches[0].get("Weights")
                    if isinstance(w, str) and w.strip():
                        return w.strip()

                if len(basename_matches) == 1:
                    w = basename_matches[0].get("Weights")
                    if isinstance(w, str) and w.strip():
                        return w.strip()
    except Exception:
        pass

    # Fallback: line-based parse (works for common MMDet metafile.yml format).
    cfg_re = re.compile(r"^\s*Config\s*:\s*(.+?)\s*$")
    weights_re = re.compile(r"^\s*Weights\s*:\s*(.+?)\s*$")

    current_cfg: Optional[str] = None
    with open(metafile, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            m_cfg = cfg_re.match(line)
            if m_cfg:
                current_cfg = _norm_cfg_path(m_cfg.group(1))
                continue
            m_w = weights_re.match(line)
            if m_w and current_cfg in rel_cfg_candidates:
                w = m_w.group(1).strip().strip("'\"")
                return w if w else None

    return None


def _maybe_set_official_detector_pretrained(cfg: Config, base_cfg_path: str) -> None:
    if cfg.get("load_from"):
        return
    weights = _find_official_weights_for_base_config(base_cfg_path)
    if weights is None:
        return
    cfg["load_from"] = weights


def _recursive_set_num_classes(obj: Any, num_classes: int) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "num_classes" and isinstance(v, int):
                obj[k] = int(num_classes)
            else:
                _recursive_set_num_classes(v, num_classes)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _recursive_set_num_classes(v, num_classes)


def patch_swin_backbone_to_base(cfg: Any, swin_pretrained_path: Optional[str] = None) -> Any:
    model = cfg.get("model")
    if model is None or not isinstance(model, dict):
        return cfg

    backbone = model.get("backbone")
    if not isinstance(backbone, dict) or "Swin" not in str(backbone.get("type", "")):
        return cfg

    backbone["embed_dims"] = 128
    backbone["depths"] = [2, 2, 18, 2]
    backbone["num_heads"] = [4, 8, 16, 32]
    backbone.setdefault("window_size", 12)

    init_cfg = backbone.get("init_cfg")
    if not isinstance(init_cfg, dict):
        init_cfg = {"type": "Pretrained", "checkpoint": ""}
        backbone["init_cfg"] = init_cfg

    ckpt = None
    if swin_pretrained_path and os.path.isfile(swin_pretrained_path):
        ckpt = swin_pretrained_path
    else:
        ckpt = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth"

    init_cfg["checkpoint"] = ckpt

    if "pretrained" in cfg:
        cfg["pretrained"] = init_cfg["checkpoint"]

    neck = model.get("neck")
    if isinstance(neck, dict) and "in_channels" in neck:
        neck["in_channels"] = [128, 256, 512, 1024]

    if isinstance(neck, dict) and neck.get("type") == "ChannelMapper":
        neck.setdefault("out_channels", 256)

    return cfg


def build_mmdet_cfg(
    *,
    images_root: str,
    coco_json: str,
    val_images_root: Optional[str] = None,
    val_coco_json: Optional[str] = None,
    save_best_only: bool = True,
    work_dir: str,
    swin_pretrained_path: Optional[str],
    use_official_detector_pretrained: bool,
    val_ratio: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    grad_clip_norm: float,
    log_interval: int,
    val_interval: int,
    classes: Sequence[str],
    multiscale_short_edges: Sequence[int],
    max_size: int,
    dataset_min_size: int = 32,
    base_config_path: Optional[str] = None,
    prefer_swin_b: bool = True,
    force_patch_swin_b: bool = True,
) -> Config:
    """Build a patched MMDetection DINO config for logo detection.

    Loads the upstream DINO-5scale config, then patches dataset paths,
    class count, LR schedule, Swin-B backbone dimensions, and AMP settings.
    When ``val_coco_json`` is provided it is used as-is; otherwise the
    training COCO is split internally.
    """
    base = base_config_path or locate_base_config(prefer_swin_b=prefer_swin_b)
    cfg = Config.fromfile(base)

    cfg.setdefault("__logo_detector_dino__", {})
    if isinstance(cfg["__logo_detector_dino__"], dict):
        cfg["__logo_detector_dino__"]["base_config"] = str(base)

    if use_official_detector_pretrained:
        _maybe_set_official_detector_pretrained(cfg, base_cfg_path=base)

    # If we are going to load the full detector weights (cfg.load_from), the backbone init_cfg
    # becomes redundant. Disable it to avoid downloading large ImageNet-22k Swin checkpoints.
    if isinstance(cfg.get("load_from"), str) and cfg.get("load_from"):
        model = cfg.get("model")
        if isinstance(model, dict):
            backbone = model.get("backbone")
            if isinstance(backbone, dict):
                backbone.pop("init_cfg", None)
        if "pretrained" in cfg:
            cfg["pretrained"] = None

    os.makedirs(work_dir, exist_ok=True)

    split_dir = os.path.join(work_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    if val_coco_json:
        train_ann = coco_json
        val_ann = val_coco_json
    else:
        train_ann = os.path.join(split_dir, "train.json")
        val_ann = os.path.join(split_dir, "val.json")
        create_train_val_coco_split(
            coco_json_path=coco_json,
            out_train_json=train_ann,
            out_val_json=val_ann,
            val_ratio=val_ratio,
            seed=seed,
        )

    cfg.work_dir = work_dir

    # Linear LR scaling: lr = base_lr * (batch_size / 16).
    # Keeps the effective step size roughly constant across batch sizes.
    base_batch_size = 16
    scaled_lr = float(lr) * (float(batch_size) / float(base_batch_size))
    if isinstance(cfg.get("__logo_detector_dino__"), dict):
        cfg["__logo_detector_dino__"]["base_batch_size"] = int(base_batch_size)
        cfg["__logo_detector_dino__"]["lr_input"] = float(lr)
        cfg["__logo_detector_dino__"]["lr_scaled"] = float(scaled_lr)

    cfg.setdefault("randomness", {})
    cfg.randomness["seed"] = int(seed)
    cfg.randomness["deterministic"] = False

    if "train_dataloader" in cfg:
        cfg.train_dataloader["batch_size"] = int(batch_size)
        cfg.train_dataloader["num_workers"] = int(num_workers)
        cfg.train_dataloader.setdefault("persistent_workers", True)

    if "val_dataloader" in cfg:
        cfg.val_dataloader["batch_size"] = 1
        cfg.val_dataloader["num_workers"] = int(num_workers)

    if "test_dataloader" in cfg:
        cfg.test_dataloader["batch_size"] = 1
        cfg.test_dataloader["num_workers"] = int(num_workers)

    patch_dataset_cfg(
        cfg,
        images_root=images_root,
        train_images_root=images_root,
        val_images_root=val_images_root,
        train_ann_file=train_ann,
        val_ann_file=val_ann,
        classes=classes,
        multiscale_short_edges=multiscale_short_edges,
        max_size=max_size,
        min_size=int(dataset_min_size),
    )

    _recursive_set_num_classes(cfg.get("model"), num_classes=len(classes))
    patch_loss_cfg(cfg, num_classes=len(classes))

    if force_patch_swin_b:
        patch_swin_backbone_to_base(cfg, swin_pretrained_path=swin_pretrained_path)

    if "train_cfg" in cfg and isinstance(cfg.train_cfg, dict):
        if "max_epochs" in cfg.train_cfg or cfg.train_cfg.get("type") == "EpochBasedTrainLoop":
            cfg.train_cfg["max_epochs"] = int(max_epochs)
        # Write val_interval into the training loop so MMDetection actually
        # runs validation at the configured cadence instead of the base default.
        if val_interval > 0:
            cfg.train_cfg["val_interval"] = int(val_interval)

    if "max_epochs" in cfg:
        cfg["max_epochs"] = int(max_epochs)

    if "optim_wrapper" in cfg and isinstance(cfg.optim_wrapper, dict):
        optim = cfg.optim_wrapper.get("optimizer")
        if isinstance(optim, dict):
            optim["lr"] = float(scaled_lr)
            optim["weight_decay"] = float(weight_decay)

        cfg.optim_wrapper.setdefault("clip_grad", None)
        if grad_clip_norm and grad_clip_norm > 0:
            cfg.optim_wrapper["clip_grad"] = {"max_norm": float(grad_clip_norm), "norm_type": 2}

        if amp:
            cfg.optim_wrapper["type"] = "AmpOptimWrapper"
            cfg.optim_wrapper["loss_scale"] = "dynamic"

    if "default_hooks" in cfg and isinstance(cfg.default_hooks, dict):
        logger_hook = cfg.default_hooks.get("logger")
        if isinstance(logger_hook, dict):
            logger_hook["interval"] = int(log_interval)

        checkpoint_hook = cfg.default_hooks.get("checkpoint")
        if isinstance(checkpoint_hook, dict):
            checkpoint_hook.setdefault("save_best", "auto")
            if bool(save_best_only):
                checkpoint_hook["interval"] = int(max_epochs) + 1
                checkpoint_hook["save_last"] = False
                checkpoint_hook["max_keep_ckpts"] = 1
            else:
                checkpoint_hook.setdefault("interval", int(val_interval))

    if "val_evaluator" in cfg:
        _set_evaluator_ann_file(cfg.val_evaluator, val_ann)

    if "test_evaluator" in cfg:
        _set_evaluator_ann_file(cfg.test_evaluator, val_ann)

    sched = cfg.get("param_scheduler")
    if isinstance(sched, list):
        for s in sched:
            if not isinstance(s, dict):
                continue
            if "end" in s and isinstance(s["end"], int):
                s["end"] = int(max_epochs)
            milestones = s.get("milestones")
            if isinstance(milestones, list) and milestones and all(isinstance(x, int) for x in milestones):
                if len(milestones) == 1:
                    s["milestones"] = [int(max_epochs) - 1]

    return cfg
