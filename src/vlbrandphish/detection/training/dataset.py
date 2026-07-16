from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Sequence, Set, Tuple


def create_train_val_coco_split(
    coco_json_path: str,
    out_train_json: str,
    out_val_json: str,
    val_ratio: float,
    seed: int,
) -> Tuple[str, str]:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images: List[Dict[str, Any]] = list(coco.get("images", []))
    annotations: List[Dict[str, Any]] = list(coco.get("annotations", []))

    if not images:
        raise RuntimeError("COCO images 为空")

    rng = random.Random(int(seed))

    # Stratified split: keep positives/negatives ratio stable in val
    imgid_has_ann: Set[int] = set()
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        imgid_has_ann.add(int(img_id))

    pos_images: List[Dict[str, Any]] = []
    neg_images: List[Dict[str, Any]] = []
    for im in images:
        img_id = im.get("id")
        if img_id is None:
            continue
        if int(img_id) in imgid_has_ann:
            pos_images.append(im)
        else:
            neg_images.append(im)

    rng.shuffle(pos_images)
    rng.shuffle(neg_images)

    n_val_pos = int(round(len(pos_images) * float(val_ratio)))
    n_val_neg = int(round(len(neg_images) * float(val_ratio)))

    if len(pos_images) > 0:
        n_val_pos = max(1, min(n_val_pos, len(pos_images) - 1))
    else:
        n_val_pos = 0

    if len(neg_images) > 0:
        n_val_neg = max(1, min(n_val_neg, len(neg_images) - 1))
    else:
        n_val_neg = 0

    val_images = pos_images[:n_val_pos] + neg_images[:n_val_neg]
    train_images = pos_images[n_val_pos:] + neg_images[n_val_neg:]

    # Final shuffle to avoid any ordering bias
    rng.shuffle(val_images)
    rng.shuffle(train_images)

    val_ids: Set[int] = {int(im["id"]) for im in val_images if "id" in im}
    train_ids: Set[int] = {int(im["id"]) for im in train_images if "id" in im}

    train_anns = [ann for ann in annotations if int(ann.get("image_id", -1)) in train_ids]
    val_anns = [ann for ann in annotations if int(ann.get("image_id", -1)) in val_ids]

    base: Dict[str, Any] = {}
    for k in ("info", "licenses", "categories"):
        if k in coco:
            base[k] = coco[k]

    train_coco = dict(base)
    train_coco["images"] = train_images
    train_coco["annotations"] = train_anns

    val_coco = dict(base)
    val_coco["images"] = val_images
    val_coco["annotations"] = val_anns

    os.makedirs(os.path.dirname(out_train_json), exist_ok=True)

    with open(out_train_json, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)

    with open(out_val_json, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)

    return out_train_json, out_val_json


def patch_dataset_cfg(
    cfg: Any,
    images_root: str,
    train_ann_file: str,
    val_ann_file: str,
    classes: Sequence[str],
    multiscale_short_edges: Sequence[int],
    max_size: int,
    min_size: int,
    train_images_root: str | None = None,
    val_images_root: str | None = None,
) -> Any:
    metainfo = {"classes": tuple(classes)}

    tr_root = train_images_root or images_root
    va_root = val_images_root or images_root

    def _set_pipeline(dset_cfg: Any, is_train: bool) -> None:
        short_edges = list(multiscale_short_edges)
        train_pipeline = [
            {"type": "LoadImageFromFile"},
            {"type": "LoadAnnotations", "with_bbox": True},
            {
                "type": "RandomChoiceResize",
                "scales": [(s, max_size) for s in short_edges],
                "keep_ratio": True,
            },
            {"type": "RandomFlip", "prob": 0.5},
            {"type": "PackDetInputs"},
        ]

        test_pipeline = [
            {"type": "LoadImageFromFile"},
            {"type": "LoadAnnotations", "with_bbox": True},
            {
                "type": "Resize",
                "scale": (max(short_edges), max_size),
                "keep_ratio": True,
            },
            {"type": "PackDetInputs"},
        ]

        dset_cfg["pipeline"] = train_pipeline if is_train else test_pipeline

    train_dl = cfg.get("train_dataloader")
    val_dl = cfg.get("val_dataloader")
    test_dl = cfg.get("test_dataloader")

    if train_dl is None or val_dl is None:
        raise RuntimeError("base config 缺少 train_dataloader/val_dataloader")

    train_dataset = train_dl.get("dataset")
    val_dataset = val_dl.get("dataset")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("base config 的 dataloader 缺少 dataset")

    train_dataset["data_root"] = ""
    train_dataset["data_prefix"] = {"img": tr_root}
    train_dataset["ann_file"] = train_ann_file
    train_dataset["metainfo"] = metainfo
    if isinstance(min_size, int):
        filter_cfg = train_dataset.get("filter_cfg")
        if isinstance(filter_cfg, dict):
            filter_cfg["min_size"] = int(min_size)
        else:
            train_dataset["filter_cfg"] = {"filter_empty_gt": False, "min_size": int(min_size)}
    _set_pipeline(train_dataset, is_train=True)

    val_dataset["data_root"] = ""
    val_dataset["data_prefix"] = {"img": va_root}
    val_dataset["ann_file"] = val_ann_file
    val_dataset["metainfo"] = metainfo
    _set_pipeline(val_dataset, is_train=False)

    if test_dl is not None:
        test_dataset = test_dl.get("dataset")
        if test_dataset is not None:
            test_dataset["data_root"] = ""
            test_dataset["data_prefix"] = {"img": va_root}
            test_dataset["ann_file"] = val_ann_file
            test_dataset["metainfo"] = metainfo
            _set_pipeline(test_dataset, is_train=False)

    return cfg
