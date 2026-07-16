from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TrainConfig:
    images_root: str = os.path.join("datasets", "PhishIntention", "train_imgs")
    coco_json: str = os.path.join("datasets", "PhishIntention", "train_coco.json")
    val_images_root: str = os.path.join("datasets", "PhishIntention", "val_imgs")
    val_coco_json: str = os.path.join("datasets", "PhishIntention", "val_coco.json")

    work_dir: str = "work_dir"

    swin_pretrained: str = os.path.join("swin_base_patch4", "swin_base_patch4_window12_384_22k.pth")

    val_ratio: float = 0.1
    seed: int = 42

    val_is_test: bool = True

    save_best_only: bool = True

    batch_size: int = 2
    num_workers: int = 4

    max_epochs: int = 24
    lr: float = 2e-4
    weight_decay: float = 1e-4

    amp: bool = True
    grad_clip_norm: float = 0.1

    log_interval: int = 200
    val_interval: int = 1

    prefer_swin_b: bool = False
    base_config_path: Optional[str] = None

    use_official_detector_pretrained: bool = True

    multiscale_short_edges: Tuple[int, ...] = (640, 800, 960, 1120, 1280)
    max_size: int = 2000

    tile_infer_size: int = 1024
    tile_overlap: float = 0.25

    classes: Tuple[str, ...] = ("logo",)

    def resolve_paths(self) -> "TrainConfig":
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(here)

        images_root = self.images_root
        coco_json = self.coco_json
        val_images_root = self.val_images_root
        val_coco_json = self.val_coco_json
        work_dir = self.work_dir
        swin_pretrained = self.swin_pretrained

        if not os.path.isabs(images_root):
            images_root = os.path.join(project_root, images_root)
        if not os.path.isabs(coco_json):
            coco_json = os.path.join(project_root, coco_json)
        if not os.path.isabs(val_images_root):
            val_images_root = os.path.join(project_root, val_images_root)
        if not os.path.isabs(val_coco_json):
            val_coco_json = os.path.join(project_root, val_coco_json)
        if not os.path.isabs(work_dir):
            work_dir = os.path.join(here, work_dir)
        if swin_pretrained and (not os.path.isabs(swin_pretrained)):
            swin_pretrained = os.path.join(here, swin_pretrained)

        return TrainConfig(
            images_root=images_root,
            coco_json=coco_json,
            val_images_root=val_images_root,
            val_coco_json=val_coco_json,
            work_dir=work_dir,
            swin_pretrained=swin_pretrained,
            val_ratio=self.val_ratio,
            seed=self.seed,
            val_is_test=self.val_is_test,
            save_best_only=self.save_best_only,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            max_epochs=self.max_epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            amp=self.amp,
            grad_clip_norm=self.grad_clip_norm,
            log_interval=self.log_interval,
            val_interval=self.val_interval,
            prefer_swin_b=self.prefer_swin_b,
            base_config_path=self.base_config_path,
            use_official_detector_pretrained=self.use_official_detector_pretrained,
            multiscale_short_edges=self.multiscale_short_edges,
            max_size=self.max_size,
            tile_infer_size=self.tile_infer_size,
            tile_overlap=self.tile_overlap,
            classes=self.classes,
        )
