from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _tqdm(iterable, *, total: Optional[int] = None, desc: str = ""):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _resolve_clip_training_root() -> str:
    # Resolve repo root by walking up until we find folder named 'CLIP_training'
    p = os.path.abspath(__file__)
    d = os.path.dirname(p)
    while True:
        if os.path.basename(d).lower() == "clip_training":
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    # Fallback: 3 levels up from Detection/ (historical layout)
    here = os.path.dirname(os.path.abspath(__file__))
    det_root = os.path.dirname(here)
    return os.path.abspath(os.path.join(det_root, "..", "..", ".."))


def _try_import_detectron2() -> Tuple[Any, ...]:
    try:
        from detectron2.config import get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        return (
            get_cfg,
            register_coco_instances,
            MetadataCatalog,
            DefaultPredictor,
            COCOEvaluator,
            inference_on_dataset,
            build_detection_test_loader,
        )
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or "detectron2"
        print(
            "Missing dependency: {}\n\n"
            "This script evaluates PhishIntention/PhishPedia logo locator models, which are Detectron2 Faster R-CNN.\n"
            "Please install detectron2 in your current env, then re-run.\n"
            .format(missing)
        )
        raise


def _count_params(model: Any) -> int:
    try:
        params = getattr(model, "parameters", None)
        if params is None:
            return 0
        return int(sum(int(p.numel()) for p in model.parameters()))
    except Exception:
        return 0


def _format_params(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def _extract_maps_from_detectron2_results(results: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    try:
        bbox = results.get("bbox") if isinstance(results, dict) else None
        if isinstance(bbox, dict):
            ap5095 = bbox.get("AP")
            ap50 = bbox.get("AP50")
            ap5095 = float(ap5095) if ap5095 is not None else None
            ap50 = float(ap50) if ap50 is not None else None
            return ap50, ap5095
    except Exception:
        pass
    return None, None


def _print_four_metrics(*, map50: Optional[float], map5095: Optional[float], fps: Optional[float], params: int) -> None:
    map50_s = "nan" if map50 is None else f"{float(map50):.4f}"
    map5095_s = "nan" if map5095 is None else f"{float(map5095):.4f}"
    fps_s = "nan" if fps is None else f"{float(fps):.2f}"
    params_s = _format_params(params)
    print("[summary] mAP@0.5=\t{}".format(map50_s))
    print("[summary] mAP@0.5:0.95=\t{}".format(map5095_s))
    print("[summary] FPS=\t\t{}".format(fps_s))
    print("[summary] Params=\t{}".format(params_s))


@dataclass
class ModelSpec:
    name: str
    cfg_path: str
    weights_path: str
    logo_class_idx: int


def _default_paths(which: str) -> ModelSpec:
    here = os.path.dirname(os.path.abspath(__file__))
    det_root = os.path.dirname(here)
    clip_root = _resolve_clip_training_root()

    if which == "pi":
        pi_root = os.path.abspath(
            os.path.join(
                clip_root,
                "comparative_experiments",
                "2_on_PhishIntention",
                "PhishIntention",

            )
        )
        return ModelSpec(
            name="pi",
            cfg_path=os.path.join(pi_root, "configs", "faster_rcnn_web.yaml"),
            weights_path=os.path.join(pi_root, "models", "layout_detector.pth"),
            logo_class_idx=0,
        )

    if which == "pp":
        pp_root = os.path.abspath(
            os.path.join(
                clip_root,
                "comparative_experiments",
                "2_on_PhishIntention",
                "Phishpedia",
            )
        )
        return ModelSpec(
            name="pp",
            cfg_path=os.path.join(pp_root, "models", "faster_rcnn.yaml"),
            weights_path=os.path.join(pp_root, "models", "rcnn_bet365.pth"),
            logo_class_idx=1,
        )

    if which == "pk":
        pk_root = os.path.abspath(
            os.path.join(
                clip_root,
                "comparative_experiments",
                "1_on_PhishPedia",
                "Phishkiller",
            )
        )
        return ModelSpec(
            name="pk",
            cfg_path=os.path.join(
                clip_root,
                "model_train",
                "logo_detect",
                "Detection",
                "work_dir",
                "patched_config.py",
            ),
            weights_path=os.path.join(pk_root, "models", "detect_6250.pth"),
            logo_class_idx=0,
        )

    raise ValueError(f"unknown which={which}")


def _try_import_pycocotools() -> Tuple[Any, ...]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        return COCO, COCOeval
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or "pycocotools"
        print(
            "Missing dependency: {}\n\n"
            "pk mode uses MMDetection model outputs and evaluates with pycocotools.\n"
            "Please install pycocotools in your current env, then re-run.\n"
            .format(missing)
        )
        raise


def _run_pk_coco_eval(
    *,
    ann_file_logo_only: str,
    images_root: str,
    det_config: str,
    det_checkpoint: str,
    score_thr: float,
    device: str,
    topk: int,
    out_dir: str,
) -> Dict[str, Any]:
    COCO, COCOeval = _try_import_pycocotools()

    # Import Phishkiller MMDet wrapper
    clip_root = _resolve_clip_training_root()
    pk_dir = os.path.abspath(os.path.join(clip_root, "comparative_experiments", "1_on_PhishPedia", "Phishkiller"))
    if pk_dir not in sys.path:
        sys.path.insert(0, pk_dir)
    try:
        from models_infer import LogoDetectorMMDet  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Failed to import Phishkiller/models_infer.py for pk mode. "
            f"err={type(e).__name__}: {e}"
        )

    coco_gt = COCO(ann_file_logo_only)
    img_ids: List[int] = list(coco_gt.getImgIds())
    detector = LogoDetectorMMDet(det_config, det_checkpoint, device=str(device))
    params = _count_params(getattr(detector, "model", None))

    detections: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for img_id in _tqdm(img_ids, total=len(img_ids), desc="pk_infer"):
        img = coco_gt.loadImgs([img_id])[0]
        file_name = str(img.get("file_name", ""))
        image_path = os.path.join(images_root, file_name)
        if not os.path.isfile(image_path):
            # Skip missing images (shouldn't happen if images_root correct)
            continue

        dets_topk = detector.detect_topk_bboxes(image_path, score_thr=float(score_thr), topk=int(topk))
        for bbox_np, sc in dets_topk:
            if bbox_np is None:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox_np.reshape(-1)[:4].tolist()]
            except Exception:
                continue
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue
            detections.append(
                {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(sc),
                }
            )

    elapsed = time.perf_counter() - t0
    fps = float(len(img_ids) / elapsed) if elapsed > 0.0 else None

    os.makedirs(out_dir, exist_ok=True)
    det_path = os.path.join(out_dir, "pk_detections_logo.json")
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(detections, f)

    coco_dt = coco_gt.loadRes(det_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = [float(x) for x in getattr(coco_eval, "stats", [])]
    map5095 = float(stats[0]) if len(stats) > 0 else None
    map50 = float(stats[1]) if len(stats) > 1 else None
    return {
        "det_path": det_path,
        "n_images": len(img_ids),
        "n_detections": len(detections),
        "stats": stats,
        "map50": map50,
        "map5095": map5095,
        "fps": fps,
        "params": int(params),
    }


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError("invalid coco json")
    return obj


def _make_abs_file_name(images_root: str, file_name: str) -> str:
    fn = str(file_name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(images_root, fn))


def _merge_two_coco_logo_only(
    *,
    src1_ann: str,
    src1_images_root: str,
    src2_ann: str,
    src2_images_root: str,
    out_dir: str,
) -> str:
    coco1 = _load_json(src1_ann)
    coco2 = _load_json(src2_ann)

    def _extract_logo_only(coco: Dict[str, Any], images_root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cats = coco.get("categories", [])
        anns = coco.get("annotations", [])
        imgs = coco.get("images", [])
        if not isinstance(cats, list) or not isinstance(anns, list) or not isinstance(imgs, list):
            raise RuntimeError("invalid coco fields")

        keep_id: Optional[int] = None
        for c in cats:
            if not isinstance(c, dict):
                continue
            if str(c.get("name", "")).strip().lower() == "logo":
                try:
                    keep_id = int(c.get("id"))
                except Exception:
                    keep_id = None
                break
        if keep_id is None:
            raise RuntimeError("Cannot find category name='logo' in coco categories")

        out_imgs: List[Dict[str, Any]] = []
        for im in imgs:
            if not isinstance(im, dict):
                continue
            im2 = dict(im)
            im2["file_name"] = _make_abs_file_name(images_root, str(im.get("file_name", "")))
            out_imgs.append(im2)

        out_anns = [a for a in anns if isinstance(a, dict) and int(a.get("category_id", -1)) == keep_id]
        for a in out_anns:
            a["category_id"] = 1
        return out_imgs, out_anns

    imgs1, anns1 = _extract_logo_only(coco1, src1_images_root)
    imgs2, anns2 = _extract_logo_only(coco2, src2_images_root)

    # Build mapping from old image_id to new image_id (per source)
    new_images: List[Dict[str, Any]] = []
    map1: Dict[int, int] = {}
    map2: Dict[int, int] = {}

    next_img_id = 0
    for im in imgs1:
        try:
            old_id = int(im.get("id"))
        except Exception:
            continue
        im2 = dict(im)
        im2["id"] = next_img_id
        map1[old_id] = next_img_id
        new_images.append(im2)
        next_img_id += 1

    for im in imgs2:
        try:
            old_id = int(im.get("id"))
        except Exception:
            continue
        im2 = dict(im)
        im2["id"] = next_img_id
        map2[old_id] = next_img_id
        new_images.append(im2)
        next_img_id += 1

    new_anns: List[Dict[str, Any]] = []
    next_ann_id = 0
    for a in anns1:
        try:
            old_img_id = int(a.get("image_id"))
        except Exception:
            continue
        if old_img_id not in map1:
            continue
        a2 = dict(a)
        a2["id"] = next_ann_id
        a2["image_id"] = map1[old_img_id]
        a2["category_id"] = 1
        new_anns.append(a2)
        next_ann_id += 1

    for a in anns2:
        try:
            old_img_id = int(a.get("image_id"))
        except Exception:
            continue
        if old_img_id not in map2:
            continue
        a2 = dict(a)
        a2["id"] = next_ann_id
        a2["image_id"] = map2[old_img_id]
        a2["category_id"] = 1
        new_anns.append(a2)
        next_ann_id += 1

    merged = {
        "images": new_images,
        "annotations": new_anns,
        "categories": [{"id": 1, "name": "logo"}],
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "merged_logo_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    return out_path


def _write_logo_only_coco(*, src_ann: str, out_dir: str) -> str:
    coco = _load_json(src_ann)
    cats = coco.get("categories", [])
    anns = coco.get("annotations", [])
    imgs = coco.get("images", [])

    if not isinstance(cats, list) or not isinstance(anns, list) or not isinstance(imgs, list):
        raise RuntimeError("invalid coco fields")

    keep_id: Optional[int] = None
    for c in cats:
        if not isinstance(c, dict):
            continue
        if str(c.get("name", "")).strip().lower() == "logo":
            try:
                keep_id = int(c.get("id"))
            except Exception:
                keep_id = None
            break
    if keep_id is None:
        raise RuntimeError("Cannot find category name='logo' in coco categories")

    keep_anns = [a for a in anns if isinstance(a, dict) and int(a.get("category_id", -1)) == keep_id]
    for a in keep_anns:
        a["category_id"] = 1

    out = {
        "images": imgs,
        "annotations": keep_anns,
        "categories": [{"id": 1, "name": "logo"}],
    }

    os.makedirs(out_dir, exist_ok=True)
    ann_stem = os.path.splitext(os.path.basename(src_ann))[0]
    out_path = os.path.join(out_dir, f"{ann_stem}_logo_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out_path


def _build_filtered_model(base_model, *, logo_class_idx: int):
    import torch

    class _Filtered(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, batched_inputs):
            outs = self.m(batched_inputs)
            for o in outs:
                inst = o.get("instances")
                if inst is None:
                    continue
                try:
                    keep = inst.pred_classes == int(logo_class_idx)
                    inst = inst[keep]
                    if len(inst) > 0:
                        inst.pred_classes = inst.pred_classes * 0
                    o["instances"] = inst
                except Exception:
                    continue
            return outs

    return _Filtered(base_model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, choices=["pi", "pp", "pk"], required=True)

    parser.add_argument(
        "--ann-file",
        type=str,
        default=os.path.abspath(
            os.path.join(
                _resolve_clip_training_root(),
                "comparative_experiments",
                "1_on_PhishPedia",
                "datasets",
                "coco_test.json",
            )
        ),
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=os.path.abspath(
            os.path.join(
                _resolve_clip_training_root(),
                "comparative_experiments",
                "1_on_PhishPedia",
                "datasets",
                "benign_sample_30k",
            )
        ),
    )

    parser.add_argument(
        "--merge-two-datasets",
        action="store_true",
        default=True,
        help="Merge current --ann-file/--images-root with PhishIntention val set into a single test set (default: enabled).",
    )
    parser.add_argument(
        "--no-merge-two-datasets",
        action="store_false",
        dest="merge_two_datasets",
        help="Disable merging datasets; evaluate only --ann-file/--images-root.",
    )

    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--logo-class-idx", type=int, default=None)

    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=100, help="(pk mode) keep top-k detections per image")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    )

    args = parser.parse_args()

    spec = _default_paths(str(args.which))
    cfg_path = os.path.abspath(str(args.cfg or spec.cfg_path))
    weights_path = os.path.abspath(str(args.weights or spec.weights_path))
    logo_class_idx = int(spec.logo_class_idx if args.logo_class_idx is None else int(args.logo_class_idx))

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    ann_file = os.path.abspath(str(args.ann_file))
    images_root = os.path.abspath(str(args.images_root))
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"ann_file not found: {ann_file}")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"images_root not found: {images_root}")

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    if bool(getattr(args, "merge_two_datasets", True)):
        clip_root = _resolve_clip_training_root()
        pi_ann = os.path.abspath(
            os.path.join(
                clip_root,
                "model_train",
                "logo_detect",
                "Detection",
                "datasets",
                "PhishIntention",
                "val_coco.json",
            )
        )
        pi_imgs = os.path.abspath(
            os.path.join(
                clip_root,
                "model_train",
                "logo_detect",
                "Detection",
                "datasets",
                "PhishIntention",
                "val_imgs",
            )
        )
        if not os.path.isfile(pi_ann) or not os.path.isdir(pi_imgs):
            raise FileNotFoundError(f"PhishIntention val set not found: ann={pi_ann} imgs={pi_imgs}")

        logo_only_ann = _merge_two_coco_logo_only(
            src1_ann=ann_file,
            src1_images_root=images_root,
            src2_ann=pi_ann,
            src2_images_root=pi_imgs,
            out_dir=out_dir,
        )
        images_root = os.path.abspath(_resolve_clip_training_root())
    else:
        logo_only_ann = _write_logo_only_coco(src_ann=ann_file, out_dir=out_dir)

    if spec.name == "pk":
        results = _run_pk_coco_eval(
            ann_file_logo_only=logo_only_ann,
            images_root=images_root,
            det_config=cfg_path,
            det_checkpoint=weights_path,
            score_thr=float(args.score_thr),
            device=str(args.device),
            topk=int(args.topk),
            out_dir=out_dir,
        )

        print(f"[done] which={spec.name} cfg={cfg_path} weights={weights_path}")
        print(f"[done] score_thr={float(args.score_thr):.4f} topk={int(args.topk)}")
        print("[coco] ann_file_logo_only:", logo_only_ann)
        print("[metrics]", results)
        _print_four_metrics(
            map50=results.get("map50"),
            map5095=results.get("map5095"),
            fps=results.get("fps"),
            params=int(results.get("params") or 0),
        )
        return

    (
        get_cfg,
        register_coco_instances,
        MetadataCatalog,
        DefaultPredictor,
        COCOEvaluator,
        inference_on_dataset,
        build_detection_test_loader,
    ) = _try_import_detectron2()

    dataset_name = f"phishintention_val_logo__{spec.name}"
    register_coco_instances(dataset_name, {}, logo_only_ann, images_root)

    # Ensure evaluator knows there is only 1 class: 'logo'
    MetadataCatalog.get(dataset_name).thing_classes = ["logo"]

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.score_thr)
    cfg.MODEL.DEVICE = str(args.device)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 0

    predictor = DefaultPredictor(cfg)
    predictor.model = _build_filtered_model(predictor.model, logo_class_idx=logo_class_idx)

    evaluator = COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=out_dir)
    loader = build_detection_test_loader(cfg, dataset_name)
    t0 = time.perf_counter()
    results = inference_on_dataset(predictor.model, loader, evaluator)
    elapsed = time.perf_counter() - t0
    n_images = None
    try:
        n_images = int(len(loader.dataset))
    except Exception:
        n_images = None
    fps = float(n_images / elapsed) if (n_images is not None and elapsed > 0.0) else None
    params = _count_params(getattr(predictor, "model", None))
    map50, map5095 = _extract_maps_from_detectron2_results(results)

    print(f"[done] which={spec.name} cfg={cfg_path} weights={weights_path}")
    print(f"[done] logo_class_idx={logo_class_idx} score_thr={float(args.score_thr):.4f}")
    print("[coco] ann_file_logo_only:", logo_only_ann)
    print("[metrics]", results)
    _print_four_metrics(map50=map50, map5095=map5095, fps=fps, params=int(params))


if __name__ == "__main__":
    main()
