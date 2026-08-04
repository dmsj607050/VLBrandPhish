from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DetBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: int

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _iou(a: DetBox, b: DetBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: Sequence[DetBox], iou_thr: float) -> List[DetBox]:
    boxes = sorted(list(boxes), key=lambda b: b.score, reverse=True)
    keep: List[DetBox] = []
    for b in boxes:
        suppressed = False
        for k in keep:
            if _iou(b, k) >= iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(b)
    return keep


def _load_image_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im)


def iter_tiles(image: np.ndarray, tile_size: int, overlap: float) -> Sequence[Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    step = max(1, int(round(tile_size * (1.0 - float(overlap)))))

    xs = list(range(0, max(1, w - tile_size + 1), step))
    ys = list(range(0, max(1, h - tile_size + 1), step))

    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))

    for y in ys:
        for x in xs:
            x1 = int(x)
            y1 = int(y)
            x2 = int(min(w, x + tile_size))
            y2 = int(min(h, y + tile_size))
            yield x1, y1, x2, y2


def _parse_mmdet_result(result: Any, score_thr: float) -> List[DetBox]:
    """Parse ``inference_detector`` output, supporting both MMDet 2.x and 3.x.

    MMDet 3.x returns ``DetDataSample`` objects with ``.pred_instances``;
    2.x returns a list of numpy arrays (one per class).  Both paths are
    normalised into ``DetBox`` objects in image coordinates.
    """

    boxes: List[DetBox] = []

    # MMDet 3.x: DetDataSample
    if hasattr(result, "pred_instances"):
        inst = result.pred_instances
        bboxes = getattr(inst, "bboxes", None)
        scores = getattr(inst, "scores", None)
        labels = getattr(inst, "labels", None)
        if bboxes is None or scores is None or labels is None:
            return []
        b = bboxes.detach().cpu().numpy()
        s = scores.detach().cpu().numpy()
        l = labels.detach().cpu().numpy()
        for (x1, y1, x2, y2), sc, lb in zip(b, s, l):
            if float(sc) < float(score_thr):
                continue
            boxes.append(DetBox(float(x1), float(y1), float(x2), float(y2), float(sc), int(lb)))
        return boxes

    # MMDet 2.x: list[np.ndarray] per class, or tuple(bbox_result, segm_result)
    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, list):
        for cls_id, arr in enumerate(result):
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            # arr: [N,5] => x1,y1,x2,y2,score
            for row in arr:
                if row.shape[0] < 5:
                    continue
                x1, y1, x2, y2, sc = row[:5].tolist()
                if float(sc) < float(score_thr):
                    continue
                boxes.append(DetBox(float(x1), float(y1), float(x2), float(y2), float(sc), int(cls_id)))
        return boxes

    return []


def draw_box(image: np.ndarray, box: DetBox, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in (box.x1, box.y1, box.x2, box.y2)]
    x1 = max(0, min(x1, out.shape[1] - 1))
    x2 = max(0, min(x2, out.shape[1] - 1))
    y1 = max(0, min(y1, out.shape[0] - 1))
    y2 = max(0, min(y2, out.shape[0] - 1))

    out[y1 : y1 + thickness, x1:x2] = color
    out[y2 - thickness : y2, x1:x2] = color
    out[y1:y2, x1 : x1 + thickness] = color
    out[y1:y2, x2 - thickness : x2] = color
    return out


def _pick_best_checkpoint(config_path: str) -> str:
    # Prefer checkpoints living next to the config (e.g., work_dir/patched_config.py)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    candidates_dirs = [
        config_dir,
        os.path.join(config_dir, "work_dir"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_dir"),
        os.path.join(os.getcwd(), "work_dir"),
    ]

    matches: List[str] = []
    for d in candidates_dirs:
        if not os.path.isdir(d):
            continue
        matches.extend(glob.glob(os.path.join(d, "best_coco_bbox_mAP_epoch_*.pth")))

    if not matches:
        raise FileNotFoundError(
            "No best checkpoint found (best_coco_bbox_mAP_epoch_*.pth). "
            "Pass --checkpoint /path/to/xxx.pth explicitly."
        )

    def _epoch_num(p: str) -> int:
        base = os.path.basename(p)
        # best_coco_bbox_mAP_epoch_2.pth
        try:
            n = base.split("epoch_")[-1].split(".pth")[0]
            return int(n)
        except Exception:
            return -1

    matches = sorted(matches, key=_epoch_num, reverse=True)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="mmdet config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint .pth path (default: auto pick best_*.pth from work_dir)")
    parser.add_argument("--image", type=str, required=True, help="input image path")

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)

    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)

    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-vis", type=str, default=None)

    args = parser.parse_args()

    from mmdet.apis import inference_detector, init_detector

    ckpt = args.checkpoint
    if ckpt is None or str(ckpt).strip() == "":
        ckpt = _pick_best_checkpoint(args.config)
        print(f"[infer_tile] auto checkpoint: {ckpt}")

    model = init_detector(args.config, ckpt, device=args.device)

    img = _load_image_rgb(args.image)
    h, w = img.shape[:2]

    all_boxes: List[DetBox] = []
    for x1, y1, x2, y2 in iter_tiles(img, tile_size=int(args.tile), overlap=float(args.overlap)):
        tile = img[y1:y2, x1:x2]
        result = inference_detector(model, tile)
        boxes = _parse_mmdet_result(result, score_thr=float(args.score_thr))
        # map to original coords
        for b in boxes:
            all_boxes.append(
                DetBox(
                    x1=b.x1 + float(x1),
                    y1=b.y1 + float(y1),
                    x2=b.x2 + float(x1),
                    y2=b.y2 + float(y1),
                    score=b.score,
                    label=b.label,
                )
            )

    kept = nms(all_boxes, iou_thr=float(args.nms_iou))
    best = kept[0] if kept else None

    output: Dict[str, Any] = {
        "image": os.path.abspath(args.image),
        "width": int(w),
        "height": int(h),
        "tile": int(args.tile),
        "overlap": float(args.overlap),
        "score_thr": float(args.score_thr),
        "nms_iou": float(args.nms_iou),
        "num_candidates": int(len(all_boxes)),
        "num_after_nms": int(len(kept)),
        "best": None,
    }

    if best is not None:
        output["best"] = {
            "bbox_xyxy": [best.x1, best.y1, best.x2, best.y2],
            "score": float(best.score),
            "label": int(best.label),
        }

    out_json = args.out_json
    if out_json is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_json = os.path.join(os.path.dirname(args.image), f"{stem}_logo_bbox.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.out_vis is not None:
        vis = img
        if best is not None:
            vis = draw_box(vis, best)
        from PIL import Image

        Image.fromarray(vis).save(args.out_vis)

    print(f"✅ done: {out_json}")
    if best is not None:
        print(f"best bbox xyxy: {output['best']['bbox_xyxy']} score={output['best']['score']:.4f}")
    else:
        print("no detection")


if __name__ == "__main__":
    main()
