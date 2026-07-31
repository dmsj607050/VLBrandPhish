from __future__ import annotations

import argparse
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


# ----------------------------
# Default paths for easy edits
# ----------------------------
DEFAULT_CONFIG_PATH = "~/Phishlogo/model_train/logo_detect/Detection/work_dir/patched_config.py"
DEFAULT_CHECKPOINT_PATH = "~/Phishlogo/comparative_experiments_B/Phishkiller/models/detect_6250.pth"
DEFAULT_INPUT_ROOT = "~/Phishlogo/comparative_experiments_B/Test-B/bengin_12K"
DEFAULT_OUTPUT_ROOT = "~/Phishlogo/comparative_experiments_B/Test-B/bengin_12K_logo_xml"
DEFAULT_DEVICE = "cuda:0"

# Detection / export settings
DEFAULT_SCORE_THR = 0.10
DEFAULT_NMS_IOU_THR = 0.50
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 0.25
DEFAULT_USE_TILED_INFERENCE = True
DEFAULT_XML_CLASS_NAME = "logo"
DEFAULT_SAVE_EMPTY_XML = True
DEFAULT_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


try:
    from mmdet.apis import inference_detector, init_detector
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "unknown")
    raise SystemExit(
        "Missing dependency: {}\n"
        "Please activate the environment that contains MMDetection runtime packages "
        "(mmengine, mmcv, mmdet).".format(missing)
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the logo detector on all images under a directory and export Pascal VOC XML files."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU_THR)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-overlap", type=float, default=DEFAULT_TILE_OVERLAP)
    parser.add_argument("--class-name", type=str, default=DEFAULT_XML_CLASS_NAME)
    parser.add_argument("--save-empty-xml", action="store_true", default=DEFAULT_SAVE_EMPTY_XML)
    parser.add_argument("--no-save-empty-xml", dest="save_empty_xml", action="store_false")
    parser.add_argument("--use-tiling", action="store_true", default=DEFAULT_USE_TILED_INFERENCE)
    parser.add_argument("--no-tiling", dest="use_tiling", action="store_false")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_SUFFIXES),
        help="Image file suffixes to search recursively, e.g. .png .jpg .jpeg",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_image_paths(root: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    normalized = {s.lower() if s.startswith(".") else "." + s.lower() for s in suffixes}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized:
            yield path


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def get_image_size(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
        depth = len(image.getbands())
    return width, height, depth


def iter_tiles(image: np.ndarray, tile_size: int, overlap: float) -> Iterable[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    step = max(1, int(round(tile_size * (1.0 - float(overlap)))))

    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))

    last_x = max(0, width - tile_size)
    last_y = max(0, height - tile_size)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)

    for y in ys:
        for x in xs:
            x1 = int(x)
            y1 = int(y)
            x2 = int(min(width, x + tile_size))
            y2 = int(min(height, y + tile_size))
            yield x1, y1, x2, y2


def parse_mmdet_result(result: Any, score_thr: float) -> List[DetBox]:
    boxes: List[DetBox] = []

    if hasattr(result, "pred_instances"):
        pred = result.pred_instances
        bboxes = getattr(pred, "bboxes", None)
        scores = getattr(pred, "scores", None)
        labels = getattr(pred, "labels", None)
        if bboxes is None or scores is None or labels is None:
            return boxes

        bbox_array = bboxes.detach().cpu().numpy()
        score_array = scores.detach().cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        for (x1, y1, x2, y2), score, label in zip(bbox_array, score_array, label_array):
            if float(score) < float(score_thr):
                continue
            boxes.append(DetBox(float(x1), float(y1), float(x2), float(y2), float(score), int(label)))
        return boxes

    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, list):
        for class_idx, arr in enumerate(result):
            arr = np.asarray(arr)
            if arr.size == 0:
                continue
            for row in arr:
                if len(row) < 5:
                    continue
                x1, y1, x2, y2, score = row[:5].tolist()
                if float(score) < float(score_thr):
                    continue
                boxes.append(DetBox(float(x1), float(y1), float(x2), float(y2), float(score), int(class_idx)))
    return boxes


def iou(a: DetBox, b: DetBox) -> float:
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
    kept: List[DetBox] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if all(iou(box, prev) < iou_thr for prev in kept):
            kept.append(box)
    return kept


def infer_single_image(
    model: Any,
    image_path: Path,
    use_tiling: bool,
    tile_size: int,
    tile_overlap: float,
    score_thr: float,
    nms_iou: float,
) -> List[DetBox]:
    if not use_tiling:
        result = inference_detector(model, str(image_path))
        return nms(parse_mmdet_result(result, score_thr=score_thr), iou_thr=nms_iou)

    image = load_image_rgb(image_path)
    all_boxes: List[DetBox] = []
    for x1, y1, x2, y2 in iter_tiles(image, tile_size=tile_size, overlap=tile_overlap):
        tile = image[y1:y2, x1:x2]
        result = inference_detector(model, tile)
        for box in parse_mmdet_result(result, score_thr=score_thr):
            all_boxes.append(
                DetBox(
                    x1=box.x1 + x1,
                    y1=box.y1 + y1,
                    x2=box.x2 + x1,
                    y2=box.y2 + y1,
                    score=box.score,
                    label=box.label,
                )
            )
    return nms(all_boxes, iou_thr=nms_iou)


def clamp_box(box: DetBox, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(box.x1))))
    y1 = max(0, min(height - 1, int(round(box.y1))))
    x2 = max(0, min(width - 1, int(round(box.x2))))
    y2 = max(0, min(height - 1, int(round(box.y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def build_voc_xml(
    image_path: Path,
    image_root: Path,
    width: int,
    height: int,
    depth: int,
    boxes: Sequence[DetBox],
    class_name: str,
) -> ET.ElementTree:
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    try:
        folder.text = str(image_path.parent.relative_to(image_root))
    except ValueError:
        folder.text = image_path.parent.name

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_path.name

    path_elem = ET.SubElement(annotation, "path")
    path_elem.text = str(image_path)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for box in boxes:
        x1, y1, x2, y2 = clamp_box(box, width=width, height=height)
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "confidence").text = "{:.6f}".format(float(box.score))

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)

    indent_xml(annotation)
    return ET.ElementTree(annotation)


def resolve_output_xml_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = image_path.relative_to(input_root)
    return output_root / relative.parent / (image_path.stem + ".xml")


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit("Input root does not exist: {}".format(input_root))
    if not config_path.is_file():
        raise SystemExit("Config file does not exist: {}".format(config_path))
    if not checkpoint_path.is_file():
        raise SystemExit("Checkpoint file does not exist: {}".format(checkpoint_path))

    image_paths = sorted(iter_image_paths(input_root, args.extensions))
    if not image_paths:
        raise SystemExit("No images found under: {}".format(input_root))

    print("[1/3] Loading detector...")
    model = init_detector(str(config_path), str(checkpoint_path), device=args.device)

    print("[2/3] Found {} images under {}".format(len(image_paths), input_root))
    written = 0
    failed = 0
    empty = 0

    progress_iter: Iterable[Path]
    if tqdm is not None:
        progress_iter = tqdm(image_paths, desc="Exporting XML", unit="img")
    else:
        progress_iter = image_paths

    for index, image_path in enumerate(progress_iter, start=1):
        try:
            boxes = infer_single_image(
                model=model,
                image_path=image_path,
                use_tiling=bool(args.use_tiling),
                tile_size=int(args.tile_size),
                tile_overlap=float(args.tile_overlap),
                score_thr=float(args.score_thr),
                nms_iou=float(args.nms_iou),
            )

            if not boxes:
                empty += 1
                if not args.save_empty_xml:
                    if index % 100 == 0 or index == len(image_paths):
                        print("[{}/{}] {} -> no detections".format(index, len(image_paths), image_path.name))
                    continue

            width, height, depth = get_image_size(image_path)
            xml_tree = build_voc_xml(
                image_path=image_path,
                image_root=input_root,
                width=width,
                height=height,
                depth=depth,
                boxes=boxes,
                class_name=args.class_name,
            )
            xml_path = resolve_output_xml_path(image_path, input_root=input_root, output_root=output_root)
            ensure_parent_dir(xml_path)
            xml_tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            written += 1

            if tqdm is not None:
                progress_iter.set_postfix(written=written, empty=empty, failed=failed)
            elif index % 100 == 0 or index == len(image_paths):
                print(
                    "[{}/{}] {} -> {} boxes".format(
                        index,
                        len(image_paths),
                        image_path.name,
                        len(boxes),
                    )
                )
        except Exception:
            failed += 1
            if tqdm is not None:
                progress_iter.set_postfix(written=written, empty=empty, failed=failed)
            print("[ERROR] Failed on {}".format(image_path))
            traceback.print_exc()

    print("[3/3] Finished.")
    print("XML written   :", written)
    print("Empty images  :", empty)
    print("Failed images :", failed)
    print("Output root   :", output_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
