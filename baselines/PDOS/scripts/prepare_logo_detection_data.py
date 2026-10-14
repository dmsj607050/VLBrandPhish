"""Convert the supplied sources to one-class YOLOv5 manifests.

Train:   PhishIntention Logo_train/train (8,109)
Default: PhishIntention Logo_train/val (901)
Merged:  Default + Phishpedia coco_test (2,480)
New:     VLBrandPhish 01_phish_12K (12,801)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pi = resolve_source_dir(args.phishintention.resolve(), ("Logo_train", "logo_train"))
    vlbp = resolve_source_dir(args.vlbrandphish.resolve(), ("01_phish_12K", "phish_12K"))

    splits = {
        "train": convert_coco_split(
            "pi_train", pi / "train_imgs", pi / "train_coco.json",
            output, args.link_mode, phishpedia=False
        ),
        "default": convert_coco_split(
            "pi_default", pi / "val_imgs", pi / "val_coco.json",
            output, args.link_mode, phishpedia=False
        ),
        "phishpedia": [],
        "new": convert_vlbp(vlbp, output, args.link_mode),
    }
    if not args.skip_phishpedia:
        pp = args.phishpedia.resolve()
        splits["phishpedia"] = convert_coco_split(
            "phishpedia", pp, pp / "coco_test.json", output, args.link_mode, phishpedia=True
        )
    splits["merged"] = merge_splits(splits["default"], splits["phishpedia"], output, args.link_mode)

    dataset = {
        "path": str(output),
        "train": "train/images",
        "val": "default/images",
        "test": "default/images",
        "nc": 1,
        "names": ["logo"],
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    manifest = {"dataset": dataset, "splits": splits}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({name: len(items) for name, items in splits.items()}, indent=2))


def convert_coco_split(prefix, image_root, annotation_file, output, link_mode, phishpedia):
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    logo_ids = {c["id"] for c in data["categories"] if c["name"].lower() == "logo"}
    annotations = {}
    for ann in data["annotations"]:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if ann.get("category_id") in logo_ids and bbox[2] > 0 and bbox[3] > 0:
            annotations.setdefault(ann["image_id"], []).append(ann)

    records = []
    for image in data["images"]:
        source = resolve_phishpedia_path(Path(image_root), image["file_name"]) if phishpedia else Path(image_root) / image["file_name"]
        if not source.exists():
            raise FileNotFoundError(f"COCO image missing: {source}")
        name = f"{prefix}_{image['id']:06d}_{Path(image['file_name']).stem}{source.suffix.lower()}"
        records.append(write_annotation_record(
            split_for_prefix(prefix), name, source, image["width"], image["height"],
            annotations.get(image["id"], []), output, link_mode
        ))
    return records


def convert_vlbp(root, output, link_mode):
    records = []
    for brand_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        screens, labels = brand_dir / "screens", brand_dir / "labels"
        for source in sorted(p for p in screens.iterdir() if p.is_file()):
            xml_path = labels / f"{source.stem}.xml"
            if not xml_path.exists():
                raise FileNotFoundError(f"VLBrandPhish label missing: {xml_path}")
            root_node = ET.parse(xml_path).getroot()
            size = root_node.find("size")
            width, height = int(size.findtext("width")), int(size.findtext("height"))
            boxes = []
            for obj in root_node.findall("object"):
                box = obj.find("bndbox")
                xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
                xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
                boxes.append({"bbox": [xmin, ymin, xmax - xmin, ymax - ymin]})
            name = f"vlbp_{safe_name(brand_dir.name)}_{source.stem}{source.suffix.lower()}"
            records.append(write_annotation_record("new", name, source, width, height, boxes, output, link_mode))
    return records


def merge_splits(first, second, output, link_mode):
    records = []
    for record in first + second:
        source = Path(record["image"])
        name = f"merged_{source.name}"
        records.append(write_yolo_record(
            "merged", name, source, Path(record["label"]), output, link_mode,
            record["width"], record["height"]
        ))
    return records


def write_annotation_record(split, name, source, width, height, annotations, output, link_mode):
    image_target, label_target = targets(output, split, name)
    link_or_copy(source, image_target, link_mode)
    lines = []
    for ann in annotations:
        x, y, w, h = (float(v) for v in ann["bbox"])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        if x2 > x1 and y2 > y1:
            lines.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"image": str(image_target), "label": str(label_target), "width": width, "height": height, "boxes": len(lines)}


def write_yolo_record(split, name, source, label_source, output, link_mode, width, height):
    image_target, label_target = targets(output, split, name)
    link_or_copy(source, image_target, link_mode)
    shutil.copy2(label_source, label_target)
    count = len(label_target.read_text(encoding="utf-8").splitlines())
    return {"image": str(image_target), "label": str(label_target), "width": width, "height": height, "boxes": count}


def targets(output, split, name):
    image_target = output / split / "images" / name
    label_target = output / split / "labels" / f"{Path(name).stem}.txt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    return image_target, label_target


def link_or_copy(source, target, mode):
    if target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    if mode == "symlink":
        try:
            target.symlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def split_for_prefix(prefix):
    return {"pi_train": "train", "pi_default": "default", "phishpedia": "phishpedia"}[prefix]


def resolve_phishpedia_path(root, file_name):
    direct = root / file_name
    candidates = [
        direct,
        root / "benign_sample_30k" / file_name,
        root / "phish_sample_30k" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return direct


def safe_name(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")[:80]


def resolve_source_dir(root, candidates):
    """Accept either a dataset parent or the actual split directory."""
    if (root / "train_imgs").exists() or (root / "screens").exists():
        return root
    if root.exists() and any((child / "screens").is_dir() and (child / "labels").is_dir() for child in root.iterdir() if child.is_dir()):
        return root
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"Cannot find dataset under {root}; expected one of: {expected}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phishintention", type=Path, default=REPO_ROOT / "data" / "PhishIntention")
    parser.add_argument("--phishpedia", type=Path, default=REPO_ROOT / "data" / "PhishPedia")
    parser.add_argument("--vlbrandphish", type=Path, default=REPO_ROOT / "data" / "VLBrandPhish")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "baselines" / "PDOS" / "data" / "prepared_logo_detection")
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--skip-phishpedia", action="store_true", help="Prepare only Default/New when Phishpedia is unavailable")
    return parser.parse_args()


if __name__ == "__main__":
    main()
