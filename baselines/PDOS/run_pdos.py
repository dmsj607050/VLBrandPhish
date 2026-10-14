"""Command-line entry point for Phase 1 PDOS inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdos.detector import YOLOv5LogoDetector
from pdos.domain_verification import LegalDomainDB
from pdos.feature_extractor import ResNet101FeatureExtractor
from pdos.legal_logo_db import LegalLogoDB
from pdos.pipeline import run_pdos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PDOS reference-based phishing detector")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo" / "yolov5s.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pdos_yolov5s_se.yaml")
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--brand-search", choices=["category", "global"], default="category")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--resnet-weights",
        type=Path,
        default=ROOT / "models" / "resnet101-cd907fc2.pth",
        help="Local ImageNet ResNet-101 weights; no network download is attempted by the CLI.",
    )
    parser.add_argument("--logo-db", type=Path, default=ROOT / "data" / "legal_logo_db")
    parser.add_argument("--domain-db", type=Path, default=ROOT / "data" / "legal_domain_db" / "domains.json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YOLOv5LogoDetector(
        ROOT, args.weights, args.device, args.confidence, args.iou,
        config=args.config, class_names=args.class_names
    )
    extractor = ResNet101FeatureExtractor(args.device, pretrained=True, weights_path=args.resnet_weights)
    result = run_pdos(
        args.image, args.url, detector, extractor, LegalLogoDB(args.logo_db), LegalDomainDB(args.domain_db),
        args.threshold, args.brand_search
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
