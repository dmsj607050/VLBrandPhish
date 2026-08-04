"""Experiment 4: manifest-driven adversarial robustness evaluation.

The runner supports the prepared six PhishEval perturbation attacks already present
locally and can also construct the visible-manipulation protocol directly from the
official PhishEval directory.  Every selected screenshot is evaluated by the same
Stage-1 detector and fixed 0.8/0.2 Stage-2 decision as the other experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._shared.e2e import (
    _first_existing,
    add_e2e_arguments,
    execute_experiment,
    limit_by_class,
)
from inference.components.brand.naming import normalize_brand_key


DEFAULT_PREPARED_ROOT = _first_existing(
    os.environ.get("VLBRANDPHISH_PHISHEVAL_ROOT", ""),
    REPO_ROOT / "data" / "PhishEval" / "inputs",
)
DEFAULT_PERTURB_ROOT = REPO_ROOT / "data" / "PhishEval" / "perturbated_dataset"
DEFAULT_VISIBLE_ROOT = REPO_ROOT / "data" / "PhishEval" / "visible_dataset2"
DEFAULT_CRAWL_ROOT = REPO_ROOT / "data" / "PhishEval" / "crawl_benign"
DEFAULT_LOGO_LIBRARY_DIR = REPO_ROOT / "data" / "PhishEval" / "expand277"
DEFAULT_NORMAL_URLS_CSV = _first_existing(
    os.environ.get("VLBRANDPHISH_NORMAL_URLS_CSV", ""),
    REPO_ROOT / "data" / "PhishEval" / "normal_urls_expand277.csv",
)

VISIBLE_ATTACKS = {
    "CRP": "original",
    "00": "elimination",
    "01": "colorreplace",
    "02": "resizing",
    "03": "rotation",
    "04": "integration",
    "05": "reposition",
    "06": "flipping",
    "07": "replacement",
    "08": "blurring",
    "09": "scaling",
    "10": "omission",
    "11": "fontreplace",
    "12": "caseconversion",
}
RESOURCE_DIRS = {"Fonts", "LogoLabelStudio"}
SQUAT_TEMPLATES = (
    ("prefix_secure", "secure-{sld}.{suffix}"),
    ("suffix_secure", "{sld}-secure.{suffix}"),
    ("suffix_login", "{sld}-login.{suffix}"),
    ("suffix_verify", "{sld}-verify.{suffix}"),
    ("suffix_account", "{sld}-account.{suffix}"),
    ("concat_login", "{sld}login.{suffix}"),
    ("concat_verify", "{sld}verify.{suffix}"),
    ("suffix_support", "{sld}-support.{suffix}"),
    ("prefix_my", "my{sld}.{suffix}"),
    ("concat_online", "{sld}online.{suffix}"),
    ("suffix_service", "{sld}-service.{suffix}"),
    ("suffix_auth", "{sld}-auth.{suffix}"),
)
MULTI_PART_SUFFIXES = {
    "ac.uk",
    "co.jp",
    "co.kr",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.mx",
    "com.sg",
    "co.in",
    "gov.uk",
    "net.cn",
    "org.cn",
    "org.uk",
}


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _reference_brand_aliases(dataset_root: str) -> dict[str, str]:
    """Map PhishEval/277 aliases conservatively onto the 152 official references.

    Returns an empty dict when the official dataset root is unavailable so that
    ``--keep-unmapped`` can retain all brands with their raw names.
    """
    root = Path(dataset_root)
    reference_root = root / "03_logo_reference_library" / "logo_reference_library"
    if not reference_root.is_dir():
        return {}
    references = [path.name for path in reference_root.iterdir() if path.is_dir()]
    aliases = {normalize_brand_key(name): name for name in references}
    split_path = root / "05_brands_overlap.csv"
    if not split_path.is_file():
        return aliases
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = str(row.get("brand_dir_7299", "") or "").strip()
            if target not in references:
                continue
            for column in ("brand_dir_7299", "canonical_name_7299", "match_277_dirs"):
                raw = str(row.get(column, "") or "").strip()
                if raw:
                    aliases.setdefault(normalize_brand_key(raw), target)
    return aliases


def _canonical_brand(raw: str, aliases: dict[str, str]) -> str:
    key = normalize_brand_key(raw)
    if key in aliases:
        return aliases[key]
    return aliases.get(_compact_key(raw), "")


def _read_csv_records(
    path: Path, *, input_root: Path, aliases: dict[str, str], keep_unmapped: bool
) -> tuple[list[dict[str, str]], int]:
    records: list[dict[str, str]] = []
    skipped = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    for index, row in enumerate(source_rows):
        raw_brand = str(row.get("brand", "") or "").strip()
        canonical = _canonical_brand(raw_brand, aliases)
        if not canonical and not keep_unmapped:
            skipped += 1
            continue
        image = input_root / str(row.get("scr_path", "") or "")
        if not image.is_file():
            image = Path(str(row.get("source_image", "") or ""))
        if not image.is_file():
            skipped += 1
            continue
        url_type = str(row.get("url_type", "") or "").strip().lower()
        is_phish = url_type == "squatted"
        attack = str(row.get("attack", "") or "").strip().lower() or "unknown"
        rule = str(row.get("rule_id", "") or "").strip() or "unknown"
        sample_key = f"perturb/{attack}/{raw_brand}/{url_type}/{rule}/{index:06d}"
        records.append(
            {
                "sample_key": sample_key,
                "image_path": str(image.resolve()),
                "url": str(row.get("domain", "") or ""),
                "gt_is_phishing": "1" if is_phish else "0",
                "gt_brand": canonical or raw_brand,
                "dataset_split": "phisheval_perturb",
                "attack": attack,
                "url_type": url_type,
                "rule_id": rule,
                "source_manifest": str(path.resolve()),
                "source_brand": raw_brand,
            }
        )
    return records, skipped


def _registered_parts(url: str) -> tuple[str, str]:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.netloc or parsed.path.split("/")[0]).split("@")[-1].split(":")[0].strip(".").lower()
    parts = [part for part in host.split(".") if part]
    if len(parts) < 2:
        return "", ""
    suffix = ".".join(parts[-2:])
    if suffix in MULTI_PART_SUFFIXES and len(parts) >= 3:
        return parts[-3], suffix
    return parts[-2], parts[-1]


def _squatted_urls(benign_url: str) -> list[tuple[str, str]]:
    sld, suffix = _registered_parts(benign_url)
    if not sld or not suffix:
        return []
    return [(rule, f"https://{template.format(sld=sld, suffix=suffix)}/") for rule, template in SQUAT_TEMPLATES]


def _crawl_lookup(crawl_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in sorted(crawl_root.iterdir()):
        if path.is_dir():
            lookup.setdefault(_compact_key(path.name), path)
    return lookup


def _visible_records(
    *,
    visible_root: Path,
    crawl_root: Path,
    aliases: dict[str, str],
    keep_unmapped: bool,
) -> tuple[list[dict[str, str]], int]:
    """Create original and 12 squatted URL records for every visible-manipulation page."""
    if not visible_root.is_dir() or not crawl_root.is_dir():
        raise FileNotFoundError("Visible PhishEval source requires both --visible-root and --crawl-root.")
    records: list[dict[str, str]] = []
    skipped = 0
    crawl = _crawl_lookup(crawl_root)
    for brand_dir in sorted(visible_root.iterdir()):
        if not brand_dir.is_dir() or not brand_dir.name.startswith("brand_"):
            continue
        raw_brand = brand_dir.name.removeprefix("brand_")
        canonical = _canonical_brand(raw_brand, aliases)
        if not canonical and not keep_unmapped:
            skipped += 1
            continue
        crawl_dir = crawl.get(_compact_key(raw_brand))
        login_path = crawl_dir / "login.txt" if crawl_dir else Path()
        if not crawl_dir or not login_path.is_file():
            skipped += 1
            continue
        benign_url = login_path.read_text(encoding="utf-8", errors="ignore").strip()
        squat_urls = _squatted_urls(benign_url)
        if not benign_url or not squat_urls:
            skipped += 1
            continue
        for source_dir in sorted(brand_dir.iterdir()):
            attack = VISIBLE_ATTACKS.get(source_dir.name, "")
            if not attack or source_dir.name in RESOURCE_DIRS:
                continue
            for image in sorted(source_dir.glob("*.png")):
                if image.stem.endswith("_crop_logo"):
                    continue
                base = {
                    "image_path": str(image.resolve()),
                    "gt_brand": canonical or raw_brand,
                    "dataset_split": "phisheval_visible",
                    "attack": attack,
                    "source_brand": raw_brand,
                    "source_subdir": source_dir.name,
                }
                sample_base = f"visible/{attack}/{raw_brand}/{image.stem}"
                records.append(
                    {
                        **base,
                        "sample_key": f"{sample_base}/benign",
                        "url": benign_url,
                        "gt_is_phishing": "0",
                        "url_type": "benign",
                        "rule_id": "original",
                    }
                )
                for rule, squatted_url in squat_urls:
                    records.append(
                        {
                            **base,
                            "sample_key": f"{sample_base}/squatted/{rule}",
                            "url": squatted_url,
                            "gt_is_phishing": "1",
                            "url_type": "squatted",
                            "rule_id": rule,
                        }
                    )
    return records, skipped


# (attack_name, subdir, glob_pattern, brand_mode, brand_strip)
# brand_mode: "parent" = parent dir name; "stem" = filename stem;
#             "stem_strip" = filename stem with suffix removed
# SRNet is excluded from perturb_raw to match the paper's Table 10 total
# (6,528 = 544 benign × 12, covering only Swin/ViT/FGSM/PGD/CW).
PERTURB_ATTACK_CONFIGS = [
    ("swin", "swin", "*/screenshot_swin.png", "parent", ""),
    ("vit", "vit", "*/screenshot_vit.png", "parent", ""),
    ("fgsm", "white_box_attack/fgm_screenshot", "*.png", "stem", ""),
    ("pgd", "white_box_attack/pgd_screenshot", "*.png", "stem", ""),
    ("cw", "white_box_attack/cw_screenshot", "*.png", "stem", ""),
]


def _perturb_raw_records(
    *,
    perturb_root: Path,
    crawl_root: Path,
    aliases: dict[str, str],
    keep_unmapped: bool,
) -> tuple[list[dict[str, str]], int]:
    """Scan perturbated_dataset directories and create benign+squatted records per image."""
    if not perturb_root.is_dir() or not crawl_root.is_dir():
        raise FileNotFoundError("Perturb raw source requires both --perturb-root and --crawl-root.")
    records: list[dict[str, str]] = []
    skipped = 0
    crawl = _crawl_lookup(crawl_root)
    for attack, subdir, glob_pattern, brand_mode, brand_strip in PERTURB_ATTACK_CONFIGS:
        attack_dir = perturb_root / subdir
        if not attack_dir.is_dir():
            continue
        for image in sorted(attack_dir.glob(glob_pattern)):
            if image.stem.endswith("_crop_logo"):
                continue
            if brand_mode == "parent":
                raw_brand = image.parent.name
            elif brand_mode == "stem_strip":
                raw_brand = image.stem.removesuffix(brand_strip)
            else:
                raw_brand = image.stem
            canonical = _canonical_brand(raw_brand, aliases)
            if not canonical and not keep_unmapped:
                skipped += 1
                continue
            crawl_dir = crawl.get(_compact_key(raw_brand))
            login_path = crawl_dir / "login.txt" if crawl_dir else Path()
            if not crawl_dir or not login_path.is_file():
                skipped += 1
                continue
            benign_url = login_path.read_text(encoding="utf-8", errors="ignore").strip()
            squat_urls = _squatted_urls(benign_url)
            if not benign_url or not squat_urls:
                skipped += 1
                continue
            base = {
                "image_path": str(image.resolve()),
                "gt_brand": canonical or raw_brand,
                "dataset_split": "phisheval_perturb",
                "attack": attack,
                "source_brand": raw_brand,
            }
            sample_base = f"perturb/{attack}/{raw_brand}/{image.stem}"
            records.append(
                {
                    **base,
                    "sample_key": f"{sample_base}/benign",
                    "url": benign_url,
                    "gt_is_phishing": "0",
                    "url_type": "benign",
                    "rule_id": "original",
                }
            )
            for rule, squatted_url in squat_urls:
                records.append(
                    {
                        **base,
                        "sample_key": f"{sample_base}/squatted/{rule}",
                        "url": squatted_url,
                        "gt_is_phishing": "1",
                        "url_type": "squatted",
                        "rule_id": rule,
                    }
                )
    return records, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: VLBrandPhish adversarial robustness.")
    add_e2e_arguments(parser)
    # Override asset defaults for this experiment's PhishEval-based protocol.
    parser.set_defaults(
        logo_library_dir=str(DEFAULT_LOGO_LIBRARY_DIR),
        normal_urls_csv=str(DEFAULT_NORMAL_URLS_CSV),
        brand_text_mode="ckpt_text_proto_mean",
        brand_checkpoint=_first_existing(
            os.environ.get("VLBRANDPHISH_BRAND_CKPT", ""),
            REPO_ROOT / "models" / "vlbrandphish" / "brand_encoder_inference.pt",
            REPO_ROOT / "models" / "vlbrandphish" / "brand_identify.pt",
        ),
    )
    parser.add_argument(
        "--source",
        choices=("perturb", "perturb_raw", "visible", "all", "manifest"),
        default="perturb_raw",
        help="PhishEval source: perturb_raw scans perturbated_dataset directly; "
        "perturb uses prepared CSVs; visible uses visible_dataset2.",
    )
    parser.add_argument("--prepared-root", default=str(DEFAULT_PREPARED_ROOT), help="Prepared PhishEval input root.")
    parser.add_argument("--perturb-root", default=str(DEFAULT_PERTURB_ROOT), help="Root of perturbated_dataset.")
    parser.add_argument("--visible-root", default=str(DEFAULT_VISIBLE_ROOT))
    parser.add_argument("--crawl-root", default=str(DEFAULT_CRAWL_ROOT))
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Custom prepared CSV. Required with --source manifest; repeat to combine files.",
    )
    parser.add_argument("--keep-unmapped", action="store_true", help="Retain samples whose brand has no 152-brand reference.")
    parser.add_argument("--max-benign", type=int, default=-1, help="Deterministic benign limit after source selection.")
    parser.add_argument("--max-phish", type=int, default=-1, help="Deterministic squatted-phishing limit after source selection.")
    args = parser.parse_args(argv)

    aliases = _reference_brand_aliases(args.dataset_root)
    prepared_root = Path(args.prepared_root)
    selected_csvs = [Path(path) for path in args.input_csv]
    if args.source in {"perturb", "all"} and not selected_csvs:
        selected_csvs = [
            prepared_root / "csv" / "all_perturb_benign.csv",
            prepared_root / "csv" / "all_perturb_squatted.csv",
        ]
    if args.source == "manifest" and not selected_csvs:
        parser.error("--source manifest requires at least one --input-csv.")

    records: list[dict[str, str]] = []
    skipped = 0
    for source_csv in selected_csvs:
        if not source_csv.is_file():
            raise FileNotFoundError(source_csv)
        rows, skipped_rows = _read_csv_records(
            source_csv,
            input_root=prepared_root,
            aliases=aliases,
            keep_unmapped=bool(args.keep_unmapped),
        )
        records.extend(rows)
        skipped += skipped_rows

    if args.source in {"perturb_raw", "all"}:
        rows, skipped_rows = _perturb_raw_records(
            perturb_root=Path(args.perturb_root),
            crawl_root=Path(args.crawl_root),
            aliases=aliases,
            keep_unmapped=bool(args.keep_unmapped),
        )
        records.extend(rows)
        skipped += skipped_rows

    if args.source in {"visible", "all"}:
        rows, skipped_rows = _visible_records(
            visible_root=Path(args.visible_root),
            crawl_root=Path(args.crawl_root),
            aliases=aliases,
            keep_unmapped=bool(args.keep_unmapped),
        )
        records.extend(rows)
        skipped += skipped_rows

    records = limit_by_class(records, max_benign=int(args.max_benign), max_phish=int(args.max_phish))
    if not records:
        raise RuntimeError("No adversarial samples survived the selected source and reference-coverage policy.")
    execute_experiment(
        experiment_name="adversarial_robustness",
        experiment_dir=Path(__file__).parent,
        records=records,
        args=args,
        group_fields=("dataset_split", "attack", "url_type"),
        protocol={
            "name": "PhishEval adversarial robustness",
            "source": args.source,
            "selection": "squatted URLs are phishing; original URLs are benign",
            "reference_coverage_policy": "keep unmapped" if args.keep_unmapped else "filter to 152-brand reference coverage",
            "skipped_source_rows_or_brands": skipped,
            "expected_perturb_counts_before_reference_filter": {"benign": 544, "squatted_phish": 6528, "attack_types": 5, "note": "SRNet (41 brands, 492 squatted) excluded from perturb_raw total to match paper Table 10"},
            "visible_protocol": "13 visible-manipulation types; 12 deterministic squatted URLs per original page",
        },
    )


if __name__ == "__main__":
    main()