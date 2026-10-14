"""Dataset manifests for the four PDOS phishing-detection experiments.

The source VLBrandPhish experiments and inference package define the sample
selection protocol.  This module ports only that protocol; model inference is
kept in the PDOS YOLO+ResNet pipeline.
"""

from __future__ import annotations

import ast
import csv
import io
import json
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_brand(value: str | None) -> str:
    text = str(value or "").replace("_", " ").strip().lower()
    out: list[str] = []
    previous_space = False
    for char in text:
        if "a" <= char <= "z" or "0" <= char <= "9":
            out.append(char)
            previous_space = False
        elif not previous_space:
            out.append(" ")
            previous_space = True
    return " ".join("".join(out).split())


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _first_existing_child(root: Path, names: Iterable[str]) -> Path:
    """Resolve a dataset child with case-insensitive, legacy-name support."""
    candidates = [root / name for name in names]
    found = _first_existing(*candidates)
    if found.exists():
        return found
    if root.is_dir():
        by_lower = {child.name.lower(): child for child in root.iterdir()}
        for name in names:
            match = by_lower.get(name.lower())
            if match is not None:
                return match
    return candidates[0]


def _first_image_by_stem(directory: Path, stem: str) -> Path | None:
    exact = [directory / f"{stem}{suffix}" for suffix in sorted(IMAGE_SUFFIXES)]
    for path in exact:
        if path.is_file():
            return path
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_pi_info(path: Path) -> dict[str, object]:
    value = ast.literal_eval(path.read_text(encoding="utf-8", errors="ignore").strip())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def _load_normal_url_csv(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return phishing-brand -> canonical-brand and canonical-brand -> URLs."""
    brand_map: dict[str, str] = {}
    domains: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            logo_brand = str(row.get("logo_brand", "") or "").strip()
            phish_brand = str(row.get("phish_brand", "") or "").strip()
            urls = [item.strip() for item in str(row.get("url", "") or "").split(";") if item.strip()]
            if not logo_brand:
                continue
            canonical_key = normalize_brand(logo_brand)
            if phish_brand:
                brand_map[normalize_brand(phish_brand)] = logo_brand
            if urls:
                domains.setdefault(canonical_key, [])
                for url in urls:
                    if url not in domains[canonical_key]:
                        domains[canonical_key].append(url)
    return brand_map, domains


def _load_noncrp(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.replace("\x00", "").strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.replace("\x00", "").strip()
    }


def _resolve_pi_roots(root: Path) -> tuple[Path, Path]:
    test = root / "test"
    phish = _first_existing(test / "phish_sample_30k", root / "phish_sample_30k")
    benign = _first_existing(test / "benign_25k", root / "benign_25k")
    return phish, benign


def _phishintention_records(root: Path) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    phish_root, benign_root = _resolve_pi_roots(root)
    normal_csv = _first_existing(root / "normal_urls.csv", root.parent / "normal_urls.csv")
    noncrp = _first_existing(root / "phish30k_nonCRP.txt", root.parent / "phish30k_nonCRP.txt")
    if not normal_csv.is_file():
        raise FileNotFoundError(f"PhishIntention normal_urls.csv not found under {root}")
    brand_map, official_urls = _load_normal_url_csv(normal_csv)
    aligned = set(official_urls)
    excluded = _load_noncrp(noncrp)
    rows: list[dict[str, str]] = []

    if not phish_root.is_dir() or not benign_root.is_dir():
        raise FileNotFoundError(
            f"Expected PhishIntention roots under {root}; found phish={phish_root}, benign={benign_root}"
        )
    for sample_dir in sorted(path for path in phish_root.iterdir() if path.is_dir()):
        shot = sample_dir / "shot.png"
        info = sample_dir / "info.txt"
        if not shot.is_file() or not info.is_file() or sample_dir.name in excluded:
            continue
        try:
            meta = _read_pi_info(info)
        except (OSError, SyntaxError, ValueError):
            continue
        url = str(meta.get("url", "") or "").strip()
        raw_brand = str(meta.get("brand", "") or "").strip()
        canonical = brand_map.get(normalize_brand(raw_brand), raw_brand)
        if not url or "\x00" in url or not canonical or normalize_brand(canonical) not in aligned:
            continue
        rows.append({
            "sample_key": sample_dir.name,
            "image_path": str(shot.resolve()),
            "url": url,
            "gt_is_phishing": "1",
            "gt_brand": canonical,
            "brand_key": normalize_brand(canonical),
            "collection_timestamp": "",
            "source_dataset": "PhishIntention",
            "url_policy_mode": "phishintention",
        })

    for sample_dir in sorted(path for path in benign_root.iterdir() if path.is_dir()):
        shot = sample_dir / "shot.png"
        info = sample_dir / "info.txt"
        if not shot.is_file() or not info.is_file():
            continue
        try:
            content = info.read_text(encoding="utf-8", errors="ignore").strip()
            try:
                value = ast.literal_eval(content)
                url = str(value.get("url", "") if isinstance(value, dict) else value).strip()
            except (SyntaxError, ValueError):
                url = content
        except OSError:
            continue
        if not url or "\x00" in url:
            continue
        rows.append({
            "sample_key": sample_dir.name,
            "image_path": str(shot.resolve()),
            "url": url,
            "gt_is_phishing": "0",
            "gt_brand": "",
            "brand_key": "",
            "collection_timestamp": "",
            "source_dataset": "PhishIntention",
            "url_policy_mode": "phishintention",
        })
    if not rows:
        raise RuntimeError("PhishIntention produced zero usable records.")
    return rows, official_urls


def _load_overlap(root: Path) -> dict[str, int]:
    path = _first_existing_child(root, ("08_brands_overlap.csv", "brands_overlap.csv", "08_brand_overlap.csv"))
    if not path.is_file():
        raise FileNotFoundError(f"Ours overlap file not found: {path}")
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            brand = str(row.get("brand_dir_7299", "") or "").strip()
            if brand:
                result[brand] = int(str(row.get("in_277", "0") or "0").strip() or "0")
    return result


def _ours_records(root: Path, experiment: str) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    # The local copy uses the paper's numbered names, while the server copy
    # may have the shorter names (phish_12K / benign_12K).  The inner layout
    # is identical, so resolve only the outer naming difference here.
    phish_root = _first_existing_child(root, ("01_phish_12K", "phish_12K", "phish_12k"))
    benign_root = _first_existing_child(root, ("02_bengin_12K", "bengin_12K", "benign_12K", "benign_12k"))
    normal_csv = _first_existing_child(root, ("07_normal_urls_testb.csv", "normal_urls_testb.csv", "normal_urls.csv"))
    if not phish_root.is_dir() or not benign_root.is_dir():
        raise FileNotFoundError(f"Expected Ours roots under {root}: {phish_root} and {benign_root}")
    if not normal_csv.is_file():
        raise FileNotFoundError(f"Ours normal URL file not found: {normal_csv}")
    _, official_urls = _load_normal_url_csv(normal_csv)
    overlap = _load_overlap(root)
    if experiment == "cross_time":
        selected = {brand for brand, flag in overlap.items() if flag == 1}
        split = "ours_cross_time_seen_brands"
    elif experiment == "cross_brand":
        selected = {brand for brand, flag in overlap.items() if flag == 0}
        split = "ours_cross_brand_novel_brands"
    else:
        selected = set(overlap)
        split = "ours_phish"
    rows: list[dict[str, str]] = []
    for brand_dir in sorted(path for path in phish_root.iterdir() if path.is_dir()):
        if brand_dir.name not in selected:
            continue
        json_dir, screen_dir = brand_dir / "jsons", brand_dir / "screens"
        if not json_dir.is_dir() or not screen_dir.is_dir():
            continue
        for meta_path in sorted(json_dir.glob("*.json")):
            image_path = _first_image_by_stem(screen_dir, meta_path.stem)
            if image_path is None:
                continue
            try:
                meta = _read_json(meta_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            url = str(meta.get("url", "") or "").strip()
            if not url or "\x00" in url:
                continue
            rows.append({
                "sample_key": f"{brand_dir.name}/{meta_path.stem}",
                "image_path": str(image_path.resolve()),
                "url": url,
                "gt_is_phishing": "1",
                "gt_brand": brand_dir.name,
                "brand_key": normalize_brand(brand_dir.name),
                "collection_timestamp": str(meta.get("timestamp", "") or ""),
                "source_dataset": "VLBrandPhish_Ours",
                "url_policy_mode": "official_plus_overrides_no_whitelist",
                "dataset_split": split,
            })
    for sample_dir in sorted(path for path in benign_root.iterdir() if path.is_dir()):
        meta_path = sample_dir / f"{sample_dir.name}.json"
        if not meta_path.is_file():
            candidates = sorted(sample_dir.glob("*.json"))
            meta_path = candidates[0] if candidates else meta_path
        image_path = _first_image_by_stem(sample_dir, meta_path.stem) if meta_path.is_file() else None
        if image_path is None or not meta_path.is_file():
            continue
        try:
            meta = _read_json(meta_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        url = str(meta.get("url", "") or "").strip()
        if not url or "\x00" in url:
            continue
        rows.append({
            "sample_key": sample_dir.name,
            "image_path": str(image_path.resolve()),
            "url": url,
            "gt_is_phishing": "0",
            "gt_brand": "",
            "brand_key": "",
            "collection_timestamp": str(meta.get("timestamp", "") or ""),
            "source_dataset": "VLBrandPhish_Ours",
            "url_policy_mode": "official_plus_overrides_no_whitelist",
            "dataset_split": "ours_benign",
        })
    if not rows:
        raise RuntimeError(f"Ours {experiment} produced zero usable records.")
    return rows, official_urls


def build_records(dataset: str, root: str | Path, experiment: str) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    dataset = str(dataset).strip().lower()
    if dataset == "phishintention":
        if experiment != "overall_phishintention":
            raise ValueError("PhishIntention supports only overall_phishintention in this protocol.")
        return _phishintention_records(Path(root).resolve())
    if dataset == "ours":
        if experiment not in {"overall_ours", "cross_time", "cross_brand"}:
            raise ValueError("Ours supports overall_ours, cross_time, and cross_brand.")
        return _ours_records(Path(root).resolve(), experiment)
    raise ValueError("dataset must be phishintention or ours")


def limit_records(records: Iterable[dict[str, str]], max_phish: int = -1, max_benign: int = -1) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    phish = benign = 0
    for row in records:
        if row["gt_is_phishing"] == "1":
            if max_phish >= 0 and phish >= max_phish:
                continue
            phish += 1
        else:
            if max_benign >= 0 and benign >= max_benign:
                continue
            benign += 1
        result.append(dict(row))
    return result


def write_manifest(records: list[dict[str, str]], path: str | Path) -> Path:
    if not records:
        raise ValueError("Cannot write an empty manifest.")
    required = {"sample_key", "image_path", "url", "gt_is_phishing", "gt_brand"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    keys = [row["sample_key"] for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest sample_key values are not unique.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return target


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    # Python 3.8's csv module rejects NUL characters before the row reaches the
    # parser.  Older manifests may contain one copied from a malformed source
    # info.txt; remove it here as a compatibility measure, while new manifests
    # are filtered before writing.
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\x00", "")
    rows = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(io.StringIO(text))]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows
