"""Overall detection on the PhishIntention public dataset.

Research question: RQ1 -- overall end-to-end phishing detection on PhishIntention.
Dataset: PhishIntention public corpus (raw: 29,496 phishing + 25,400 benign directories).
Metrics: Precision, Recall, F1, AUROC, AP, TPR@FPR<=1%, Match Acc.

**Data filtering chain** (faithful to ``evaluation_30k/cli.py`` + ``dataset.py``)::

1. ``_load_samples`` -- keep only directories with both ``shot.png`` and ``info.txt``
   whose URL is non-empty.
2. ``phish30k_nonCRP.txt`` exclusion list -- remove 4,093 non-credential-requiring
   phishing samples (keys excluded by the PhishIntention authors).
3. ``aligned_phish_brands_only`` -- drop 871 phishing samples whose brand, after
   normalisation and ``normal_urls.csv`` lookup, does not map to one of the 277
   reference brands *that has a non-empty domain list* (=> ``canonical_to_domains``).

Final clean counts: **24,532 phishing + 21,973 benign = 46,505**.

Raw screenshots live under::

   <workspace>/model_train/Brand_Identity/datasets/test/
     phish_sample_30k/
     benign_25k/

The 277-brand reference logo library is at::

   <workspace>/model_train/Brand_Identity/datasets/Stage2/277_brand_pages/

``normal_urls.csv`` and ``phish30k_nonCRP.txt`` are bundled in the repo at
``data/PhishIntention/`` and are wired as built-in defaults.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments._shared.e2e import (
    _first_existing,
    add_e2e_arguments,
    execute_experiment,
)


# ----- Shared asset defaults -----
PI_TEST_ROOT = Path(_first_existing(
    os.environ.get("VLBRANDPHISH_PI_TEST_ROOT", ""),
    REPO_ROOT / "data" / "PhishIntention" / "test",
))
DEFAULT_PHISH_ROOT = PI_TEST_ROOT / "phish_sample_30k"
DEFAULT_BENIGN_ROOT = PI_TEST_ROOT / "benign_25k"
DEFAULT_LOGO_LIBRARY_DIR = _first_existing(
    os.environ.get("VLBRANDPHISH_PI_LOGO_LIBRARY_DIR", ""),
    REPO_ROOT / "data" / "PhishIntention" / "277_brand_pages",
)
DEFAULT_NORMAL_URLS_CSV = REPO_ROOT / "data" / "PhishIntention" / "normal_urls.csv"
DEFAULT_NONCRP_TXT = REPO_ROOT / "data" / "PhishIntention" / "phish30k_nonCRP.txt"


# ====================================================================
#   Brand-normalisation helpers  (exact copies of
#   evaluation_30k/brand_utils.py logic)
# ====================================================================

def _normalize_brand_key(name: Optional[str]) -> str:
    """Remove all non-alphanumeric characters, lower-case, collapse whitespace."""
    if name is None:
        return ""
    s = str(name).replace("_", " ").strip().lower()
    out: List[str] = []
    prev_space = False
    for ch in s:
        is_alnum = ("0" <= ch <= "9") or ("a" <= ch <= "z")
        if is_alnum:
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return " ".join("".join(out).split())


def _load_aligned_brands(
    path: str | Path,
) -> tuple[Dict[str, str], List[str]]:
    """Return (phish_key -> canonical, [aligned canonical keys with domains])."""
    phish_key_to_canonical: Dict[str, str] = {}
    canonical_to_domains: Dict[str, List[str]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            logo_brand = str(row.get("logo_brand", "") or "").strip()
            phish_brand = str(row.get("phish_brand", "") or "").strip()
            domains_field = str(row.get("url", "") or "").strip()
            if not logo_brand:
                continue
            canonical = _normalize_brand_key(logo_brand)
            if not canonical:
                continue
            pk = _normalize_brand_key(phish_brand)
            if pk:
                phish_key_to_canonical[pk] = canonical
            domains = [d.strip() for d in domains_field.split(";") if d.strip()]
            if not domains:
                continue
            canonical_to_domains.setdefault(canonical, [])
            existing = set(canonical_to_domains[canonical])
            for d in domains:
                if d not in existing:
                    canonical_to_domains[canonical].append(d)

    aligned = sorted(canonical_to_domains.keys())
    return phish_key_to_canonical, aligned


def _canonicalize_phish_brand(
    name: Optional[str],
    phish_key_to_canonical: Dict[str, str],
) -> str:
    k = _normalize_brand_key(name)
    return phish_key_to_canonical.get(k, k) if k else ""


def _load_noncrp_keys(path: str | Path) -> set[str]:
    keys: set[str] = set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\x00" in line:
                line = line.replace("\x00", "")
            s = line.strip()
            if s:
                keys.add(s)
    return keys


# ====================================================================
#   Sample loading & filtering (dataset.py / cli.py replica)
# ====================================================================

def _load_phishintention_samples(
    phish_root: str | Path,
    benign_root: str | Path,
    normal_urls_csv: str | Path,
    noncrp_txt: str | Path,
) -> list[dict[str, str]]:
    """Load and filter samples exactly as PhishIntention's evaluation_30k does.

    Returns a manifest list compatible with ``execute_experiment``.
    """
    # --- Brand alignment lookup ---
    pk2c, aligned = _load_aligned_brands(normal_urls_csv)
    aligned_set = set(aligned)

    # --- Non-CRP exclusion keys ---
    noncrp_keys = _load_noncrp_keys(noncrp_txt)

    # --- Phishing samples ---
    phish_root = Path(phish_root)
    benign_root = Path(benign_root)
    records: list[dict[str, str]] = []

    for name in sorted(d.name for d in phish_root.iterdir() if d.is_dir()):
        dp = phish_root / name
        shot = dp / "shot.png"
        info = dp / "info.txt"
        if not shot.is_file() or not info.is_file():
            continue
        try:
            raw = info.read_text(encoding="utf-8", errors="ignore").strip()
            meta = ast.literal_eval(raw)
            url = str(meta.get("url", "") or "").strip()
            brand = str(meta.get("brand", "") or "").strip()
        except Exception:
            continue
        if not url or "\x00" in url:
            continue
        # nonCRP exclusion
        if name in noncrp_keys:
            continue
        # aligned_phish_brands_only
        canon_k = _canonicalize_phish_brand(brand, pk2c)
        if not canon_k or canon_k not in aligned_set:
            continue
        records.append(
            {
                "sample_key": name,
                "image_path": str(shot.resolve()),
                "url": url,
                "gt_is_phishing": "1",
                "gt_brand": brand,
                "dataset_split": "phishintention_phish",
                "collection_timestamp": "",
                "source_dataset": "PhishIntention",
            }
        )

    # --- Benign samples ---
    for name in sorted(d.name for d in benign_root.iterdir() if d.is_dir()):
        dp = benign_root / name
        shot = dp / "shot.png"
        info = dp / "info.txt"
        if not shot.is_file() or not info.is_file():
            continue
        try:
            content = info.read_text(encoding="utf-8", errors="ignore").strip()
            url = ""
            if content:
                try:
                    meta = ast.literal_eval(content)
                    if isinstance(meta, dict):
                        url = str(meta.get("url", "") or "").strip()
                    else:
                        url = str(content)
                except Exception:
                    url = str(content)
        except Exception:
            continue
        if not url or "\x00" in url:
            continue
        records.append(
            {
                "sample_key": name,
                "image_path": str(shot.resolve()),
                "url": url,
                "gt_is_phishing": "0",
                "gt_brand": "",
                "dataset_split": "phishintention_benign",
                "collection_timestamp": "",
                "source_dataset": "PhishIntention",
            }
        )

    if not records:
        raise RuntimeError("PhishIntention filtering produced zero samples -- check data roots.")
    return records


def _patch_parser_defaults(parser: argparse.ArgumentParser) -> None:
    overrides = {
        "logo_library_dir": str(DEFAULT_LOGO_LIBRARY_DIR),
        "normal_urls_csv": str(DEFAULT_NORMAL_URLS_CSV),
        "benign_root": str(DEFAULT_BENIGN_ROOT),
        "phish_root": str(DEFAULT_PHISH_ROOT),
        "brand_text_mode": "ckpt_text_proto_mean",
    }
    for action in parser._actions:
        dest = getattr(action, "dest", "")
        if dest in overrides:
            action.default = overrides[dest]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RQ1: VLBrandPhish end-to-end on the PhishIntention dataset (N=46,505)."
    )
    add_e2e_arguments(parser)
    _patch_parser_defaults(parser)
    parser.set_defaults(url_policy_mode="phishintention")
    parser.add_argument(
        "--noncrp-txt",
        default=str(DEFAULT_NONCRP_TXT),
        help="phish30k_nonCRP.txt exclusion list (directory-name keys to exclude).",
    )
    parser.add_argument(
        "--max-benign",
        type=int,
        default=-1,
        help="Deterministic benign limit; -1 = all (21,973).",
    )
    parser.add_argument(
        "--max-phish",
        type=int,
        default=-1,
        help="Deterministic phishing limit; -1 = all (24,532).",
    )
    args = parser.parse_args(argv)

    records = _load_phishintention_samples(
        phish_root=args.phish_root or str(DEFAULT_PHISH_ROOT),
        benign_root=args.benign_root or str(DEFAULT_BENIGN_ROOT),
        normal_urls_csv=args.normal_urls_csv,
        noncrp_txt=args.noncrp_txt,
    )

    # Apply optional class caps (stable order = directory-name sorted)
    capped: list[dict[str, str]] = []
    n_phish = 0
    n_benign = 0
    for row in records:
        is_phish = row["gt_is_phishing"] == "1"
        if is_phish:
            if int(args.max_phish) >= 0 and n_phish >= int(args.max_phish):
                continue
            n_phish += 1
        else:
            if int(args.max_benign) >= 0 and n_benign >= int(args.max_benign):
                continue
            n_benign += 1
        capped.append(row)

    protocol = {
        "name": "VLBP-PhishIntention / overall",
        "filtering": {
            "phish_noncrp_excluded": len(_load_noncrp_keys(args.noncrp_txt)),
            "phish_aligned_brands_only": "phish_key to canonical via normal_urls.csv, canonical must have non-empty domain list",
            "benign_info_txt_required": True,
            "benign_empty_url_excluded": True,
        },
        "expected_full_counts": {"phish": 24532, "benign": 21973, "total": 46505},
    }
    execute_experiment(
        experiment_name="overall_phishintention",
        experiment_dir=Path(__file__).parent,
        records=capped,
        args=args,
        group_fields=("dataset_split",),
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
