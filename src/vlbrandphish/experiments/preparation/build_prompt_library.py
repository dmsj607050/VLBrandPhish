from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


PATTERN_LINE1 = re.compile(
    r"^the (?P<color>.+?) (?P<content>.+?) logo of (?P<brand>.+?) in the (?P<industry>.+?) industry$",
    flags=re.IGNORECASE,
)
PATTERN_LINE2 = re.compile(
    r"^(?P<brand>.+?), a (?P<industry>.+?) brand with a (?P<style_content>.+?) logo$",
    flags=re.IGNORECASE,
)
PATTERN_LINE3 = re.compile(
    r"^the official (?P<color>.+?) (?P<style>.+?) logo of (?P<brand>.+?), a company in the (?P<industry>.+?) industry$",
    flags=re.IGNORECASE,
)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [str(x).strip() for x in f.read().splitlines() if str(x).strip()]


def _write_lines(path: str, lines: List[str]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            if i > 0:
                f.write("\n")
            f.write(str(line))


def _safe_first_text_file(brand_dir: str) -> str:
    files = sorted([x for x in os.listdir(brand_dir) if x.lower().endswith("_logo_description.txt")])
    return os.path.join(brand_dir, files[0]) if files else ""


def _parse_brand_five_dims(brand_name: str, lines: List[str]) -> Tuple[Dict[str, str], List[str]]:
    warn: List[str] = []
    if len(lines) < 3:
        warn.append(f"lines_lt_3={len(lines)}")
        return {
            "brand_name": str(brand_name),
            "color": "",
            "content": "",
            "style": "",
            "industry": "",
        }, warn

    l1 = str(lines[0])
    l2 = str(lines[1])
    l3 = str(lines[2])

    color = ""
    content = ""
    style = ""
    industry = ""

    m1 = PATTERN_LINE1.match(l1)
    if m1:
        color = str(m1.group("color")).strip()
        content = str(m1.group("content")).strip()
        industry = str(m1.group("industry")).strip()
    else:
        warn.append("line1_parse_failed")

    m2 = PATTERN_LINE2.match(l2)
    if m2:
        sc = str(m2.group("style_content")).strip()
        if content and sc.lower().endswith(content.lower()):
            style = sc[: len(sc) - len(content)].strip()
        else:
            if " " in sc:
                style = sc.split(" ", 1)[0].strip()
                if not content:
                    content = sc[len(style) :].strip()
            else:
                style = sc
                if not content:
                    content = sc
            warn.append("line2_style_content_heuristic")
        if not industry:
            industry = str(m2.group("industry")).strip()
    else:
        warn.append("line2_parse_failed")

    m3 = PATTERN_LINE3.match(l3)
    if m3:
        if not color:
            color = str(m3.group("color")).strip()
        if not style:
            style = str(m3.group("style")).strip()
        if not industry:
            industry = str(m3.group("industry")).strip()
    else:
        warn.append("line3_parse_failed")

    if not style:
        style = "modern"
        warn.append("style_fallback_default")
    if not content:
        content = "a logo mark"
        warn.append("content_fallback_default")
    if not color:
        color = "default"
        warn.append("color_fallback_default")
    if not industry:
        industry = "other"
        warn.append("industry_fallback_default")

    return {
        "brand_name": str(brand_name),
        "color": str(color),
        "content": str(content),
        "style": str(style),
        "industry": str(industry),
    }, warn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract five single-dimension texts from all_five_dimensions prompts, "
            "and export a mean-ready 5-line prompt library."
        )
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(str(args.input_dir))
    out_dir = os.path.abspath(str(args.out_dir))
    _ensure_dir(out_dir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    brands = [x for x in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, x))]
    rows: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    for brand in brands:
        src_brand_dir = os.path.join(input_dir, brand)
        src_file = _safe_first_text_file(src_brand_dir)
        if not src_file:
            warnings.append({"brand": brand, "warning": "missing_source_txt"})
            continue

        lines = _load_lines(src_file)
        dims, warn = _parse_brand_five_dims(brand, lines)
        for w in warn:
            warnings.append({"brand": brand, "warning": w})

        out_brand_dir = os.path.join(out_dir, brand)
        _ensure_dir(out_brand_dir)
        out_file = os.path.join(out_brand_dir, f"{brand}_logo_description.txt")
        out_lines = [
            f"{dims['brand_name']}",
            f"logo primary color: {dims['color']}",
            f"logo content type: {dims['content']}",
            f"logo visual style: {dims['style']}",
            f"brand industry: {dims['industry']}",
        ]
        _write_lines(out_file, out_lines)
        rows.append(
            {
                "brand": brand,
                "src_file": src_file,
                "out_file": out_file,
                "brand_name": dims["brand_name"],
                "color": dims["color"],
                "content": dims["content"],
                "style": dims["style"],
                "industry": dims["industry"],
            }
        )

    csv_path = os.path.join(out_dir, "_extracted_five_dims.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv

        fieldnames = list(rows[0].keys()) if rows else ["brand", "src_file", "out_file", "brand_name", "color", "content", "style", "industry"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    warn_path = os.path.join(out_dir, "_extraction_warnings.json")
    with open(warn_path, "w", encoding="utf-8") as f:
        json.dump({"warnings": warnings}, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(out_dir, "_extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": input_dir,
                "out_dir": out_dir,
                "n_brands_in_input": int(len(brands)),
                "n_brands_extracted": int(len(rows)),
                "n_warnings": int(len(warnings)),
                "csv": csv_path,
                "warnings_json": warn_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[extract] done input={input_dir}")
    print(f"[extract] out_dir={out_dir}")
    print(f"[extract] n_brands_extracted={len(rows)}")
    print(f"[extract] n_warnings={len(warnings)}")
    print(f"[extract] csv={csv_path}")
    print(f"[extract] warnings={warn_path}")


if __name__ == "__main__":
    main()
