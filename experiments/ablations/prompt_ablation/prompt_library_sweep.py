from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from inference.vlbrandphish.runtime.paths import default_run_dir, resolve_default_paths
from .prompt_modes import _discover_latest_stage2_summary


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_md(path: str, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "rank",
        "prompt_family",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "brand_acc_on_phish",
        "top1_brand_acc_on_all_phish",
        "auroc",
        "average_precision",
    ]
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---" for _ in cols]) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |\n")


def _discover_families(prompt_root: str) -> List[str]:
    root = os.path.abspath(prompt_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"prompt_root not found: {root}")

    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if str(name).startswith("_"):
            continue
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        out.append(name)
    if not out:
        raise RuntimeError(f"No prompt family directories found under: {root}")
    return out


def _parse_family_list(raw: str, discovered: List[str]) -> List[str]:
    txt = str(raw or "").strip()
    if not txt:
        return discovered
    names = [str(x).strip() for x in txt.replace(";", ",").split(",") if str(x).strip()]
    bad = [x for x in names if x not in discovered]
    if bad:
        raise ValueError(f"Unknown family names: {bad}")
    return names


def _read_single_compare_csv(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly 1 row in {path}, got {len(rows)}")
    row = dict(rows[0])

    float_fields = [
        "best_alpha",
        "best_threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "brand_acc_on_phish",
        "top1_brand_acc_on_all_phish",
        "final_brand_acc_on_all_phish",
        "auroc",
        "average_precision",
    ]
    int_fields = ["tp", "tn", "fp", "fn", "query_cache_hit", "repo_cache_hit", "feature_cache_hit"]

    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k in float_fields:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = None
        elif k in int_fields:
            try:
                out[k] = int(float(v))
            except Exception:
                out[k] = 0
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    defaults = resolve_default_paths(__file__)
    default_outputs = os.path.join(defaults.phishkiller_root, "outputs")
    default_prompt_root = os.path.join(defaults.testb_root, "generated_prompt_libraries")

    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluate all prompt families under generated_prompt_libraries by reusing "
            "OpenBrandPhish query cache, and summarize the best family."
        )
    )
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--outputs-dir", type=str, default=default_outputs)
    parser.add_argument("--prompt-root", type=str, default=default_prompt_root)
    parser.add_argument("--query-cache-path", type=str, required=True)
    parser.add_argument("--families", type=str, default="", help="Comma-separated family names. Empty means all.")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--out-base-dir", type=str, default=default_outputs)
    parser.add_argument(
        "--compare-module",
        type=str,
        default="vlbrandphish.experiments.ablation.prompt_modes",
        help="Installed module used to evaluate each prompt family.",
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--alpha-grid", type=str, default="")
    parser.add_argument("--threshold-grid", type=str, default="")
    parser.add_argument(
        "--grid-objective",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "precision", "recall", "brand_acc_on_phish"],
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--rebuild-repo-cache", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_json = str(args.summary_json or "").strip()
    if not summary_json:
        summary_json = _discover_latest_stage2_summary(str(args.outputs_dir))
    summary_json = os.path.abspath(summary_json)
    if not os.path.isfile(summary_json):
        raise FileNotFoundError(f"summary.json not found: {summary_json}")

    prompt_root = os.path.abspath(str(args.prompt_root))
    families_all = _discover_families(prompt_root)
    families = _parse_family_list(str(args.families), families_all)

    query_cache_path = os.path.abspath(str(args.query_cache_path))
    if not os.path.isfile(query_cache_path):
        raise FileNotFoundError(f"query_cache_path not found: {query_cache_path}")

    compare_module = str(args.compare_module).strip()
    if not compare_module:
        raise ValueError("compare_module must not be empty")

    out_dir = str(args.out_dir or "").strip()
    if not out_dir:
        out_dir = default_run_dir(str(args.out_base_dir), "prompt_library_sweep")
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)

    print(f"[sweep] summary_json={summary_json}")
    print(f"[sweep] prompt_root={prompt_root}")
    print(f"[sweep] query_cache_path={query_cache_path}")
    print(f"[sweep] families={families}")
    print(f"[sweep] out_dir={out_dir}")

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for idx, family in enumerate(families, start=1):
        fam_dir = os.path.join(prompt_root, family)
        fam_out = os.path.join(out_dir, family)
        cmd = [
            str(args.python_exe),
            "-m",
            compare_module,
            "--summary-json",
            summary_json,
            "--modes",
            "dbi5",
            "--dbi5-dir",
            fam_dir,
            "--query-cache-path",
            query_cache_path,
            "--out-dir",
            fam_out,
            "--grid-objective",
            str(args.grid_objective),
        ]
        if str(args.alpha_grid).strip():
            cmd += ["--alpha-grid", str(args.alpha_grid)]
        if str(args.threshold_grid).strip():
            cmd += ["--threshold-grid", str(args.threshold_grid)]
        if str(args.device).strip():
            cmd += ["--device", str(args.device)]
        if bool(args.no_progress):
            cmd.append("--no-progress")
        if bool(args.rebuild_repo_cache):
            cmd.append("--rebuild-repo-cache")
        if bool(args.rebuild_feature_cache):
            cmd.append("--rebuild-feature-cache")

        print(f"[sweep] ({idx}/{len(families)}) family={family}")
        print(f"[sweep] cmd={' '.join(cmd)}")
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=defaults.phishkiller_root)
        elapsed = time.perf_counter() - t0
        print(f"[sweep] family={family} exit_code={int(p.returncode)} elapsed_sec={elapsed:.1f}")

        if int(p.returncode) != 0:
            failed.append({"family": family, "exit_code": int(p.returncode), "elapsed_sec": float(elapsed)})
            continue

        compare_csv = os.path.join(fam_out, "metrics_compare.csv")
        if not os.path.isfile(compare_csv):
            failed.append({"family": family, "exit_code": -1, "elapsed_sec": float(elapsed), "reason": "metrics_compare.csv missing"})
            continue

        try:
            m = _read_single_compare_csv(compare_csv)
            m["prompt_family"] = family
            m["family_dir"] = fam_dir
            m["compare_csv"] = compare_csv
            m["elapsed_sec"] = float(elapsed)
            rows.append(m)
        except Exception as e:
            failed.append({"family": family, "exit_code": -2, "elapsed_sec": float(elapsed), "reason": f"parse_failed: {e}"})

    rows_sorted = sorted(rows, key=lambda x: float(x.get("f1", -1.0) if x.get("f1") is not None else -1.0), reverse=True)
    for rank, row in enumerate(rows_sorted, start=1):
        row["rank"] = int(rank)

    sweep_csv = os.path.join(out_dir, "sweep_metrics.csv")
    sweep_md = os.path.join(out_dir, "sweep_metrics.md")
    sweep_json = os.path.join(out_dir, "sweep_summary.json")
    failed_json = os.path.join(out_dir, "sweep_failed.json")

    if rows_sorted:
        _write_csv(sweep_csv, rows_sorted)
        _write_md(sweep_md, rows_sorted)
    else:
        with open(sweep_csv, "w", encoding="utf-8", newline="") as f:
            f.write("")
        with open(sweep_md, "w", encoding="utf-8") as f:
            f.write("")

    _save_json(failed_json, {"failed": failed})

    best = rows_sorted[0] if rows_sorted else None
    _save_json(
        sweep_json,
        {
            "summary_json": summary_json,
            "prompt_root": prompt_root,
            "query_cache_path": query_cache_path,
            "n_families_total": int(len(families)),
            "n_success": int(len(rows_sorted)),
            "n_failed": int(len(failed)),
            "grid_objective": str(args.grid_objective),
            "best_family": best,
            "sweep_csv": sweep_csv,
            "sweep_md": sweep_md,
            "failed_json": failed_json,
            "rows": rows_sorted,
        },
    )

    print(f"[sweep] done success={len(rows_sorted)} failed={len(failed)}")
    print(f"[sweep] sweep_csv={sweep_csv}")
    print(f"[sweep] sweep_md={sweep_md}")
    if best is not None:
        print(
            f"[sweep] best family={best.get('prompt_family')} "
            f"f1={float(best.get('f1', 0.0)):.4f} "
            f"acc={float(best.get('accuracy', 0.0)):.4f} "
            f"brand_acc={float(best.get('brand_acc_on_phish', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()
