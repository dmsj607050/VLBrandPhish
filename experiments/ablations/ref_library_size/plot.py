"""Plot F1 versus inference reference count from ablation_table.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


TRAIN_COLORS = {1: "#0072B2", 3: "#D55E00", 5: "#009E73", 10: "#CC79A7"}
TRAIN_MARKERS = {1: "o", 3: "s", 5: "D", 10: "^"}
BASELINE_STYLES = {
    "PhishIntention": {"color": "#222222", "marker": "X", "linestyle": "--"},
    "Phishpedia": {"color": "#777777", "marker": "v", "linestyle": "-."},
    "PDOS": {"color": "#E69F00", "marker": "P", "linestyle": ":"},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="ablation_table.csv from run.py")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError as exc:  # pragma: no cover - depends on optional plotting extra
        raise SystemExit("Install the plotting dependency with: pip install '.[figures]'") from exc

    rows = read_rows(args.input_csv)
    vlbp: dict[int, list[tuple[int, float]]] = defaultdict(list)
    baselines: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "").strip()
        try:
            x = int(row["infer_ref"])
            score = float(row["f1"])
        except (KeyError, TypeError, ValueError):
            continue
        if method == "VLBrandPhish":
            try:
                train_n = int(row["train_ref"])
            except (KeyError, TypeError, ValueError):
                continue
            vlbp[train_n].append((x, score))
        elif method in BASELINE_STYLES:
            baselines[method].append((x, score))

    if not vlbp and not baselines:
        raise ValueError(f"No plottable F1 rows found in {args.input_csv}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    for train_n in sorted(vlbp):
        points = sorted(vlbp[train_n])
        ax.plot(
            [p[0] for p in points], [p[1] for p in points],
            color=TRAIN_COLORS.get(train_n, "#333333"), marker=TRAIN_MARKERS.get(train_n, "o"),
            linewidth=1.8, markersize=5, label=f"VLBrandPhish (train cap {train_n})",
        )
    for method, style in BASELINE_STYLES.items():
        points = sorted(baselines.get(method, []))
        if points:
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                linewidth=1.7, markersize=5, label=method,
            )

    ax.set_xlabel("Inference reference logos per brand (maximum)")
    ax.set_ylabel("F1 score")
    ax.set_xticks([1, 3, 5, 10])
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    if vlbp:
        inset = inset_axes(ax, width="39%", height="38%", loc="lower right", borderpad=1.4)
        zoom_scores = [score for pts in vlbp.values() for x, score in pts if x in (3, 5, 10)]
        if zoom_scores:
            lo = max(0.0, min(zoom_scores) - 0.015)
            hi = min(1.02, max(zoom_scores) + 0.015)
            for train_n in sorted(vlbp):
                points = [(x, y) for x, y in sorted(vlbp[train_n]) if x in (3, 5, 10)]
                if points:
                    inset.plot(
                        [p[0] for p in points], [p[1] for p in points],
                        color=TRAIN_COLORS.get(train_n, "#333333"), marker=TRAIN_MARKERS.get(train_n, "o"),
                        linewidth=1.3, markersize=3.5,
                    )
            inset.set_xlim(2.7, 10.3)
            inset.set_ylim(lo, hi)
            inset.set_xticks([3, 5, 10])
            inset.tick_params(labelsize=7, length=2)
            inset.grid(axis="y", color="#E4E6E9", linewidth=0.5)
            inset.set_title("VLBrandPhish detail", fontsize=7, pad=3)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, columnspacing=1.0)
    out_dir = args.out_dir or args.input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"reference_count_f1.{suffix}", dpi=300, bbox_inches="tight")
    print(f"Saved paper figure to {out_dir}")


if __name__ == "__main__":
    main()
