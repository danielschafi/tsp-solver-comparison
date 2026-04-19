"""Create aggregated comparison plots from benchmark results.

Usage:
    uv run -m src.utils.create_aggregate_plots [--save] [--out results/plots]

Produces five figures:
  1. Time scaling      — mean solve time vs. problem size (log-log) per problem type
  2. Solution quality  — mean tour cost vs. problem size per problem type
  3. Optimality gap    — mean % gap above best-known vs. problem size (line plot)
  4. Optimality gap    — mean % gap above best-known per algorithm/size (bar chart)
  5. Gap heatmap       — algorithm × size grid, color = mean gap %
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.utils.compute_statistics import compute_stats, load_results, save_txt

ALGO_ORDER = [
    "Greedy",
    "TwoOpt",
    "SimulatedAnnealing",
    "TabuSearch",
    "IterativeLocalSearch",
    "Genetic",
    "Concorde",
]

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALGO_COLOR = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ALGO_ORDER)}
MARKERS = ["o", "s", "^", "D", "v", "P", "*"]
ALGO_MARKER = {a: MARKERS[i % len(MARKERS)] for i, a in enumerate(ALGO_ORDER)}


def _group_by(rows: list[dict], keys: list[str]) -> dict:
    from collections import defaultdict
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def fig_time_scaling(rows: list[dict], ax_uniform: plt.Axes, ax_clustered: plt.Axes) -> None:
    axes_map = {"uniform": ax_uniform, "clustered": ax_clustered}
    by_type_algo = _group_by(rows, ["problem_type", "algorithm"])

    for (ptype, algo), grp in sorted(by_type_algo.items()):
        ax = axes_map.get(ptype)
        if ax is None:
            continue
        grp_sorted = sorted(grp, key=lambda r: r["problem_size"])
        sizes = [r["problem_size"] for r in grp_sorted]
        times = [r["mean_time_s"] for r in grp_sorted]
        errs = [r["std_time_s"] for r in grp_sorted]
        ax.errorbar(
            sizes, times, yerr=errs,
            label=algo,
            color=ALGO_COLOR.get(algo),
            marker=ALGO_MARKER.get(algo),
            linewidth=1.5, markersize=6, capsize=3,
        )

    for ptype, ax in axes_map.items():
        ax.set_title(f"{ptype.capitalize()} instances")
        ax.set_xlabel("Problem size (n)")
        ax.set_ylabel("Mean solve time (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)


def fig_solution_quality(rows: list[dict], ax_uniform: plt.Axes, ax_clustered: plt.Axes) -> None:
    axes_map = {"uniform": ax_uniform, "clustered": ax_clustered}
    by_type_algo = _group_by(rows, ["problem_type", "algorithm"])

    for (ptype, algo), grp in sorted(by_type_algo.items()):
        ax = axes_map.get(ptype)
        if ax is None:
            continue
        grp_sorted = sorted(grp, key=lambda r: r["problem_size"])
        sizes = [r["problem_size"] for r in grp_sorted]
        costs = [r["mean_cost"] for r in grp_sorted]
        errs = [r["std_cost"] for r in grp_sorted]
        valid = [(s, c, e) for s, c, e in zip(sizes, costs, errs) if not np.isnan(c)]
        if not valid:
            continue
        s, c, e = zip(*valid)
        ax.errorbar(
            s, c, yerr=e,
            label=algo,
            color=ALGO_COLOR.get(algo),
            marker=ALGO_MARKER.get(algo),
            linewidth=1.5, markersize=6, capsize=3,
        )

    for ptype, ax in axes_map.items():
        ax.set_title(f"{ptype.capitalize()} instances")
        ax.set_xlabel("Problem size (n)")
        ax.set_ylabel("Mean tour cost")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)


def fig_gap_bars(rows: list[dict], axes: list[plt.Axes], ptypes: list[str]) -> None:
    for ax, ptype in zip(axes, ptypes):
        subset = [r for r in rows if r["problem_type"] == ptype and not np.isnan(r["mean_gap_pct"])]
        if not subset:
            ax.set_visible(False)
            continue

        sizes = sorted({r["problem_size"] for r in subset})
        algos = [a for a in ALGO_ORDER if any(r["algorithm"] == a for r in subset)]

        x = np.arange(len(sizes))
        width = 0.8 / max(len(algos), 1)

        for i, algo in enumerate(algos):
            gaps = []
            errs = []
            for size in sizes:
                match = next((r for r in subset if r["algorithm"] == algo and r["problem_size"] == size), None)
                gaps.append(match["mean_gap_pct"] if match else 0)
                errs.append(match["std_gap_pct"] if match else 0)
            offset = (i - len(algos) / 2 + 0.5) * width
            ax.bar(
                x + offset, gaps, width,
                label=algo, color=ALGO_COLOR.get(algo), alpha=0.85,
                yerr=errs, capsize=3, error_kw={"linewidth": 1},
            )

        ax.set_title(f"{ptype.capitalize()} instances")
        ax.set_xlabel("Problem size (n)")
        ax.set_ylabel("Mean gap above best known (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)


def fig_gap_lines(rows: list[dict], axes: list[plt.Axes], ptypes: list[str]) -> None:
    for ax, ptype in zip(axes, ptypes):
        subset = [r for r in rows if r["problem_type"] == ptype]
        by_algo = _group_by(subset, ["algorithm"])

        for (algo,), grp in sorted(by_algo.items()):
            grp_sorted = sorted(grp, key=lambda r: r["problem_size"])
            valid = [(r["problem_size"], r["mean_gap_pct"], r["std_gap_pct"])
                     for r in grp_sorted if not np.isnan(r["mean_gap_pct"])]
            if not valid:
                continue
            sizes, gaps, errs = zip(*valid)
            ax.errorbar(
                sizes, gaps, yerr=errs,
                label=algo,
                color=ALGO_COLOR.get(algo),
                marker=ALGO_MARKER.get(algo),
                linewidth=1.5, markersize=6, capsize=3,
            )

        ax.set_title(f"{ptype.capitalize()} instances")
        ax.set_xlabel("Problem size (n)")
        ax.set_ylabel("Mean gap above best known (%)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)


def fig_gap_heatmap(rows: list[dict], axes: list[plt.Axes], ptypes: list[str]) -> None:
    for ax, ptype in zip(axes, ptypes):
        subset = [r for r in rows if r["problem_type"] == ptype]
        sizes = sorted({r["problem_size"] for r in subset})
        algos = [a for a in ALGO_ORDER if any(r["algorithm"] == a for r in subset)]

        matrix = np.full((len(algos), len(sizes)), np.nan)
        for i, algo in enumerate(algos):
            for j, size in enumerate(sizes):
                match = next(
                    (r for r in subset if r["algorithm"] == algo and r["problem_size"] == size),
                    None,
                )
                if match and not np.isnan(match["mean_gap_pct"]):
                    matrix[i, j] = match["mean_gap_pct"]

        masked = np.ma.array(matrix, mask=np.isnan(matrix))
        im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=0)
        plt.colorbar(im, ax=ax, label="Mean gap (%)")

        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        ax.set_xlabel("Problem size (n)")
        ax.set_title(f"{ptype.capitalize()} — gap % heatmap")

        for i in range(len(algos)):
            for j in range(len(sizes)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aggregated TSP benchmark plots.")
    parser.add_argument("--results", default="results/results.json", help="Path to results JSON.")
    parser.add_argument("--save", action="store_true", help="Save figures instead of displaying.")
    parser.add_argument("--out", default="results/plots", metavar="DIR", help="Output directory for saved figures.")
    args = parser.parse_args()

    records = load_results(Path(args.results))
    rows = compute_stats(records)
    ptypes = sorted({r["problem_type"] for r in rows})

    out_dir = Path(args.out)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_txt(rows, out_dir / "statistics.txt")

    def _save_or_show(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        if args.save:
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        else:
            plt.show()
        plt.close(fig)

    # Figure 1: Time scaling
    fig1, axes1 = plt.subplots(1, len(ptypes), figsize=(6 * len(ptypes), 5), squeeze=False)
    fig1.suptitle("Time Scaling by Algorithm", fontsize=13, fontweight="bold")
    ax_map = {ptype: axes1[0, i] for i, ptype in enumerate(ptypes)}
    fig_time_scaling(rows, ax_map.get("uniform", axes1[0, 0]), ax_map.get("clustered", axes1[0, -1]))
    _save_or_show(fig1, "time_scaling")

    # Figure 2: Solution quality
    fig2, axes2 = plt.subplots(1, len(ptypes), figsize=(6 * len(ptypes), 5), squeeze=False)
    fig2.suptitle("Solution Quality by Algorithm", fontsize=13, fontweight="bold")
    ax_map2 = {ptype: axes2[0, i] for i, ptype in enumerate(ptypes)}
    fig_solution_quality(rows, ax_map2.get("uniform", axes2[0, 0]), ax_map2.get("clustered", axes2[0, -1]))
    _save_or_show(fig2, "solution_quality")

    # Figure 3: Optimality gap line plot
    fig3, axes3 = plt.subplots(1, len(ptypes), figsize=(6 * len(ptypes), 5), squeeze=False)
    fig3.suptitle("Optimality Gap by Problem Size", fontsize=13, fontweight="bold")
    fig_gap_lines(rows, list(axes3[0]), ptypes)
    _save_or_show(fig3, "optimality_gap_lines")

    # Figure 4: Optimality gap bars
    fig4, axes4 = plt.subplots(1, len(ptypes), figsize=(7 * len(ptypes), 5), squeeze=False)
    fig4.suptitle("Optimality Gap (% above best known)", fontsize=13, fontweight="bold")
    fig_gap_bars(rows, list(axes4[0]), ptypes)
    _save_or_show(fig4, "optimality_gap_bars")

    # Figure 5: Gap heatmap
    fig5, axes5 = plt.subplots(1, len(ptypes), figsize=(5 * len(ptypes), 5), squeeze=False)
    fig5.suptitle("Mean Gap Heatmap (algorithm × size)", fontsize=13, fontweight="bold")
    fig_gap_heatmap(rows, list(axes5[0]), ptypes)
    _save_or_show(fig5, "gap_heatmap")


if __name__ == "__main__":
    main()
