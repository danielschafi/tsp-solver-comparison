"""Compute and print aggregated statistics from benchmark results.

Usage:
    uv run -m src.utils.compute_statistics [--csv results/statistics.csv]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: Path = Path("results/results.json")) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def best_per_instance(records: list[dict]) -> dict[tuple, float]:
    """Returns the lowest tour_cost found for each (problem, problem_type, problem_size)."""
    best: dict[tuple, float] = {}
    for r in records:
        if not r.get("valid_solution", True):
            continue
        key = (r["problem"], r["problem_type"], r["problem_size"])
        if key not in best or r["tour_cost"] < best[key]:
            best[key] = r["tour_cost"]
    return best


def compute_stats(records: list[dict]) -> list[dict]:
    """Aggregate records into per-(algorithm, problem_type, problem_size) statistics."""
    best = best_per_instance(records)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["algorithm"], r["problem_type"], r["problem_size"])
        groups[key].append(r)

    rows = []
    for (algorithm, ptype, size), recs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        valid = [r for r in recs if r.get("valid_solution", True)]
        costs = np.array([r["tour_cost"] for r in valid])
        times = np.array([r["time_to_solve"] for r in valid])

        gaps = []
        for r in valid:
            inst_key = (r["problem"], r["problem_type"], r["problem_size"])
            b = best.get(inst_key)
            if b and b > 0:
                gaps.append((r["tour_cost"] - b) / b * 100)

        rows.append({
            "algorithm": algorithm,
            "problem_type": ptype,
            "problem_size": size,
            "n_instances": len(recs),
            "n_valid": len(valid),
            "mean_cost": float(np.mean(costs)) if len(costs) else float("nan"),
            "std_cost": float(np.std(costs)) if len(costs) else float("nan"),
            "min_cost": float(np.min(costs)) if len(costs) else float("nan"),
            "max_cost": float(np.max(costs)) if len(costs) else float("nan"),
            "mean_time_s": float(np.mean(times)) if len(times) else float("nan"),
            "std_time_s": float(np.std(times)) if len(times) else float("nan"),
            "mean_gap_pct": float(np.mean(gaps)) if gaps else float("nan"),
            "std_gap_pct": float(np.std(gaps)) if gaps else float("nan"),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Algorithm':<25} {'Type':<10} {'n':>4} {'Size':>5}  "
        f"{'Mean Cost':>11} {'Std Cost':>10}  "
        f"{'Mean Time(s)':>12} {'Std Time(s)':>11}  "
        f"{'Mean Gap%':>10} {'Std Gap%':>9}"
    )
    sep = "-" * len(header)

    current_type = None
    for row in rows:
        if row["problem_type"] != current_type:
            current_type = row["problem_type"]
            print(f"\n=== {current_type.upper()} INSTANCES ===")
            print(header)
            print(sep)

        def fmt(v, w=11, d=4):
            return f"{v:{w}.{d}f}" if not np.isnan(v) else f"{'N/A':>{w}}"

        print(
            f"{row['algorithm']:<25} {row['problem_type']:<10} "
            f"{row['n_valid']:>4} {row['problem_size']:>5}  "
            f"{fmt(row['mean_cost'])} {fmt(row['std_cost'])}  "
            f"{fmt(row['mean_time_s'], 12, 6)} {fmt(row['std_time_s'], 11, 6)}  "
            f"{fmt(row['mean_gap_pct'], 10, 2)} {fmt(row['std_gap_pct'], 9, 2)}"
        )


def save_csv(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved statistics to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute aggregated benchmark statistics.")
    parser.add_argument("--results", default="results/results.json", help="Path to results JSON.")
    parser.add_argument("--csv", default=None, metavar="PATH", help="Save stats to CSV file.")
    args = parser.parse_args()

    records = load_results(Path(args.results))
    rows = compute_stats(records)
    print_table(rows)

    if args.csv:
        save_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
