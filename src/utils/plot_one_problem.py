# 1. read results/results.json
# 2. get the record that matches the arguments "problem" and "algorithm" get "tour" etc from that record
# 3. read the node coords file at the "problem"_nodes.npy path
# 4. plot the "tour"
# 5. show the tour

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot a TSP solution tour.")
    parser.add_argument(
        "--problem",
        help='Problem path, e.g. "data/uniform/10/0_edges.npy" also allows abosulte path',
    )
    parser.add_argument("--algorithm", help='Algorithm name, e.g. "Greedy"')
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="If True, saves the plot",
    )
    args = parser.parse_args()

    problem = args.problem
    problem = problem.split("data/")[-1]
    problem = "data/" + problem

    suffixes_to_remove = ["_edges.npy", "_nodes.npy"]
    for suffix in suffixes_to_remove:
        if problem.endswith(suffix):
            problem = problem[: -len(suffix)]

    results_file = Path("results/results.json")
    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found.")

    with open(results_file) as f:
        records = json.load(f)

    match = next(
        (
            r
            for r in records
            if r["problem"] == problem and r["algorithm"] == args.algorithm
        ),
        None,
    )
    if match is None:
        raise ValueError(
            f"No record found for problem='{problem}' algorithm='{args.algorithm}'"
        )

    nodes_path = Path(f"{problem}_nodes.npy")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

    nodes = np.load(nodes_path)
    tour = match["tour"]

    _, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(tour) - 1):
        start = nodes[tour[i]]
        end = nodes[tour[i + 1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], color="darkred", linewidth=2)

    ax.scatter(nodes[:, 0], nodes[:, 1], color="blue", s=50, zorder=5)

    for idx, (x, y) in enumerate(nodes):
        ax.annotate(
            str(idx), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8
        )

    ax.set_title(f"{args.algorithm} | {problem} | Cost: {match['tour_cost']:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    if args.save:
        plt.savefig(f"{args.algorithm}_{problem.replace('/', '_')}.png")
    print(f"Saving to: {args.algorithm}_{problem.replace('/', '_')}.png")
    plt.show()


if __name__ == "__main__":
    main()
