import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver
from src.utils.shared_util_funcs import apply_two_opt_improvement, get_greeedy_initial_solution


class TwoOpt(TSPSolver):
    """
    Basic 2-opt local search for the Traveling Salesman Problem.

    Starts from a greedy initial solution and iteratively removes two edges and
    reconnects the tour in a shorter way. Uses first-improvement and restarts
    after every successful move.
    """

    def __init__(self, time_limit: float | None = None):
        super().__init__("TwoOpt", time_limit)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        # Greedy returns a closed tour; strip the closing node for improvement
        tour = get_greeedy_initial_solution(nodes, edges)[:-1]
        tour = apply_two_opt_improvement(tour, edges)
        return tour + [tour[0]]


def main():
    parser = argparse.ArgumentParser(
        description="Run the 2-opt solver on a TSP instance."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing the TSP files.",
    )
    parser.add_argument(
        "--problem_id",
        type=int,
        required=False,
        default=None,
        help="ID of the problem to solve (e.g., 29).",
    )
    args = parser.parse_args()

    path = Path(args.path)
    solver = TwoOpt()

    if path.is_dir() and args.problem_id is not None:
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        # Solve all instances in the directory
        files = sorted(path.glob("*_edges.npy"))
        for edges_file in files:
            problem_id = int(edges_file.stem.split("_")[0])
            print(f"Solving problem {problem_id} from {path}")
            solver.run(str(path), problem_id)
    else:
        print("Error: --path must be a directory.")


if __name__ == "__main__":
    main()