import argparse
import random
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver


class TwoOpt(TSPSolver):
    """
    Basic 2-opt local search for the Traveling Salesman Problem.

    The algorithm starts with a random permutation of the nodes and iteratively
    removes two edges and reconnects the tour in a shorter way if possible.
    This implementation uses first‑improvement and restarts after every successful move.
    """

    def __init__(self, time_limit: float | None = None):
        super().__init__("TwoOpt", time_limit)

    def _tour_length(self, tour: List[int], edges: np.ndarray) -> float:
        """Compute the total length of a closed tour."""
        total = 0.0
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            total += edges[a, b]
        return total

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solve the TSP using the 2-opt local search heuristic.

        Steps:
        1. Create a random initial tour.
        2. Repeatedly scan all pairs of non‑adjacent edges.
        3. If reversing the segment between them shortens the tour, apply the move
           and restart scanning.
        4. Stop when a full scan finds no improvement.
        """
        n = len(nodes)
        # 1. Random initial tour (closed loop: first city repeated at the end)
        tour = list(range(n))
        random.shuffle(tour)
        tour.append(tour[0])          # close the cycle

        improved = True
        while improved:
            improved = False
            # i is the first edge (i, i+1), j is the second edge (j, j+1)
            # We ensure edges are non-adjacent: j >= i+2
            for i in range(0, n - 2):                # up to second-last edge
                for j in range(i + 2, n):            # j can go up to last edge (j=n-1 gives edge between last and first node)
                    # Wrap-around index for j+1
                    j1 = (j + 1) % n

                    # Current two edges
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j1]

                    # Cost of current edges
                    current_cost = edges[a, b] + edges[c, d]
                    # Cost if we reconnect by reversing the segment (i+1 ... j)
                    new_cost = edges[a, c] + edges[b, d]

                    if new_cost < current_cost:
                        # Apply 2-opt move: reverse the segment from i+1 to j (inclusive)
                        tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                        improved = True
                        break          # break inner loop after a successful move
                if improved:
                    break              # break outer loop and restart scanning

        # Ensure the tour is correctly closed (already is, but for safety)
        if tour[-1] != tour[0]:
            tour.append(tour[0])

        return tour


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