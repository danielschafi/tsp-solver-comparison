import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver
from src.utils.shared_util_funcs import apply_two_opt_improvement, get_greeedy_initial_solution


class IterativeLocalSearch(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("IterativeLocalSearch", time_limit)
        self._initial_tour: list = []

    def setup_problem(self, directory: str, problem_id: int):
        super().setup_problem(directory, problem_id)
        tour = get_greeedy_initial_solution(self.nodes, self.edges)[:-1]
        self._initial_tour = apply_two_opt_improvement(tour, self.edges)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solves the TSP using Iterated Local Search (ILS).

        Reference: Lourenço, H.R., Martin, O.C., Stützle, T. (2019).
        Iterated Local Search: Framework and Applications.
        In: Handbook of Metaheuristics (3rd ed.).
        https://doi.org/10.1007/978-3-319-91086-4_4

        Algorithm:
            function ILS(problem):
                current ← LOCAL_SEARCH(GREEDY_SOLUTION(problem))
                best ← current

                for i = 1 to MAX_ITERATIONS:
                    perturbed  ← DOUBLE_BRIDGE(current)
                    candidate  ← LOCAL_SEARCH(perturbed)

                    if COST(candidate) < COST(best):
                        best ← candidate

                    if COST(candidate) < COST(current):   // accept if improving
                        current ← candidate

                return best

        Steps:
            1. Build a greedy initial tour and apply 2-opt to reach a local optimum
               (done in setup_problem, excluded from timing).
            2. Repeat for MAX_ITERATIONS:
               a. Perturb current solution with a double-bridge move — a 4-opt move
                  that cannot be undone by 2-opt, ensuring escape from local optima.
               b. Apply exhaustive 2-opt improvement to the perturbed tour.
               c. Update best if the candidate improves on it.
               d. Accept the candidate as the new current if it improves on current
                  (simple hill-climbing acceptance).
            3. Return the best tour found.

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix containing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour found, including the return to the first node
        """
        MAX_ITERATIONS = 200

        current = self._initial_tour[:]
        best = current[:]

        for _ in range(MAX_ITERATIONS):
            perturbed = self._double_bridge(current)
            candidate = apply_two_opt_improvement(perturbed, edges)

            if self.calculate_tour_cost(candidate) < self.calculate_tour_cost(best):
                best = candidate[:]

            if self.calculate_tour_cost(candidate) < self.calculate_tour_cost(current):
                current = candidate

        return best + [best[0]]

    def _double_bridge(self, tour: List[int]) -> List[int]:
        """
        Perturbs the tour with a double-bridge (4-opt) move.

        Splits the tour into four segments at three random cut points and
        reconnects them as A+C+B+D — a reconnection that 2-opt cannot reverse,
        guaranteeing escape from the current local optimum.
        """
        n = len(tour)
        cuts = sorted(np.random.choice(range(1, n), size=3, replace=False))
        a, b, c = cuts
        return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the Iterative Local Search solver on a TSP instance."
    )
    arg_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing the TSP files.",
    )
    arg_parser.add_argument(
        "--problem_id",
        type=str,
        required=False,
        default=None,
        help="ID of the problem to solve.",
    )

    args = arg_parser.parse_args()
    path = Path(args.path)

    if path.is_dir() and args.problem_id is not None:
        solver = IterativeLocalSearch()
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        files = sorted(path.glob("*_edges.npy"))
        solver = IterativeLocalSearch()
        for edges_file in files:
            problem_id = int(edges_file.stem.split("_")[0])
            print(f"Solving problem {problem_id} from {path}")
            solver.run(str(path), problem_id)


if __name__ == "__main__":
    main()
