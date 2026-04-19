import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.algorithms.base import TSPSolver
from src.utils.shared_util_funcs import get_greeedy_initial_solution


class TabuSearch(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("TabuSearch", time_limit)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solves the TSP using Tabu Search.

        Reference: Glover, F. (1989). Tabu Search — Part I.
        ORSA Journal on Computing, 1(3), 190-206.
        https://doi.org/10.1287/ijoc.1.3.190

        Pseudocode:
            function TABU_SEARCH(problem):
                current ← GREEDY_SOLUTION(problem)
                best ← current
                tabu_list ← empty queue of capacity TABU_TENURE

                for iteration = 1 to MAX_ITERATIONS:
                    best_neighbor ← None
                    best_neighbor_cost ← ∞
                    best_move ← None

                    for each move (i, j) in NEIGHBORHOOD(current):
                        neighbor ← APPLY_2OPT_MOVE(current, i, j)
                        cost ← COST(neighbor)
                        is_tabu ← (i, j) in tabu_list

                        if (not is_tabu) or cost < COST(best):   // aspiration criterion
                            if cost < best_neighbor_cost:
                                best_neighbor ← neighbor
                                best_neighbor_cost ← cost
                                best_move ← (i, j)

                    current ← best_neighbor
                    tabu_list.enqueue(best_move)

                    if COST(current) < COST(best):
                        best ← current

                return best

        Steps:
            1. Build a greedy initial tour as the starting solution.
            2. Repeat for MAX_ITERATIONS:
               a. Enumerate the full 2-opt neighborhood of the current solution.
               b. Select the best neighbor whose generating move (i, j) is either
                  not in the tabu list, or satisfies the aspiration criterion
                  (its cost beats the global best, overriding the tabu status).
               c. Apply that move to become the new current solution and add
                  the move to the tabu list. Evict the oldest move once the
                  list exceeds TABU_TENURE.
               d. Update the global best if the new current is an improvement.
            3. Return the best tour found.

        Tabu tenure: sqrt(n), which balances diversification and intensification.
        Neighborhood: all O(n²) 2-opt swaps of the current tour.

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix containing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour found, including the return to the first node
        """
        n = len(nodes)
        MAX_ITERATIONS = max(50, 5000 // n)
        tabu_tenure = max(7, int(np.sqrt(n)))

        current = get_greeedy_initial_solution(nodes, edges)[:-1]
        best = current[:]
        best_cost = self.calculate_tour_cost(best)

        tabu_list: List[Tuple[int, int]] = []

        for _ in range(MAX_ITERATIONS):
            best_neighbor = None
            best_neighbor_cost = float("inf")
            best_move: Tuple[int, int] | None = None

            for i in range(n - 1):
                for j in range(i + 1, n):
                    neighbor = current[:]
                    neighbor[i : j + 1] = reversed(neighbor[i : j + 1])
                    cost = self.calculate_tour_cost(neighbor)

                    is_tabu = (i, j) in tabu_list
                    aspiration = cost < best_cost

                    if (not is_tabu or aspiration) and cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_move = (i, j)

            if best_neighbor is None or best_move is None:
                break

            current = best_neighbor
            tabu_list.append(best_move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            if best_neighbor_cost < best_cost:
                best = best_neighbor[:]
                best_cost = best_neighbor_cost

        return best + [best[0]]


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the Tabu Search solver on a TSP instance."
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
        solver = TabuSearch()
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        files = sorted(path.glob("*_edges.npy"))
        solver = TabuSearch()
        for edges_file in files:
            problem_id = int(edges_file.stem.split("_")[0])
            print(f"Solving problem {problem_id} from {path}")
            solver.run(str(path), problem_id)


if __name__ == "__main__":
    main()
