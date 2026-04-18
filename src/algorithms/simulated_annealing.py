import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver
from src.utils.shared_util_funcs import get_greeedy_initial_solution

np.random.seed(42)


class SimulatedAnnealing(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("SimulatedAnnealing", time_limit)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solves one tsp instance and returns a tour containing the indices of the nodes

        Uses the SimulatedAnnealing algorithm

        1. sample initial node
        2. while not all nodes visited
            get closest univisited node to current node
            add that node to the tour


        function SIMULATED_ANNEALING(problem, schedule):
            current ← INITIAL_SOLUTION(problem)
            best ← current

            for t = 1 to MAX_ITERATIONS:
                T ← schedule(t)          // temperature at time t
                if T = 0:
                    return best

                next ← RANDOM_NEIGHBOR(current)
                ΔE ← COST(next) - COST(current)

                if ΔE < 0:
                    current ← next        // always accept improvements
                    if COST(current) < COST(best):
                        best ← current
                else:
                    with probability exp(−ΔE / T):
                        current ← next    // sometimes accept worse solutions

            return best

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix contaiing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour that was found, include the return to the first node in your solution
        """
        size = nodes.shape[0]
        MAX_ITERATIONS = 10000

        current = get_greeedy_initial_solution(nodes, edges)
        best = current

        print(best)

        for t in range(MAX_ITERATIONS):
            print(t)
            T = self.schedule(t)

            # Fully cooled down return best found solution
            if T == 0:
                return best

            next = self.two_opt(current, edges)

            delta = self.calculate_tour_cost(
                next + [next[0]]
            ) - self.calculate_tour_cost(current + [current[0]])

            print(delta)

            if delta < 0:
                current = next
                if self.calculate_tour_cost(
                    current + [current[0]]
                ) < self.calculate_tour_cost(best + [best[0]]):
                    best = current

            else:
                if np.random.binomial(1, np.exp(-delta / T), 1) == 1:
                    current = next

        # append wrapparound
        return best + [best[0]]

    def two_opt(self, tour, edges):
        n = len(tour)

        i = np.random.randint(n)
        j = np.random.randint(n)

        # reverse segment between i and j

        tour[i : j + 1] = reversed(tour[i : j + 1])

        return tour

    def schedule(self, T: int) -> float:
        """Returns the temperature at time t. according to geometric schedule"""
        return 0.99**T


def main():
    arg_parser = argparse.ArgumentParser(
        description="run the greedy solver on a .npy file or a folder containing .npy files "
    )
    arg_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing the tsp files.",
    )

    arg_parser.add_argument(
        "--problem_id",
        type=str,
        required=False,
        default=None,
        help="index of the problem to solve. So if --path data/uniform/10 is passed, then you can set the problem-id 1 to solve the problem with the id 1. where the problem consists of the two files [id]_edges.npy and [id]_nodes.npy",
    )

    args = arg_parser.parse_args()
    path = Path(args.path)

    # check if it is a file or a folder
    if path.is_dir() and args.problem_id is not None:
        solver = SimulatedAnnealing()
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        files = sorted(path.glob("*.npy"))
        solver = SimulatedAnnealing()
        for i, tsp_file in enumerate(files):
            problem_id = tsp_file.stem.split("_")[0]
            print(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver.run(str(path), problem_id)


if __name__ == "__main__":
    main()
