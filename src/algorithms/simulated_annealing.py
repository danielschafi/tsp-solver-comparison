import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver
from src.utils.shared_util_funcs import get_greeedy_initial_solution, two_opt_move

class SimulatedAnnealing(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("SimulatedAnnealing", time_limit)
        self.T_0: float = 0.1
    def setup_problem(self, directory: str, problem_id: int):
        super().setup_problem(directory, problem_id)
        random_tour = list(np.random.permutation(len(self.nodes)))
        self.T_0 = self._estimate_initial_temperature(random_tour)

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
        MAX_ITERATIONS = 100000

        current = get_greeedy_initial_solution(nodes, edges)[:-1]
        best = current[:]

        for t in range(MAX_ITERATIONS):
            T = self.schedule(t)

            if T < 1e-10:
                return best + [best[0]]

            next_tour = two_opt_move(current)

            delta = self.calculate_tour_cost(next_tour) - self.calculate_tour_cost(
                current
            )

            if delta < 0:
                current = next_tour
                if self.calculate_tour_cost(current) < self.calculate_tour_cost(best):
                    best = current[:]

            else:
                if np.random.binomial(1, np.exp(-delta / T), 1) == 1:
                    current = next_tour

        return best + [best[0]]

    def _estimate_initial_temperature(
        self, tour: List[int], n_samples: int = 200, target_acceptance: float = 0.6
    ) -> float:
        """Sample random 2-opt deltas to set T_0 so ~60% of bad moves are accepted initially."""
        deltas = []
        for _ in range(n_samples):
            neighbor = two_opt_move(tour)
            delta = self.calculate_tour_cost(neighbor) - self.calculate_tour_cost(tour)
            if delta > 0:
                deltas.append(delta)
        mean_delta = np.mean(deltas) if deltas else 0.1
        return mean_delta / np.log(1 / target_acceptance)

    def schedule(self, t: int) -> float:
        """Geometric cooling: T_0 * 0.9999^t, calibrated to the problem scale."""
        return self.T_0 * (0.9999**t)


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
