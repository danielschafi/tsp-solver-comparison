import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver

np.random.seed(42)


class Greedy(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("Greedy", time_limit)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solves one tsp instance and returns a tour containing the indices of the nodes

        Uses the greedy algorithm

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix contaiing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour that was found, include the return to the first node in your solution
        """
        tour = []
        size = self.result["problem_size"]

        visited_mask = np.zeros(size)  # mask out the already visited nodes

        # start node random
        current_node = np.random.choice(range(size))
        tour.append(current_node)
        nodes_remaining = size - 1

        visited_mask[current_node] = 1

        while len(tour) < size:
            current_node = tour[-1]
            # unvisited indices
            unvisited_indices = np.where(visited_mask == 0)[0]

            # filter edges of current node to include only the unvisited ones
            filtered_edges = self.edges[current_node][unvisited_indices]

            # index of the minimum edge
            min_filtered_idx = np.argmin(filtered_edges)

            # get which edge this corresponds to in the original list
            next_node = unvisited_indices[min_filtered_idx]

            tour.append(next_node)

        return tour


def main():
    arg_parser = argparse.ArgumentParser(
        description="run the greedy solver on a .npy file "
    )
    arg_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the .tsp file to solve.",
    )

    args = arg_parser.parse_args()
    path = Path(args.path)

    # check if it is a file or a folder
    if path.is_file():
        solver = Greedy()
        solver.run(str(path))
    elif path.is_dir():
        files = sorted(path.rglob("*.npy"))
        solver = Greedy()
        for i, tsp_file in enumerate(files):
            print(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver.run(str(tsp_file))


if __name__ == "__main__":
    main()
