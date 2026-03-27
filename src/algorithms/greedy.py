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

        1. sample initial node
        2. while not all nodes visited
            get closest univisited node to current node
            add that node to the tour

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix contaiing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour that was found, include the return to the first node in your solution
        """
        size = nodes.shape[0]

        visited_mask = np.zeros(size)  # mask out the already visited nodes

        # start node random

        rng = np.random.default_rng()
        current_node = rng.integers(low=0, high=size)
        tour = [int(current_node)]

        visited_mask[current_node] = 1
        while len(tour) < size:
            current_node = tour[-1]
            # unvisited indices
            unvisited_indices = np.where(visited_mask == 0)[0]

            # filter edges of current node to include only the unvisited ones
            filtered_edges = edges[current_node][unvisited_indices]

            # index of the minimum edge
            min_filtered_idx = np.argmin(filtered_edges)

            # get which edge this corresponds to in the original list
            next_node = unvisited_indices[min_filtered_idx]

            tour.append(int(next_node))
            visited_mask[next_node] = 1

        tour.append(tour[0])  # loop around to start
        return tour


def main():
    arg_parser = argparse.ArgumentParser(
        description="run the greedy solver on a .npy file "
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
        solver = Greedy()
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        files = sorted(path.rglob("*.npy"))
        solver = Greedy()
        for i, tsp_file in enumerate(files):
            problem_id = tsp_file.stem.split("_")[0]
            print(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver.run(str(path), problem_id)


if __name__ == "__main__":
    main()
