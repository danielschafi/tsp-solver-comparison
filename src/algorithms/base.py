import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


class TSPSolver(ABC):
    """
    Base Class for TSP Solvers

    Ensures, that each solver can be used with a common interface, tracks metrics and provides convenience methods.
    """

    def __init__(self, algorithm: str, time_limit: float | None = None):
        self.algorithm = algorithm
        self.time_limit = time_limit
        self.result: dict = {
            "timestamp": None,
            "problem": "",
            "problem_type": "",
            "problem_size": 0,
            "algorithm": algorithm,
            "time_to_solve": 0,
            "tour_cost": 0,
            "tour": [],
            "valid_solution": None,
        }
        self._start_time = None
        self._end_time = None

        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def setup_problem(self, directory: str, problem_id: int):
        """
        Sets up the TSP Problem.

            - Use this if you need to process the data in some way, or do something else
            that is not directly related to running the algorithm
        """
        return

    @abstractmethod
    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solves one tsp instance and returns a tour containing the indices of the nodes

        !!! You probably only need to implement this !!!

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix contaiing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour that was found, include the return to the first node in your solution
        """

    def load_problem(self, directory: Path, problem_id: int):
        """Loads the nodes and edges from the npy file"""
        edges_path = Path(directory) / f"{problem_id}_edges.npy"
        nodes_path = Path(directory) / f"{problem_id}_nodes.npy"

        self.result["problem"] = str(Path(directory) / str(problem_id))

        if not Path(directory).exists():
            raise FileNotFoundError(f"Directory: {directory} does not exist.")

        if not edges_path.exists():
            raise FileNotFoundError(f"edges_path: {directory} does not exist.")
        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes_path: {directory} does not exist.")

        self.tsp_file = Path(directory)
        self.nodes = np.load(nodes_path)  # List of node coords [[x,y], [x.y], ...]
        self.edges = np.load(edges_path)  # Adjacency matrix

        self.result["problem_size"] = self.nodes.shape[0]

        self.result["problem_type"] = (
            "uniform" if "uniform" in directory else "clustered"
        )

    def run(
        self,
        directory: str,
        problem_id: int,
        plot: bool = False,
    ):
        """
        Runs the solver from start to finish on a problem instance.

        1. setup_problem
        2. solve_tsp
        3. print_solution
        4. plot_solution
        5. save_results

        Parameters
        ----------
            - directory: the direcory that contains the .npy files. eg. data/uniform/10
            - problem_id: the id of the problem file e.g. 1 -> 1_edges.npy, 1_nodes.npy belong to that problem and will be solved
        """

        self.result["timestamp"] = datetime.now().isoformat()
        self.load_problem(directory, problem_id)

        print("=" * 100)
        print(f"Algorithm: {self.algorithm}")
        print(f"Problem: {self.result['problem']}")
        print(f"Problem Size: {self.result['problem_size']}")
        print(f"Problem Type: {self.result['problem_type']}")

        print("=" * 100)

        print("1. Setting up problem")
        self.setup_problem(directory, problem_id)

        print("2. Start solving TSP")
        self._start_time = time.time()
        self.result["tour"] = self.solve_tsp(self.nodes, self.edges)
        self._end_time = time.time()

        print("3. Calculating tour cost")
        self.result["tour_cost"] = self.calculate_tour_cost()

        self.result["time_to_solve"] = self._end_time - self._start_time
        if self.result["tour"]:
            print("4. Printing tour")
            self.print_tour(self.result["tour"])
        else:
            print("4. No tour computed — skipping tour print")

        print("5. Checking validity of tour")
        self.check_solution_validity(self.result["tour"])

        print("6. Printing results")
        self.print_results()

        print("7. Saving results")
        self.save_results()
        print(f"   -> {self.results_dir / 'results.json'}")

        if plot and self.result["tour"]:
            print("8. Making plot of solution")
            self.plot_solution()

        print("Done!")

    def check_solution_validity(self, tour: list[int] | None):
        """Checks if the tour is valid.
        - Each node is visited exactly once
        - Tour starts and ends at the same node
        - All nodes in the problem are visited

        tour format is: [1,5,2,3,1]
        """

        if tour is None:
            print("Tour is empty, is invalid.")
            self.result["valid_solution"] = False
            return

        if len(set(tour)) != len(tour) - 1:
            print("Some nodes visited more than once.")
            self.result["valid_solution"] = False
            return

        # -1 because we include the return to the start in the tour
        if self.result["problem_size"] != len(tour) - 1:
            print("Not all nodes in problem were visited.")
            self.result["valid_solution"] = False
            return

        if tour[0] != tour[-1]:
            print("Tour is not finished, tour[0] must equal tour[-1]")
            self.result["valid_solution"] = False
            return

        print("Solution is valid")
        self.result["valid_solution"] = True

    def calculate_tour_cost(self, tour=None):
        """
        Calculate tour cost by summing up edge weights along the tour
        Assumes tour is a list of node indices: [0, 5, 2, 1]
        """
        assert self.edges is not None

        if tour is None and self.result["tour"] is None:
            raise ValueError(
                "Cant calculate_tour_cost, both tour parameter and self.result['tour'] are None"
            )

        if tour is None:
            tour = self.result["tour"]
        total_cost = 0
        n = len(self.result["tour"])
        for k in range(n):
            i = self.result["tour"][k]
            j = self.result["tour"][(k + 1) % n]  # Connects back to start
            total_cost += self.edges[i, j]

        return float(total_cost)

    def print_tour(self, tour: list[int]):
        """Prints the tour on the terminal"""
        route_str = str(tour[0])
        for i, node in enumerate(tour[1:]):
            if (i + 1) % 10 == 0:
                route_str += "\n "
            route_str += " -> " + str(node)
        print("Tour:\n" + route_str)

    def print_results(self):
        """Prints the solution found by the solver on the terminal"""

        print("Results:\n" + json.dumps(self.result, indent=4))

    def save_results(self):
        """Saves the results to a shared JSON file in the results directory.

        Keyed on (algorithm, problem): re-running the same instance overwrites the previous result.
        Writes atomically via a temp file to avoid corrupting the results on error.
        """
        results_file = self.results_dir / "results.json"
        tmp_file = results_file.with_suffix(".tmp")

        # if the problem already has an entry overwrite that one
        if results_file.exists():
            with open(results_file) as f:
                records = json.load(f)
            records = [
                r
                for r in records
                if not (
                    r["algorithm"] == self.result["algorithm"]
                    and r["problem"] == self.result["problem"]
                )
            ]  # read only records that are not our new one
        else:
            records = []

        records.append(self.result)

        with open(tmp_file, "w") as f:
            json.dump(records, f, indent=2)
        tmp_file.replace(results_file)

    def plot_solution(self):
        """Plots the found solution"""

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot tour path
        tour = self.result["tour"]
        for i in range(len(tour) - 1):
            start = self.nodes[tour[i]]
            end = self.nodes[tour[i + 1]]
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], color="darkred", linewidth=2
            )

        # Plot nodes in blue
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color="blue", s=50, zorder=5)

        ax.set_title(
            f"{self.algorithm} | {self.result['problem']} | Cost: {self.result['tour_cost']:.2f}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
        plt.savefig(
            self.results_dir
            / f"{self.result['problem_type']}_{self.result['problem_size']}_{self.algorithm}_solution.png"
        )
        plt.close()
