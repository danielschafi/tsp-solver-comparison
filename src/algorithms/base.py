import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            "problem_file": "",
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

    @abstractmethod
    def setup_problem(self, tsp_file):
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

    def load_problem(self, tsp_file: str):
        """Loads the nodes and edges from the npy file"""
        if not Path(tsp_file).exists():
            raise FileNotFoundError(f"tsp_file: {tsp_file} does not exist.")

        self.tsp_file = Path(tsp_file)
        problem = np.load(self.tsp_file)
        self.nodes = problem[0]  # List of node coords [[x,y], [x.y], ...]
        self.edges = problem[1]  # Adjacency matrix

    def run(
        self,
        tsp_file: str,
        plot: bool = True,
    ):
        """
        Runs the solver from start to finish on a problem instance.

        1. setup_problem
        2. solve_tsp
        3. print_solution
        4. plot_solution
        5. save_results
        """

        self.load_problem(tsp_file)

        print("=" * 100)
        print(f"Algorithm: {self.algorithm}")
        print(f"Problem: {self.result['problem_file']}")
        print(f"Problem Size: {self.result['problem_size']}")
        print(f"Problem Type: {self.result['problem_type']}")

        print("=" * 100)

        print("Setting up problem")
        self.setup_problem(tsp_file)

        print("Start solving TSP")
        self.solve_tsp()

        if self.result["tour"]:
            print("Printing tour")
            self.print_tour(self.result["tour"])
        else:
            print("No tour computed — skipping tour print")

        print("Checking validity of tour")
        self.check_solution_validity(self.result["tour"])

        print("Printing results")
        self.print_results()

        print("Saving results")
        self.save_results()

        if plot and self.result["tour"]:
            print("Making plot of solution")
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

    def calculate_tour_cost(self, tour):
        """
        Calculate tour cost by summing up edge weights along the tour
        Assumes tour is a list of node indices: [0, 5, 2, 1]
        """
        assert self.edges is not None
        total_cost = 0
        n = len(tour)
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]  # Connects back to start
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
        """Saves the results to a shared JSON file in the results directory, appending as a new row."""

        results_file = self.results_dir / "results.json"

        if results_file.exists():
            df = pd.read_json(results_file, orient="records")
            df = pd.concat([df, pd.DataFrame([self.result])], ignore_index=True)
        else:
            df = pd.DataFrame([self.result])

        df.to_json(results_file, orient="records", indent=2)

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

        ax.set_title(f"{self.algorithm} - Cost: {self.result['tour_cost']:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
        plt.savefig(
            self.results_dir
            / f"{self.result['problem_type']}_{self.result['problem_size']}_{self.algorithm}_solution.png"
        )
        plt.close()
