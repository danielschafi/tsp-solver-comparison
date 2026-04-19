import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver
from src.utils.shared_util_funcs import get_greeedy_initial_solution


POPULATION_SIZE = 100
CROSSOVER_RATE = 0.9
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 2


class Genetic(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("Genetic", time_limit)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        """
        Solves the TSP using a standard Genetic Algorithm with Order Crossover (OX).

        Reference: Larranaga, P., Kuijpers, C.M.H., Murga, R.H., Inza, I., Dizdarevic, S. (1999).
        Genetic Algorithms for the Travelling Salesman Problem: A Review of Representations
        and Operators. Artificial Intelligence Review, 13(2), 129-170.
        https://doi.org/10.1023/A:1006529012972

        Pseudocode:
            function GENETIC_ALGORITHM(problem):
                population ← INITIALIZE_POPULATION(problem)
                best ← argmin(COST(p) for p in population)

                for generation = 1 to MAX_GENERATIONS:
                    new_population ← ELITES(population, ELITISM_COUNT)

                    while |new_population| < POPULATION_SIZE:
                        parent1 ← TOURNAMENT_SELECT(population)
                        parent2 ← TOURNAMENT_SELECT(population)

                        if random() < CROSSOVER_RATE:
                            child ← ORDER_CROSSOVER(parent1, parent2)
                        else:
                            child ← copy(parent1)

                        if random() < MUTATION_RATE:
                            child ← INVERSION_MUTATE(child)

                        new_population ← new_population ∪ {child}

                    population ← new_population
                    candidate ← argmin(COST(p) for p in population)
                    if COST(candidate) < COST(best):
                        best ← candidate

                return best

        Steps:
            1. Initialize population: one greedy tour + (POPULATION_SIZE - 1) random permutations
               (done in setup_problem, excluded from timing).
            2. Repeat for MAX_GENERATIONS:
               a. Carry the top ELITISM_COUNT tours unchanged into the next generation.
               b. Fill remaining slots by selecting two parents via tournament selection,
                  applying Order Crossover (OX) with probability CROSSOVER_RATE,
                  then inversion mutation with probability MUTATION_RATE.
               c. Track the best tour seen across all generations.
            3. Return the best tour found.

        Operators:
            - Tournament selection: pick TOURNAMENT_SIZE random individuals, return the best.
            - Order Crossover (OX): copy a random segment from parent1 into the child,
              then fill remaining positions with cities from parent2 in their original order,
              skipping cities already placed. Preserves relative order from both parents.
            - Inversion mutation: reverse a random segment of the tour.

        Parameters
        ----------
            - nodes: the coordinates of the nodes
            - edges: the adjacency matrix containing the distances between the nodes
        Returns
        ----------
            - List[int]: The tour found, including the return to the first node
        """
        n = len(nodes)
        generations = max(100, 10 * n)
        mutation_rate = 1 / n

        population = self._initialize_population(nodes, edges)
        best = min(population, key=lambda t: self.calculate_tour_cost(t))

        for _ in range(generations):
            costs = [self.calculate_tour_cost(t) for t in population]
            sorted_indices = np.argsort(costs)

            new_population = [population[i][:] for i in sorted_indices[:ELITISM_COUNT]]

            while len(new_population) < POPULATION_SIZE:
                parent1 = self._tournament_select(population, costs)
                parent2 = self._tournament_select(population, costs)

                if np.random.random() < CROSSOVER_RATE:
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = parent1[:]

                if np.random.random() < mutation_rate:
                    child = self._inversion_mutate(child)

                new_population.append(child)

            population = new_population
            candidate = min(population, key=lambda t: self.calculate_tour_cost(t))
            if self.calculate_tour_cost(candidate) < self.calculate_tour_cost(best):
                best = candidate[:]

        return [int(x) for x in best] + [int(best[0])]

    def _initialize_population(self, nodes: np.ndarray, edges: np.ndarray) -> List[List[int]]:
        """One greedy tour plus random permutations to fill the rest of the population."""
        n = nodes.shape[0]
        greedy = get_greeedy_initial_solution(nodes, edges)[:-1]
        population = [greedy]
        while len(population) < POPULATION_SIZE:
            population.append(list(np.random.permutation(n)))
        return population

    def _tournament_select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Return the lowest-cost individual from a random sample of TOURNAMENT_SIZE."""
        indices = np.random.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
        best_idx = min(indices, key=lambda i: costs[i])
        return population[best_idx]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX): copy a random segment from parent1, then fill the
        remaining positions with cities from parent2 in their original order.
        """
        n = len(parent1)
        i, j = sorted(np.random.choice(n, size=2, replace=False))

        child = [-1] * n
        child[i : j + 1] = parent1[i : j + 1]
        placed = set(child[i : j + 1])

        fill = [city for city in parent2 if city not in placed]
        idx = 0
        for k in list(range(j + 1, n)) + list(range(0, i)):
            child[k] = fill[idx]
            idx += 1

        return child

    def _inversion_mutate(self, tour: List[int]) -> List[int]:
        """Reverse a random segment of the tour."""
        n = len(tour)
        i, j = sorted(np.random.choice(n, size=2, replace=False))
        tour = tour[:]
        tour[i : j + 1] = reversed(tour[i : j + 1])
        return tour


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the Genetic Algorithm solver on a TSP instance."
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
        solver = Genetic()
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        files = sorted(path.glob("*_edges.npy"))
        solver = Genetic()
        for edges_file in files:
            problem_id = int(edges_file.stem.split("_")[0])
            print(f"Solving problem {problem_id} from {path}")
            solver.run(str(path), problem_id)


if __name__ == "__main__":
    main()
