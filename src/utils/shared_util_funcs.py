from typing import List

import numpy as np

from src.algorithms.greedy import Greedy


def get_greeedy_initial_solution(nodes, edges) -> List[int]:
    """Returns a greedy solution for the given nodes and edges. Used for initializing other algorithms."""
    greedy_solver = Greedy()
    return greedy_solver.solve_tsp(nodes, edges)


def two_opt_move(tour: List[int]) -> List[int]:
    """Applies a single random 2-opt move and returns the new open tour."""
    n = len(tour)
    next_tour = tour[:]

    i = np.random.randint(n)
    j = np.random.randint(n)
    if i > j:
        i, j = j, i

    next_tour[i : j + 1] = reversed(next_tour[i : j + 1])
    return next_tour
