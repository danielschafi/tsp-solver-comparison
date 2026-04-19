from typing import List

import numpy as np

from src.algorithms.greedy import Greedy


def get_greeedy_initial_solution(nodes, edges) -> List[int]:
    """Returns a greedy solution for the given nodes and edges. Used for initializing other algorithms."""
    greedy_solver = Greedy()
    return greedy_solver.solve_tsp(nodes, edges)


def apply_two_opt_improvement(tour: List[int], edges: np.ndarray) -> List[int]:
    """
    Exhaustive 2-opt improvement until no improving move exists (first-improvement).
    Accepts and returns an open tour (without the closing duplicate of the first node).
    """
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                j1 = (j + 1) % n
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[j1]
                if edges[a, c] + edges[b, d] < edges[a, b] + edges[c, d]:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break
    return tour


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
