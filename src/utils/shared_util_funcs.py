from src.algorithms.greedy import Greedy


def get_greeedy_initial_solution(nodes, edges):
    """Returns a greedy solution for the given nodes and edges. Used for initializing other algorithms."""
    greedy_solver = Greedy()
    return greedy_solver.solve_tsp(nodes, edges)
