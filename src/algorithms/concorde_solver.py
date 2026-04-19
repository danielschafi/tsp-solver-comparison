import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from src.algorithms.base import TSPSolver

CONCORDE_BINARY = Path(__file__).parent.parent.parent / "bin" / "concorde"

# Concorde requires integer edge weights — scale floats up to preserve precision.
_WEIGHT_SCALE = 1_000_000


class ConcordeSolver(TSPSolver):
    def __init__(self, time_limit: float | None = None):
        super().__init__("Concorde", time_limit)

    def solve_tsp(self, nodes: np.ndarray, edges: np.ndarray) -> List[int]:
        if not CONCORDE_BINARY.exists():
            raise FileNotFoundError(
                f"Concorde binary not found at {CONCORDE_BINARY}.\n"
                "Download it from https://www.math.uwaterloo.ca/tsp/concorde/ "
                "and place the executable at bin/concorde."
            )

        n = nodes.shape[0]
        int_edges = np.round(edges * _WEIGHT_SCALE).astype(int)

        with tempfile.TemporaryDirectory() as tmpdir:
            tsp_path = Path(tmpdir) / "problem.tsp"
            sol_path = Path(tmpdir) / "problem.sol"

            _write_tsplib(tsp_path, n, int_edges)

            timeout = self.time_limit if self.time_limit else 3600
            proc = subprocess.run(
                [str(CONCORDE_BINARY), "-s", "99", "-o", str(sol_path), str(tsp_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Concorde exited with code {proc.returncode}:\n{proc.stderr}"
                )

            tour = _parse_sol(sol_path)

        return tour


def _write_tsplib(path: Path, n: int, int_edges: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("NAME: problem\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in int_edges:
            f.write(" ".join(str(w) for w in row) + "\n")
        f.write("EOF\n")


def _parse_sol(sol_path: Path) -> List[int]:
    tokens = sol_path.read_text().split()
    n = int(tokens[0])
    tour = [int(t) for t in tokens[1 : n + 1]]
    tour.append(tour[0])
    return tour


def main():
    parser = argparse.ArgumentParser(
        description="Run the Concorde exact TSP solver on .npy problem files."
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--problem_id", type=str, required=False, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    solver = ConcordeSolver()

    if path.is_dir() and args.problem_id is not None:
        solver.run(str(path), args.problem_id)
    elif path.is_dir():
        files = sorted(path.glob("*_nodes.npy"))
        for i, f in enumerate(files):
            problem_id = f.stem.split("_")[0]
            print(f"Solving {f} ({i + 1}/{len(files)})")
            solver.run(str(path), problem_id)


if __name__ == "__main__":
    main()
