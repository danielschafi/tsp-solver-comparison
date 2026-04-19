"""Benchmark all TSP solvers across problem instances.

Usage:
    uv run -m src.benchmark [--clean] [--type uniform|clustered|all]
                            [--sizes 10 25 50] [--algorithms greedy two_opt ...]
                            [--skip-concorde] [--workers N]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.algorithms.concorde_solver import ConcordeSolver
from src.algorithms.genetic import Genetic
from src.algorithms.greedy import Greedy
from src.algorithms.iterative_local_search import IterativeLocalSearch
from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.algorithms.tabu_search import TabuSearch
from src.algorithms.two_opt import TwoOpt

SOLVERS: dict[str, type] = {
    "greedy": Greedy,
    "two_opt": TwoOpt,
    "simulated_annealing": SimulatedAnnealing,
    "tabu_search": TabuSearch,
    "genetic": Genetic,
    "iterative_local_search": IterativeLocalSearch,
    "concorde": ConcordeSolver,
}

SIZES = [10, 25, 50, 100, 200, 500]
TYPES = ["uniform", "clustered"]
RESULTS_FILE = Path("results/results.json")

# Maps SOLVERS key → algorithm name stored in results
ALGORITHM_NAMES: dict[str, str] = {
    "greedy": "Greedy",
    "two_opt": "TwoOpt",
    "simulated_annealing": "SimulatedAnnealing",
    "tabu_search": "TabuSearch",
    "genetic": "Genetic",
    "iterative_local_search": "IterativeLocalSearch",
    "concorde": "Concorde",
}


def run_instance(args: tuple) -> dict:
    """Module-level so ProcessPoolExecutor can pickle it."""
    solver_name, path, problem_id = args
    return SOLVERS[solver_name]().run(path, problem_id, write_results=False)


def get_problem_ids(path: Path) -> list[str]:
    return sorted({f.stem.split("_")[0] for f in path.glob("*_nodes.npy")}, key=int)


def save(new_results: list[dict], existing: list[dict]) -> None:
    new_keys = {(r["algorithm"], r["problem"]) for r in new_results}
    all_results = [
        r for r in existing if (r["algorithm"], r["problem"]) not in new_keys
    ] + new_results

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2)
    tmp.replace(RESULTS_FILE)

    df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "tour"} for r in all_results]
    )
    df.to_csv(RESULTS_FILE.with_suffix(".csv"), index=False)
    print(f"Saved {len(all_results)} result(s) → {RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TSP solvers in parallel.")
    parser.add_argument(
        "--clean", action="store_true", help="Discard all existing results."
    )
    parser.add_argument(
        "--type",
        choices=["uniform", "clustered", "all"],
        default="all",
        dest="instance_type",
        help="Instance type to benchmark (default: all).",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES, metavar="N")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(SOLVERS),
        default=None,
        metavar="ALG",
        help="Algorithms to run (default: all).",
    )
    parser.add_argument("--skip-concorde", action="store_true", help="Skip Concorde.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), metavar="N")
    parser.add_argument(
        "--save-every", type=int, default=50, metavar="N",
        help="Flush results to disk every N completions (default: 50).",
    )
    args = parser.parse_args()

    types = TYPES if args.instance_type == "all" else [args.instance_type]
    solver_names = args.algorithms or list(SOLVERS)
    if args.skip_concorde:
        solver_names = [s for s in solver_names if s != "concorde"]

    existing = (
        []
        if args.clean
        else (json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else [])
    )

    existing_keys = {(r["algorithm"], r["problem"]) for r in existing}

    all_tasks = [
        (name, str(path), pid)
        for t in types
        for size in args.sizes
        for path in [Path(f"data/{t}/{size}")]
        if path.exists()
        for pid in get_problem_ids(path)
        for name in solver_names
    ]
    tasks = [
        t for t in all_tasks
        if (ALGORITHM_NAMES[t[0]], str(Path(t[1]) / t[2])) not in existing_keys
    ]

    skipped = len(all_tasks) - len(tasks)
    if skipped:
        print(f"Skipping {skipped} already-computed task(s).")
    print(f"{len(tasks)} task(s) to run across {args.workers} worker(s)\n")

    new_results: list[dict] = []
    failed = 0
    width = len(str(len(tasks)))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_instance, t): t for t in tasks}
        for future in as_completed(future_to_task):
            name, path, pid = future_to_task[future]
            n = len(new_results) + failed + 1
            try:
                result = future.result()
                new_results.append(result)
                print(
                    f"[{n:{width}}/{len(tasks)}] {'✓' if result['valid_solution'] else '✗'} "
                    f"{name:<26} {path}/{pid}  "
                    f"cost={result['tour_cost']:.4f}  t={result['time_to_solve']:.2f}s"
                )
            except Exception as exc:
                failed += 1
                print(f"[{n:{width}}/{len(tasks)}] FAILED {name} {path}/{pid} — {exc}")

            if len(new_results) % args.save_every == 0 and new_results:
                save(new_results, existing)

    save(new_results, existing)

    if new_results:
        df = pd.DataFrame(new_results)
        print(
            "\n"
            + df.groupby(["algorithm", "problem_type", "problem_size"])["tour_cost"]
            .mean()
            .rename("mean_cost")
            .reset_index()
            .to_string(index=False)
        )

    if failed:
        print(f"\n{failed} task(s) failed.")


if __name__ == "__main__":
    main()
