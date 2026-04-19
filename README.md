# TSP Algorithms

## Introduction

The traveling salesman problem (TSP) is a classic combinatorial optimization problem that asks for the shortest possible route that visits a set of cities and returns to the origin city. This repository contains implementations of various algorithms to solve the TSP. The goal is to compare the algorithms based on their scalability and solution quality. The algorithms implemented include:

- Simulated Annealing
- Tabu Search
- Genetic Algorithm
- 2-opt
- Iterated Local Search

And are compared against the optimal solution obtained using the Concorde TSP/LHK3 Solver.

## Traveling Salesman Problem

**Given:** A complete graph $G = (V, E)$ with $n$ cities and distance matrix $d_{ij}$

**Minimize:**

$$
\min \sum_{i=1}^{n} d_{i,\sigma(i+1)}
$$

where $\sigma$ is a permutation of cities.

**Subject to:**

- Each city is visited **exactly once**
- The route forms a **single closed tour**
- $\sigma(n+1) = \sigma(1)$ (return to start)

## Project Structure

```
tsp-solver-comparison/
├── bin/                          # Concorde binary
│   └── concorde
├── data/                         # Generated TSP instances
│   ├── uniform/                  # Uniformly distributed cities
│   │   ├── 10/ 25/ 50/ 100/ 200/ 500/
│   │   └── {id}_nodes.npy, {id}_edges.npy
│   └── clustered/                # Clustered city distributions
│       ├── 10/ 25/ 50/ 100/ 200/ 500/
│       └── {id}_nodes.npy, {id}_edges.npy
├── results/
│   ├── results.json              # Raw benchmark results (per-instance)
│   ├── results.csv               # Same data in CSV format
│   └── plots/                    # Saved plots (when using --save)
├── src/
│   ├── algorithms/
│   │   ├── base.py               # Base class for all solvers
│   │   ├── greedy.py             # Nearest-neighbor greedy heuristic
│   │   ├── two_opt.py            # 2-opt local search
│   │   ├── simulated_annealing.py
│   │   ├── tabu_search.py
│   │   ├── genetic.py            # Genetic algorithm
│   │   ├── iterative_local_search.py
│   │   └── concorde_solver.py    # Wrapper for Concorde exact solver
│   ├── data_gen/
│   │   └── generate_tsp_instances.py
│   ├── utils/
│   │   ├── compute_statistics.py # Aggregated stats (mean cost, gap%, time)
│   │   ├── create_aggregate_plots.py
│   │   ├── plot_one_problem.py   # Visualize a single tour
│   │   └── shared_util_funcs.py
│   └── benchmark.py              # Parallel benchmark runner
├── pyproject.toml
└── README.md
```

## Installation

### Using `uv` (Python package manager)

1. Install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

### Main Dependencies

```plain
numpy
matplotlib
pandas
```

### Concorde Setup

Create a folder, download the pre-compiled binary, and make it executable:

```bash
mkdir -p bin/ && cd bin/
wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/linux24/concorde.gz
gunzip concorde.gz
chmod +x concorde
```

Verify it works:

```bash
./concorde -h
./concorde -s 99 -k 100
```

## Running the Benchmark

### 1. Generate TSP instances

```bash
uv run -m src.data_gen.generate_tsp_instances
```

This generates 30 uniform and 30 clustered instances for each size (10, 25, 50, 100, 200, 500). Options:

| Flag             | Description                            | Default                    |
| ---------------- | -------------------------------------- | -------------------------- |
| `--sizes`        | Problem sizes to generate              | `10 25 50 100 200 500`     |
| `--repetitions`  | Instances per size/type                | `30`                       |
| `--clean`        | Delete existing data before generating | off                        |
| `--testdata`     | Write to `data/test/` instead          | off                        |

### 2. Run the benchmark

```bash
uv run -m src.benchmark
```

This runs all algorithms on all generated instances in parallel and writes results to `results/results.json` and `results/results.csv`. Already-computed results are skipped automatically.

| Flag               | Description                                    | Default       |
| ------------------ | ---------------------------------------------- | ------------- |
| `--type`           | Instance type: `uniform`, `clustered`, or `all`| `all`         |
| `--sizes`          | Problem sizes to benchmark                     | `10 25 50 100 200 500` |
| `--algorithms`     | Specific algorithms to run (space-separated)   | all           |
| `--skip-concorde`  | Skip the Concorde solver                       | off           |
| `--workers`        | Number of parallel workers                     | CPU count     |
| `--save-every`     | Flush results to disk every N completions       | `50`          |
| `--clean`          | Discard all existing results and re-run         | off           |

Example: benchmark only simulated annealing and tabu search on uniform instances of size 100:

```bash
uv run -m src.benchmark --type uniform --sizes 100 --algorithms simulated_annealing tabu_search
```

## Statistics

Compute aggregated statistics (mean cost, standard deviation, solve time, optimality gap) from the benchmark results:

```bash
uv run -m src.utils.compute_statistics
```

Optionally save to CSV:

```bash
uv run -m src.utils.compute_statistics --csv results/statistics.csv
```

The optimality gap is computed as the percentage above the best-known solution (typically the Concorde optimal) for each instance.

## Plots

### Aggregate comparison plots

Generate four comparison figures from the benchmark results:

1. **Time scaling** -- mean solve time vs. problem size (log-log)
2. **Solution quality** -- mean tour cost vs. problem size
3. **Optimality gap bars** -- mean % gap above best-known per algorithm/size
4. **Gap heatmap** -- algorithm x size grid colored by mean gap %

```bash
uv run -m src.utils.create_aggregate_plots          # display interactively
uv run -m src.utils.create_aggregate_plots --save    # save to results/plots/
uv run -m src.utils.create_aggregate_plots --save --out my_plots/
```

### Plot a single solution

Visualize the tour found by a specific algorithm on a specific problem instance:

```bash
uv run -m src.utils.plot_one_problem --problem <path> --algorithm <algorithm>
```

- `--problem`: path to an instance, e.g. `data/uniform/100/0` or the full path to a `_nodes.npy`/`_edges.npy` file
- `--algorithm`: algorithm name as stored in results (e.g. `Greedy`, `TwoOpt`, `SimulatedAnnealing`, `TabuSearch`, `Genetic`, `IterativeLocalSearch`, `Concorde`)
- `--save`: save the plot to disk instead of displaying

## Running a Single Algorithm

Run one algorithm on a specific data directory:

```bash
uv run -m src.algorithms.greedy --path data/uniform/100 --problem_id 0
```

- `--path`: directory containing the instance files (e.g. `data/uniform/100`)
- `--problem_id` (optional): run only on that instance; omit to run on all instances in the directory

## References

- [Handbook of Metaheuristics - Large Neighborhood Search](https://link.springer.com/chapter/10.1007/978-3-319-91086-4_4) Very good handbook with many local search methods.
- [Local Search and Metaheuristics for the Traveling Salesman Problem](https://leeds-faculty.colorado.edu/glover/Publications/TSP.pdf): Overview over some solution methods, not that mathematical
