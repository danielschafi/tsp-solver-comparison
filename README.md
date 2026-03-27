# TSP Algorithms

## Introduction

The traveling salesman problem (TSP) is a classic combinatorial optimization problem that asks for the shortest possible route that visits a set of cities and returns to the origin city. This repository contains implementations of various algorithms to solve the TSP. The goal is to compare the algorithms based on their scalability and solution quality. The algorithms implemented include:

- Simulated Annealing
- Taboo Search
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

3. Run:

   ```bash
   uv run main.py
   ```

### Main Dependencies

```plain
numpy
matplotlib
pandas
```

## Running the Benchmark

1. Generate tsp problems
```bash
uv run src/data_gen/generate_tsp_instances.py
```
This will generate uniform and clustered samples of sizes 10-500.

2. Run the benchmark
```bash
echo to be done 
```


## Running one algorithm
```bash
uv run -m src.algorithms.greedy --path <path to data dir> --problem_id <id of the problem>
```
- path: E.g to data/uniform/100
- problem_id (optional): If provided, runs the algorithm only for that instance, otherwise on all instances in the folder
