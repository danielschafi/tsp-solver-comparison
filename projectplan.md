# TSP Solver Comparison — Project Plan

## Overview

A systematic experimental comparison of classical and metaheuristic TSP solving methods on synthetic instances. The study covers five competing algorithms spanning two paradigms (trajectory and population-based), benchmarked against a state-of-the-art reference solver. All implementations are in Python with Numba-accelerated inner loops.

---

## 1. Algorithms

### 1.1 Competing algorithms

| Algorithm | Paradigm | Escape mechanism | Reference |
|---|---|---|---|
| 2-opt | Trajectory | None (baseline) | [Lin (1965) — "Computer solutions of the traveling salesman problem"](https://doi.org/10.1002/j.1538-7305.1965.tb04146.x) |
| Simulated Annealing | Trajectory | Probabilistic acceptance | [Kirkpatrick et al. (1983) — "Optimization by Simulated Annealing"](https://doi.org/10.1126/science.220.4598.671) |
| Tabu Search | Trajectory | Memory-based forbidden moves | [Glover (1989) — "Tabu Search — Part I"](https://doi.org/10.1287/ijoc.1.3.190) |
| Iterated Local Search | Trajectory | Perturbation + re-optimisation | [Lourenço et al. (2003) — "Iterated Local Search"](https://doi.org/10.1007/0-306-48056-5_11) |
| Genetic Algorithm | Population | Selection, crossover, mutation | [Goldberg & Lingle (1985) — "Alleles, loci and the TSP"](https://dl.acm.org/doi/10.5555/645511.657055) — use Order Crossover (OX): [Davis (1985)](https://dl.acm.org/doi/10.5555/645511.657041) |

### 1.2 Reference ceiling

| Solver | Role | Reference |
|---|---|---|
| LKH-3 | Near-optimal quality reference. All solution quality results are reported as % gap to LKH. LKH is not included in runtime comparisons. | [Helsgott (2017) — LKH-3](http://webhotel4.ruc.dk/~keld/research/LKH-3/) |
| Concorde | Exact optimality reference for small instances (n ≤ 200) used in the validation phase only. | [Applegate et al. — Concorde TSP Solver](https://www.math.uwaterloo.ca/tsp/concorde.html) |

**Framing note:** LKH-3 is infrastructure, not a competitor. It is called via subprocess from Python using its official C implementation — reimplementing it would misrepresent its performance. A suggested methods-section sentence: *"LKH-3 is used to establish near-optimal reference solutions for each instance. All solution quality results are reported as percentage gap to LKH. The comparison study proper is conducted among the five implemented methods under equal time budgets."*

---

## 2. Instance design

### 2.1 Instance sizes

Instance sizes are determined empirically via a **pilot experiment** (see Section 5) rather than fixed in advance. The target ceiling is n ≤ 500, subject to all algorithms reaching convergence within a reasonable time budget. Candidate sizes: **n = 100, 200, 300, 500**.

### 2.2 Synthetic data generation

Two distribution types are used, covering qualitatively different problem structures:

**Uniform random.** Cities are sampled independently from a uniform distribution over a unit square `[0, 1]²`. This is the standard baseline in the TSP literature and produces instances with no exploitable geographic structure.

```python
import numpy as np

def generate_uniform(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n, 2))
```

**Clustered.** Cities are grouped into k clusters, each with a Gaussian spread. This tests whether algorithms exploit geographic locality differently — 2-opt and Or-opt tend to benefit more from cluster structure than population methods do.

```python
def generate_clustered(n, k, spread, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(k, 2))
    cities = []
    for i in range(n):
        c = centers[i % k]
        cities.append(rng.normal(c, spread))
    return np.clip(cities, 0, 1)
```

Recommended parameters: k = n // 10 clusters, spread = 0.05. This produces visually distinct clusters without being trivially easy.

**Distance metric:** 2D Euclidean throughout. The distance matrix is precomputed once per instance as a NumPy `float32` array and shared read-only across all algorithms.

### 2.3 Replications

At least **30 independent seeds** per (algorithm × instance type × instance size) cell. Reported statistics: mean gap, standard deviation, best-of-30. This supports non-parametric statistical tests.

### 2.4 Fixed instances

All algorithms are evaluated on the **exact same set of instances** (same seeds). Generating instances separately per algorithm would introduce uncontrolled variance.

---

## 3. Fairness protocol

### 3.1 Stopping criteria

Each algorithm run is subject to a **fixed wall-clock time budget** determined during the pilot experiment (see Section 5). The budget is set so that all algorithms reach convergence on the largest instance size within that budget.

Additionally, **convergence curves** (best solution quality vs. elapsed time) are recorded for every run. This allows post-hoc quality-at-convergence comparisons independent of the time budget.

### 3.2 Initialisation

All algorithms start from the **same nearest-neighbour greedy tour** for a given instance and seed. Stochastic algorithms use the same seed for their internal random number generators.

### 3.3 Hyperparameter tuning

Each algorithm is tuned independently on a **held-out validation set** of instances not used in the main experiment. Reporting the tuned values in the paper is mandatory for reproducibility.

Suggested starting points:

| Algorithm | Key parameters |
|---|---|
| SA | Initial temperature (use Ben-Ameur 2004 auto-calibration), cooling rate α ∈ [0.995, 0.9999] |
| Tabu | Tabu tenure ∈ [7, 20], aspiration criterion: accept if better than best known |
| ILS | Perturbation strength (double-bridge move is standard for TSP), local search = 2-opt |
| GA | Population size ∈ [50, 200], crossover rate ≈ 0.85, mutation rate ≈ 0.02, tournament selection |

### 3.4 Implementation parity

- All algorithms implemented in Python, same codebase.
- Shared Numba-JIT kernels for `tour_cost()`, `two_opt_improve()`, and distance lookup — applied equally to all algorithms.
- No algorithm uses multi-threading or multi-core within a single run. All search is single-threaded.
- LKH-3 called via subprocess; its runtime is never compared against Python algorithm runtimes.

---

## 4. Metrics

| Metric | Description |
|---|---|
| **% gap to LKH** | Primary quality metric: `(cost − lkh_cost) / lkh_cost × 100` |
| **% gap to optimal** | For validation instances (n ≤ 200) only, using Concorde |
| **Wall-clock runtime** | Seconds to convergence, single-threaded, same hardware |
| **Standard deviation** | Across 30 seeds — measures robustness |
| **Best-of-30** | Best solution found across all seeds — upper bound on algorithm capability |
| **Convergence curve** | Best cost vs. time — reported as a plot, not a single number |

**Statistical testing:** Use the Wilcoxon signed-rank test for pairwise comparisons, or Friedman test for omnibus ranking across all algorithms. Do not rely on means alone given the non-Gaussian distribution of TSP solution quality across seeds.

---

## 5. Pilot experiment

Before the main experiment, run a pilot to determine:

1. **Feasible instance ceiling.** Run the slowest algorithm (expected: GA or Tabu) on 5 instances at each candidate size (n = 100, 200, 300, 500) with a generous budget. Record time to convergence.
2. **Time budget.** Set the main experiment budget at 2–3× the convergence time of the slowest algorithm on the largest feasible size.
3. **Concorde feasibility.** Run Concorde on 5 instances at each size ≤ 300. If it solves them in under a few minutes each, use exact gaps for that size tier. If runtime is unpredictable, fall back to LKH gaps and note it.
4. **Implementation sanity check.** On small instances (n ≤ 100), verify each algorithm finds solutions within a few percent of Concorde's exact optimum.

---

## 6. Implementation architecture

```
experiment_runner.py          # multiprocessing.Pool — one job per (algo × seed × instance)
│
├── algorithms/
│   ├── two_opt.py            # single-threaded
│   ├── simulated_annealing.py
│   ├── tabu_search.py
│   ├── ils.py
│   └── genetic_algorithm.py
│
├── kernels.py                # @njit shared Numba functions: tour_cost, two_opt_move, etc.
├── instances.py              # generate_uniform, generate_clustered, precompute_dist_matrix
├── lkh_runner.py             # subprocess wrapper for LKH-3 binary
├── concorde_runner.py        # subprocess wrapper for Concorde (validation only)
└── results_logger.py         # logs cost, runtime, seed, gap-to-LKH per run → CSV
```

**Parallelism strategy:**
- *Across runs:* `multiprocessing.Pool` or `joblib` — embarrassingly parallel, no fairness concerns.
- *Within a run:* single-threaded for all algorithms, ensuring wall-clock time comparisons are valid.
- *Inner loops:* Numba `@njit` applied to shared kernels only — identical speedup available to all algorithms.

---

## 7. Reporting

Results are organised around three questions:

1. **Which algorithm finds the best solutions?** — gap-to-LKH table across all sizes and distributions.
2. **Which algorithm converges fastest?** — convergence curves (best cost vs. time).
3. **How consistent is each algorithm?** — standard deviation across seeds; are some methods high-variance?

A secondary analysis should examine whether the **ranking of algorithms changes between uniform and clustered instances**, as this is often the most informative finding in a comparison study.

---

## 8. References

- Applegate, D., Bixby, R., Chvátal, V., Cook, W. — *Concorde TSP Solver*. https://www.math.uwaterloo.ca/tsp/concorde.html
- Ben-Ameur, W. (2004). Computing the Initial Temperature of Simulated Annealing. *Computational Optimization and Applications*, 29(3), 369–383. https://doi.org/10.1023/B:COAP.0000044187.23143.bd
- Davis, L. (1985). Applying Adaptive Algorithms to Epistatic Domains. *IJCAI*.
- Glover, F. (1989). Tabu Search — Part I. *ORSA Journal on Computing*, 1(3), 190–206. https://doi.org/10.1287/ijoc.1.3.190
- Goldberg, D. E., & Lingle, R. (1985). Alleles, loci and the TSP. *ICGA*.
- Helsgott, K. (2017). *LKH-3*. http://webhotel4.ruc.dk/~keld/research/LKH-3/
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680. https://doi.org/10.1126/science.220.4598.671
- Lin, S. (1965). Computer solutions of the traveling salesman problem. *Bell System Technical Journal*, 44(10), 2245–2269. https://doi.org/10.1002/j.1538-7305.1965.tb04146.x
- Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). Iterated Local Search. In *Handbook of Metaheuristics*. https://doi.org/10.1007/0-306-48056-5_11
