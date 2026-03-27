import argparse
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")

np.random.seed(42)
import shutil


def gen_uniform(
    n: int, repetitions: int = 30, save_dir_base: Path = Path("data"), clean=False
):
    """
    Generates instances of uniformly sampled tsp problems of size n
    And saves them as npy files
    """
    print(f"Generating samples uniform with n: {n} for {repetitions} repetitions")
    if clean:
        shutil.rmtree(save_dir_base)
    save_dir = save_dir_base / "uniform" / f"{n}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for k in range(repetitions):
        node_coords = np.random.uniform(0, 1, size=(n, 2))
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                adj[i, j] = np.linalg.norm(node_coords[i] - node_coords[j])

        np.save(save_dir / f"{k}_nodes.npy", node_coords)
        np.save(save_dir / f"{k}_edges.npy", adj)


def gen_clustered(
    n: int, repetitions: int = 30, save_dir_base: Path = Path("data"), clean=False
):
    """
    Generates instances of uniformly sampled tsp problems of size n
    And saves them as npy files
    """
    print(f"Generating samples clustered with n: {n} for {repetitions} repetitions")
    if clean:
        shutil.rmtree(save_dir_base)
    save_dir = save_dir_base / "clustered" / f"{n}"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = n // 10 if n > 10 else 1
    spread = 0.05
    for k in range(repetitions):
        centers = np.random.uniform(0.1, 0.9, size=(n_clusters, 2))
        cities = []
        for i in range(n):
            c = centers[i % n_clusters]
            cities.append(np.random.normal(c, spread))
        node_coords = np.clip(cities, 0, 1)

        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                adj[i, j] = np.linalg.norm(node_coords[i] - node_coords[j])

        # shows one sample
        # plt.scatter(node_coords[:, 0], node_coords[:, 1])
        # plt.show()

        np.save(save_dir / f"{k}_nodes.npy", node_coords)
        np.save(save_dir / f"{k}_edges.npy", adj)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Generates random instances for the tsp problem."
    )
    arg_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200, 500],
        help="List of sample sizes to generate.",
    )
    arg_parser.add_argument(
        "--repetitions",
        type=int,
        default=30,
        help="Number of repetitions for each sample size.",
    )
    arg_parser.add_argument(
        "--clean",
        action="store_true",
        help="Whether to clean the output directory before building the dataset.",
    )
    arg_parser.add_argument(
        "--testdata",
        action="store_true",
        default=False,
        help="Pass this if you want to generate some additional data for e.g. hyperparameter tuning",
    )
    args = arg_parser.parse_args()

    if args.testdata:
        data_base_dir = DATA_DIR / "test"
    else:
        data_base_dir = DATA_DIR

    for size in args.sizes:
        gen_uniform(size, args.repetitions, data_base_dir, args.clean)
        gen_clustered(size, args.repetitions, data_base_dir, args.clean)


if __name__ == "__main__":
    main()
