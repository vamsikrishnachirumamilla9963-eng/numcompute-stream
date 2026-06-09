"""
benchmark.py — Reproducible benchmarks for NumCompute.

Includes:
- Environment information
- Python loop vs NumPy vectorised comparisons
- Streaming Decision Tree vs Ensemble training benchmark
"""

import time
import platform
import os
import sys
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import EnsembleClassifier
from numcompute.utils import batch_iter


def environment_info():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def timeit(func, *args, n_runs=5, warmup=1):
    for _ in range(warmup):
        func(*args)

    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def loop_sum_squares(x):
    total = 0.0

    for val in x:
        total += val * val

    return total


def numpy_sum_squares(x):
    return np.sum(x * x)


def loop_mean(x):
    total = 0.0

    for val in x:
        total += val

    return total / len(x)


def numpy_mean(x):
    return np.mean(x)


def loop_top_k(x, k):
    return sorted(x, reverse=True)[:k]


def numpy_top_k(x, k):
    return np.sort(x)[-k:][::-1]


def print_benchmark(name, loop_res, np_res):
    print(f"\n==== Benchmark: {name} ====")
    print(f"Python loop      : {loop_res['mean']:.6f}s")
    print(f"NumPy vectorised : {np_res['mean']:.6f}s")

    if np_res["mean"] > 0:
        print(f"Speedup          : {loop_res['mean'] / np_res['mean']:.1f}x")


def benchmark_streaming_models():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(1000, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    chunks = list(batch_iter(X, batch_size=100, y=y))

    def train_tree():
        tree = DecisionTreeClassifier(max_depth=3)

        for X_chunk, y_chunk in chunks:
            tree.partial_fit(X_chunk, y_chunk)

        return tree

    def train_ensemble():
        ensemble = EnsembleClassifier(
            n_estimators=5,
            max_depth=3,
            random_state=42,
        )

        for X_chunk, y_chunk in chunks:
            ensemble.partial_fit(X_chunk, y_chunk)

        return ensemble

    tree_res = timeit(train_tree)
    ensemble_res = timeit(train_ensemble)

    print("\n==== Benchmark: Streaming Tree vs Ensemble ====")
    print(f"Decision Tree : {tree_res['mean']:.6f}s")
    print(f"Ensemble      : {ensemble_res['mean']:.6f}s")

    if tree_res["mean"] > 0:
        print(f"Ratio         : {ensemble_res['mean'] / tree_res['mean']:.1f}x ensemble/tree")


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1_000_000)

    print("Environment:")
    for key, value in environment_info().items():
        print(f"  {key}: {value}")

    loop_res = timeit(loop_sum_squares, x)
    np_res = timeit(numpy_sum_squares, x)
    print_benchmark("Sum of Squares", loop_res, np_res)

    loop_res = timeit(loop_mean, x)
    np_res = timeit(numpy_mean, x)
    print_benchmark("Mean", loop_res, np_res)

    k = 5
    loop_res = timeit(loop_top_k, x, k)
    np_res = timeit(numpy_top_k, x, k)
    print_benchmark("Top-K", loop_res, np_res)

    benchmark_streaming_models()


if __name__ == "__main__":
    main()