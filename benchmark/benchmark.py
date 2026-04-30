import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from numcompute.benchmarking import (
    environment_info,
    loop_sum_squares,
    time_function,
    vectorized_sum_squares,
)


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1_000_000)

    print("Environment:")
    for k, v in environment_info().items():
        print(f"  {k}: {v}")

    loop = time_function(loop_sum_squares, x, repeat=3)
    vec = time_function(vectorized_sum_squares, x, repeat=3)

    print("\nBenchmark: sum of squares")
    print("Implementation      mean seconds")
    print(f"Python loop         {loop['mean']:.6f}")
    print(f"NumPy vectorised    {vec['mean']:.6f}")
    print(f"Speedup             {loop['mean'] / vec['mean']:.1f}x")


if __name__ == "__main__":
    main()
