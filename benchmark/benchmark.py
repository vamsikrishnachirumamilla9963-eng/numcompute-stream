import time
import platform
import numpy as np


# ---------------------------
# Environment Info
# ---------------------------
def environment_info():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


# ---------------------------
# Timing function
# ---------------------------
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
    }


# ---------------------------
# Implementations
# ---------------------------
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


# ---------------------------
# Print helper
# ---------------------------
def print_benchmark(name, loop_res, np_res):
    print(f"\n==== Benchmark: {name} ====")
    print(f"Python loop     : {loop_res['mean']:.6f}s")
    print(f"NumPy vectorised: {np_res['mean']:.6f}s")
    print(f"Speedup         : {loop_res['mean']/np_res['mean']:.1f}x")


# ---------------------------
# Main
# ---------------------------
def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1_000_000)

    print("Environment:")
    for k, v in environment_info().items():
        print(f"  {k}: {v}")

    # Sum of squares
    loop_res = timeit(loop_sum_squares, x)
    np_res = timeit(numpy_sum_squares, x)
    print_benchmark("Sum of Squares", loop_res, np_res)

    # Mean
    loop_res = timeit(loop_mean, x)
    np_res = timeit(numpy_mean, x)
    print_benchmark("Mean", loop_res, np_res)

    # Top-K
    k = 5
    loop_res = timeit(loop_top_k, x, k)
    np_res = timeit(numpy_top_k, x, k)
    print_benchmark("Top-K", loop_res, np_res)


if __name__ == "__main__":
    main()