import time
import numpy as np
def timeit(func, *args, n_runs=5, warmup=1, **kwargs):
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
    }
def compare(functions, args=(), kwargs=None, n_runs=5, warmup=1):
    if kwargs is None:
        kwargs = {}

    results = {}
    for name, func in functions.items():
        results[name] = timeit(func, *args, n_runs=n_runs, warmup=warmup, **kwargs)

    return results


def speedup(results, baseline):
    if baseline not in results:
        raise KeyError(f"Baseline '{baseline}' not found")

    base_time = results[baseline]["mean_s"]

    return {
        name: base_time / value["mean_s"]
        for name, value in results.items()
        if name != baseline
    }