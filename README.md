# NumCompute – Assignment 2.1 & 2.2
# ASSIGNMENT2.1
The Repository is for the Assignment 2.1 for our team.

# NumCompute

A modular, computing toolkit built with plain Python and NumPy — no external ML/DL libraries.

NumCompute reimplements the core data-science primitives you'd normally reach for in scikit-learn or pandas: I/O, preprocessing, sorting/search, ranking, descriptive statistics, evaluation metrics, finite-difference calculus, and pipeline composition. The emphasis is on **deep algorithmic understanding**, **numerical stability**.

---

## Installation

```bash
# From the project root
pip install -e .
```

Or without installing (add to path manually):

```python
import sys
sys.path.insert(0, "/path/to/NumCompute")
import numcompute as nc
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.23. No other dependencies.

---

## Quick Example

```python
import numpy as np
from numcompute import io, preprocessing, metrics, pipeline

# 1. Load CSV (handles missing values automatically)
data = io.load_csv("data.csv", skip_header=True)
X, y = data[:, :-1], data[:, -1].astype(int)

# 2. Build a preprocessing pipeline
pipe = pipeline.Pipeline([
    ("impute", preprocessing.SimpleImputer(strategy="mean")),
    ("scale",  preprocessing.StandardScaler()),
])
X_clean = pipe.fit_transform(X)

# 3. Evaluate predictions
y_pred = (X_clean[:, 0] > 0).astype(int)          # dummy predictor
print("Accuracy :", metrics.accuracy(y, y_pred))
print("F1       :", metrics.f1(y, y_pred, average="binary"))
```

---

## Module Reference

### `numcompute.io`
| Function | Description |
|----------|-------------|
| `load_csv(path, ...)` | Load CSV → `np.ndarray`; missing values → NaN |
| `load_csv_chunks(path, chunk_size, ...)` | Generator yielding row-chunks for out-of-memory datasets |
| `save_csv(array, path, ...)` | Write array to CSV |

### `numcompute.preprocessing`
All transformers implement `fit(X)`, `transform(X)`, `fit_transform(X)`.

| Class | Description |
|-------|-------------|
| `StandardScaler` | Zero mean, unit variance (z-score); handles zero-variance columns |
| `MinMaxScaler(feature_range)` | Scale to `[lo, hi]`; constant columns map to `lo` |
| `SimpleImputer(strategy)` | Replace NaN with `mean`, `median`, or `constant` |
| `OneHotEncoder` | Integer → binary indicator columns; infers categories from training data |

### `numcompute.sort_search`
| Function | Description |
|----------|-------------|
| `stable_sort(values, axis, descending)` | Mergesort-backed stable sort |
| `argsort_stable(values, ...)` | Returns sorting indices |
| `multi_key_sort(data, keys, descending)` | Lexicographic multi-column sort via `np.lexsort` |
| `topk(values, k, largest, return_indices)` | O(n + k log k) top-k via `np.argpartition` |
| `quickselect(values, k, largest)` | O(n) expected k-th order statistic (educational) |
| `binary_search(sorted_array, x)` | Returns `(insertion_index, found: bool)` |

### `numcompute.rank`
| Function | Description |
|----------|-------------|
| `rank(data, method, ascending)` | Rank with ties: `average`, `dense`, `ordinal`, `min`, `max` |
| `percentile(data, q, interpolation)` | Percentile with `linear`, `lower`, `higher`, `midpoint`, `nearest` interpolation |

### `numcompute.stats`
| Function / Class | Description |
|------------------|-------------|
| `describe(X, axis)` | Summary dict: n, mean, std, var, min, max, median, Q25, Q75, nan_count |
| `mean / std / median / quantiles` | NaN-safe axis-wise statistics |
| `histogram(X, bins, range, density)` | Histogram ignoring NaNs |
| `WelfordStats(n_features)` | Streaming mean/variance via Welford's algorithm (O(1) memory) |

### `numcompute.metrics`
**Classification:** `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`
(all support `average='macro'|'micro'|'binary'`)

**Regression:** `mse`, `mae`, `r2_score`

**Bonus:** `roc_curve(y_true, y_score)` → `(fpr, tpr, thresholds)` and `auc(fpr, tpr)`

### `numcompute.optim`
| Function | Description |
|----------|-------------|
| `grad(f, x, h, method)` | Numerical gradient ∇f(x); `central` (O(h²)), `forward`/`backward` (O(h)) |
| `jacobian(F, x, h, method)` | Jacobian matrix ∂F/∂x for vector-valued F |
| `backtracking_line_search(...)` | Armijo line search (optional) |

### `numcompute.pipeline`
| Class | Description |
|-------|-------------|
| `Pipeline(steps)` | Chain of `(name, transformer)` pairs; supports `fit`, `transform`, `fit_transform`, `predict` |
| `FeatureUnion(transformer_list)` | Apply transformers in parallel and `hstack` outputs |
| `Transformer` | Base class (fit + transform protocol) |
| `Estimator` | Base class (fit + predict protocol) |

### `numcompute.utils`
Distances (`euclidean_distance`, `manhattan_distance`, `cosine_similarity`, `pairwise_euclidean`),
activations (`sigmoid`, `softmax`, `relu`, `leaky_relu`, `tanh`),
`logsumexp`, and `batch_iter`.

### `numcompute.benchmarking`
`timeit`, `compare`, `speedup`, `print_benchmark_table`, `run_standard_benchmarks`.

---

## Performance Table

Benchmarked on Python 3.12 / NumPy 2.4 — array size **500,000 float64** elements.

| Operation | Loop (ms) | NumPy (ms) | Speedup |
|-----------|----------:|----------:|--------:|
| Sum       | 16.9      | 0.06      | **277.7×** |
| Mean      | 15.5      | 0.09      | **177.5×** |
| Std       | 56.7     | 0.47      | **120.8×** |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

85 unit tests covering: normal cases, edge cases (empty arrays, all-equal values,
ties, extreme k, NaNs, non-contiguous strides, shape mismatches).

---

## Running Benchmarks

```bash
python benchmark/run_benchmarks.py
```

---

## Project Structure

```
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py              # CSV reader, chunked loading
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, Imputer, OneHot
│   ├── sort_search.py     # Stable sort, multi-key, top-k, quickselect, binary search
│   ├── rank.py            # Ranking (5 tie methods), percentile (5 interpolations)
│   ├── stats.py           # Descriptive stats, Welford, histogram, quantiles
│   ├── metrics.py         # Accuracy/P/R/F1, MSE, confusion matrix, ROC/AUC
│   ├── optim.py           # Finite-diff gradients, Jacobians, line search
│   ├── pipeline.py        # Pipeline, FeatureUnion, Transformer/Estimator
│   ├── utils.py           # Distances, activations, logsumexp, batching
│   └── benchmarking.py    # Micro-benchmark harness
├── tests/
│   └── test_all.py        # 85 unit tests
├── demo/
│   ├── quickstart.ipynb   # End-to-end notebook demo
│   └── sample_data.csv    # Sample CSV with missing values
├── benchmark/
│   └── run_benchmarks.py  # Reproducible benchmark script
├── README.md
└── pyproject.toml
```

---

## Design Notes

**Vectorisation:** All core computations use NumPy broadcasting and ufuncs.
Python loops appear only in `quickselect`.

**Numerical stability:**
- `softmax` uses max-shifting to prevent overflow
- `logsumexp` uses the log-sum-exp trick
- `sigmoid` uses two-branch evaluation to avoid large-positive overflow
- `StandardScaler` and `MinMaxScaler` clamp zero-variance/zero-range denominators to 1
- All statistics use `np.nan*` variants for robust NaN handling

**API consistency:** All transformers share `fit / transform / fit_transform`; shapes are documented; `axis` semantics follow NumPy conventions.



# Assignment 2.2 – Streaming Machine Learning Framework

## Overview

Assignment 2.2 extends the NumCompute toolkit with streaming machine learning capabilities. The framework now supports incremental preprocessing, chunk-based model training, streaming evaluation metrics, logging, benchmarking, and visualisation.

The goal of this extension is to allow machine learning workflows to operate on data arriving in small batches rather than requiring the entire dataset to be loaded at once.

---

## Features Implemented

### Streaming Models

#### DecisionTreeClassifier

* Supports fit()
* Supports partial_fit()
* Gini and Entropy splitting criteria
* Configurable tree depth
* Feature subsampling support

#### EnsembleClassifier

* Multiple Decision Trees
* Bootstrap sampling
* Majority voting
* Incremental learning using partial_fit()

---

### Streaming Training

#### StreamTrainer

Provides:

* Chunk-based training
* Chunk accuracy tracking
* Cumulative accuracy tracking
* Memory usage monitoring
* Training logs

---

### Streaming Metrics

Implemented:

* StreamingAccuracy
* StreamingClassificationMetrics

Supports:

* Accuracy
* Precision
* Recall
* F1-score
* Rolling-window evaluation
* Confusion matrix tracking

---

### Streaming Statistics

Implemented:

* WelfordStats
* StreamingStats
* update_stats()

Supports:

* Running mean
* Running variance
* Standard deviation
* Quantiles
* Histograms
* Chunk-based updates

---

### Incremental Preprocessing

Extended preprocessing components:

* StandardScaler
* MinMaxScaler
* SimpleImputer
* OneHotEncoder

All support incremental updates through partial_fit().

---

### Pipeline Support

The Pipeline class now supports:

* fit()
* transform()
* predict()
* partial_fit()

allowing preprocessing and model training to operate together in a streaming environment.

---

### Visualisation

Implemented visualisation utilities:

* plot_metric_over_time()
* compare_models()
* plot_predictions_vs_ground_truth()

These functions assist with monitoring streaming performance and comparing models.

---

## Usage Instructions

### Install the Package

```bash
pip install -e .
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

Expected result:

```text
141 passed
```

### Run Benchmarks

```bash
python benchmark/benchmark.py
```

Benchmarks include:

* Python loops vs NumPy vectorisation
* Streaming Decision Tree vs Ensemble comparison

### Run the Streaming Demonstration

Open:

```text
demo/stream_demo.ipynb
```

The notebook demonstrates:

* CSV loading
* Chunk generation
* Incremental preprocessing
* Streaming model training
* StreamTrainer logging
* Visualisation of results

---

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── tree.py
│   ├── ensemble.py
│   ├── stream.py
│   ├── visualise.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── io.py
│   └── utils.py
│
├── demo/
│   ├── quickstart.ipynb
│   ├── stream_demo.ipynb
│   └── sample.csv
│
├── benchmark/
│   └── benchmark.py
│
├── tests/
│   ├── test_all.py
│   └── test_streaming_components.py
│
├── README.md
└── pyproject.toml
```

---

## Testing

The framework was validated using automated unit tests covering both Assignment 2.1 and Assignment 2.2 functionality.

Test coverage includes:

* Decision Tree
* Ensemble Classifier
* StreamTrainer
* Streaming Metrics
* Streaming Statistics
* Incremental Preprocessing
* Pipeline Integration
* Visualisation Utilities

Final Results:

```text
141 passed
0 failed
```

---

## Benchmark Results

Example benchmark results:

| Benchmark                  | Result                 |
| -------------------------- | ---------------------- |
| Sum of Squares             | 44.9× NumPy speedup    |
| Mean                       | 80.7× NumPy speedup    |
| Top-K                      | 43.2× NumPy speedup    |
| Streaming Tree vs Ensemble | Ensemble ≈ 3.2× slower |

---

## Conclusion

The Assignment 2.2 extension successfully adds streaming machine learning functionality to NumCompute while maintaining a consistent API and modular design. The framework supports incremental preprocessing, model training, evaluation, benchmarking, testing, and visualisation for streaming data applications.
