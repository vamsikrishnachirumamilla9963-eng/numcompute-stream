"""
NumCompute — a modular, production-grade scientific computing toolkit.

Modules
-------
io             CSV reader with streaming/chunking and dtype handling
preprocessing  StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
sort_search    Stable sort, multi-key sort, top-k, quickselect, binary search
rank           Ranking with tie handling; percentile computation
stats          Descriptive statistics, streaming Welford, histogram, quantiles
metrics        Accuracy/Precision/Recall/F1, MSE, confusion matrix, ROC/AUC
optim          Finite-difference gradients and Jacobians
pipeline       Transformer/Estimator protocol, Pipeline, FeatureUnion
utils          Distances, activations, logsumexp, batching helpers
benchmarking   Micro-benchmark harness and loop vs. vectorised comparisons
"""

from . import io, preprocessing, sort_search, rank, stats, metrics, optim, utils, pipeline, utils, benchmarking
__version__ = "0.1.0"
__all__ = [
    "io","preprocessing","sort_search","rank","stats",
    "metrics","optim","pipeline","utils","benchmarking",
]