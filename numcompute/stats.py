"""
stats.py — Descriptive statistics for NumCompute.

Provides batch (vectorised NumPy) and streaming (Welford) statistics,
histogram computation, and quantiles with NaN handling.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union


def describe(
    X: np.ndarray,
    axis: Optional[int] = None,
    ddof: int = 0,
) -> dict:
    """Compute a summary dict of descriptive statistics.

    Parameters
    ----------
    X    : np.ndarray
    axis : int or None, default None
    ddof : int, default 0 (0=population, 1=sample)

    Returns
    -------
    dict with keys: n, mean, std, var, min, max, median, q25, q75, nan_count

    """
    X = np.asarray(X, dtype=float)
    kw = dict(axis=axis)
    return {
        "n"        : int(np.sum(~np.isnan(X))),
        "mean"     : np.nanmean(X, **kw),
        "std"      : np.nanstd(X, ddof=ddof, **kw),
        "var"      : np.nanvar(X, ddof=ddof, **kw),
        "min"      : np.nanmin(X, **kw),
        "max"      : np.nanmax(X, **kw),
        "median"   : np.nanmedian(X, **kw),
        "q25"      : np.nanpercentile(X, 25, **kw),
        "q75"      : np.nanpercentile(X, 75, **kw),
        "nan_count": int(np.sum(np.isnan(X))),
    }


def mean(X: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """NaN-safe mean."""
    return np.nanmean(np.asarray(X, dtype=float), axis=axis)


def std(X: np.ndarray, axis: Optional[int] = None, ddof: int = 0) -> Union[float, np.ndarray]:
    """NaN-safe standard deviation."""
    return np.nanstd(np.asarray(X, dtype=float), axis=axis, ddof=ddof)


def median(X: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """NaN-safe median."""
    return np.nanmedian(np.asarray(X, dtype=float), axis=axis)

def quantiles(
    X: np.ndarray,
    q: Union[float, list, np.ndarray],
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Compute quantile(s) with NaN handling.

    Parameters
    ----------
    X    : np.ndarray
    q    : float or array-like — quantile(s) in [0, 1]
    axis : int or None

    Raises
    ------
    ValueError — if any q value is outside [0, 1]
    """
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("All q values must be in [0, 1].")
    return np.nanpercentile(np.asarray(X, dtype=float), q_arr * 100, axis=axis)


def histogram(
    X: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram, ignoring NaN values.

    Parameters
    ----------
    X       : np.ndarray — flattened before binning
    bins    : int or np.ndarray, default 10
    range   : (float, float) or None
    density : bool, default False

    Returns
    -------
    counts    : np.ndarray, shape (n_bins,)
    bin_edges : np.ndarray, shape (n_bins + 1,)

    """
    X     = np.asarray(X, dtype=float).ravel()
    clean = X[~np.isnan(X)]
    return np.histogram(clean, bins=bins, range=range, density=density)

class WelfordStats:
    """Incremental mean and variance via Chan's parallel Welford algorithm.

    Parameters
    ----------
    n_features : int, default 1

    Attributes
    ----------
    n_    : int
    mean_ : np.ndarray, shape (n_features,)
    M2_   : np.ndarray, shape (n_features,)
    """

    def __init__(self, n_features: int = 1) -> None:
        self.n_features = n_features
        self.n_:    int        = 0
        self.mean_: np.ndarray = np.zeros(n_features, dtype=float)
        self.M2_:   np.ndarray = np.zeros(n_features, dtype=float)

    def update(self, x: np.ndarray) -> "WelfordStats":
        """Incorporate a batch via Chan's parallel formula — no Python row loop.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (batch, n_features)

        Returns
        -------
        self

        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        b = x.shape[0]
        if b == 0:
            return self
        b_mean = x.mean(axis=0)
        b_M2   = np.sum((x - b_mean) ** 2, axis=0)
        n_a    = self.n_
        n_t    = n_a + b
        delta  = b_mean - self.mean_
        self.mean_ = self.mean_ + delta * (b / n_t)
        self.M2_   = self.M2_ + b_M2 + delta ** 2 * (n_a * b / n_t)
        self.n_    = n_t
        return self

    @property
    def variance(self) -> np.ndarray:
        """Population variance (ddof=0)."""
        return self.M2_ / self.n_ if self.n_ >= 1 else np.full(self.n_features, np.nan)

    @property
    def sample_variance(self) -> np.ndarray:
        """Sample variance (ddof=1)."""
        return self.M2_ / (self.n_ - 1) if self.n_ >= 2 else np.full(self.n_features, np.nan)

    @property
    def std(self) -> np.ndarray:
        """Population standard deviation."""
        return np.sqrt(self.variance)

    def reset(self) -> "WelfordStats":
        """Reset all running statistics."""
        self.n_    = 0
        self.mean_ = np.zeros(self.n_features, dtype=float)
        self.M2_   = np.zeros(self.n_features, dtype=float)
        return self
    


# Streaming Statistics


class StreamingStats:
    """Chunk-based streaming statistics for NumCompute."""

    def __init__(self, n_features: int | None = None, bins: int = 10):
        self.n_features = n_features
        self.bins = bins
        self.reset()

    def update_stats(self, X_chunk: np.ndarray) -> "StreamingStats":
        """Update running statistics from one data chunk."""
        X = np.asarray(X_chunk, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"X_chunk must be 2-D, got shape {X.shape}")

        if self.n_features is None:
            self.n_features = X.shape[1]
            self.welford_ = WelfordStats(self.n_features)
        elif X.shape[1] != self.n_features:
            raise ValueError("X_chunk feature count does not match previous chunks.")

        clean = np.where(np.isnan(X), np.nan, X)

        self.welford_.update(np.nan_to_num(clean, nan=0.0))
        self.values_.append(X.copy())
        self.n_chunks_ += 1

        return self

    def result(self):
        """Return current streaming statistics."""
        if self.n_chunks_ == 0:
            return {
                "mean": np.array([]),
                "variance": np.array([]),
                "std": np.array([]),
                "quantiles": np.array([]),
                "histogram": (np.array([]), np.array([])),
                "n_chunks": 0,
            }

        all_values = np.vstack(self.values_)

        return {
            "mean": np.nanmean(all_values, axis=0),
            "variance": np.nanvar(all_values, axis=0),
            "std": np.nanstd(all_values, axis=0),
            "quantiles": np.nanpercentile(all_values, [25, 50, 75], axis=0),
            "histogram": np.histogram(all_values[~np.isnan(all_values)], bins=self.bins),
            "n_chunks": self.n_chunks_,
        }

    def reset(self):
        self.values_ = []
        self.n_chunks_ = 0
        self.welford_ = WelfordStats(self.n_features or 1)
        return self


def update_stats(X_chunk: np.ndarray, state: StreamingStats | None = None) -> StreamingStats:
    """Functional helper to update streaming statistics state."""
    if state is None:
        state = StreamingStats()
    return state.update_stats(X_chunk)