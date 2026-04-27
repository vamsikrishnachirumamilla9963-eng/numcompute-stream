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

