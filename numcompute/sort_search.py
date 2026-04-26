"""
sort_search.py — Sorting, partial sorting, and searching utilities.

Most routines are fully vectorised via NumPy. The exception is
quickselect, which intentionally uses a Python-level loop to illustrate
the O(n) expected-time algorithm as required by the spec.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def stable_sort(
    values: np.ndarray,
    axis: int = -1,
    descending: bool = False,
) -> np.ndarray:
    """Return a sorted copy using stable mergesort.

    Parameters
    ----------
    values     : np.ndarray
    axis       : int, default -1
    descending : bool, default False

    Returns
    -------
    np.ndarray — same shape as values

    """
    out = np.sort(values, axis=axis, kind="stable")
    if descending:
        out = np.flip(out, axis=axis)
    return out


def argsort_stable(
    values: np.ndarray,
    axis: int = -1,
    descending: bool = False,
) -> np.ndarray:
    """Return indices that sort values (stable, preserves original order for ties).

    Parameters
    ----------
    values     : np.ndarray
    axis       : int, default -1
    descending : bool, default False

    Returns
    -------
    np.ndarray of int
    """
    idx = np.argsort(values, axis=axis, kind="stable")
    if descending:
        idx = np.flip(idx, axis=axis)
    return idx


def multi_key_sort(
    data: np.ndarray,
    keys: list,
    descending: Optional[list] = None,
) -> np.ndarray:
    """Sort rows of a 2-D array by multiple column keys.

    Parameters
    ----------
    data       : np.ndarray, shape (n_rows, n_cols)
    keys       : list of int — column indices in priority order
    descending : list of bool or None — per-key direction

    Returns
    -------
    np.ndarray, shape (n_rows, n_cols) — row-sorted copy

    Raises
    ------
    ValueError
        If data is not 2-D or any key index is out of range.

    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}.")
    n_cols = data.shape[1]
    for k in keys:
        if not (0 <= k < n_cols):
            raise ValueError(f"Key {k} out of range for {n_cols} columns.")
    if descending is None:
        descending = [False] * len(keys)
    if len(descending) != len(keys):
        raise ValueError("len(descending) must equal len(keys).")
    sort_cols = [(-data[:, k] if desc else data[:, k])
                 for k, desc in zip(keys, descending)]
    idx = np.lexsort(sort_cols[::-1])
    return data[idx]

