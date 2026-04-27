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

def topk(
    values: np.ndarray,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-k values and their indices via np.argpartition.

    Parameters
    ----------
    values         : np.ndarray, shape (n,)
    k              : int — number of elements; clamped to [1, n]
    largest        : bool, default True
    return_indices : bool, default True

    Returns
    -------
    top_values  : np.ndarray, shape (k,) — sorted
    top_indices : np.ndarray of int, shape (k,)  [only if return_indices=True]

    Raises
    ------
    ValueError
        If values is not 1-D.

    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}.")
    n = len(values)
    if n == 0:
        empty = np.array([], dtype=float)
        return (empty, np.array([], dtype=int)) if return_indices else empty
    k = int(np.clip(k, 1, n))
    if largest:
        part_idx = np.argpartition(values, n - k)[-k:]
        order    = np.argsort(values[part_idx], kind="stable")[::-1]
    else:
        part_idx = np.argpartition(values, k - 1)[:k]
        order    = np.argsort(values[part_idx], kind="stable")
    top_indices = part_idx[order]
    top_values  = values[top_indices]
    return (top_values, top_indices) if return_indices else top_values

def quickselect(
    values: np.ndarray,
    k: int,
    largest: bool = True,
) -> float:
    """Return the k-th order statistic using randomised quickselect.

    This is an O(n) expected-time algorithm provided for educational
    purposes. For production use prefer topk.

    Parameters
    ----------
    values  : np.ndarray, shape (n,)
    k       : int — 1-based rank (k=1 -> largest/smallest element)
    largest : bool, default True

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If values is not 1-D or k is out of range.

    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}.")
    n = len(values)
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}], got {k}.")
    arr    = values.copy()
    target = (n - k) if largest else (k - 1)
    return _quickselect_inplace(arr, 0, n - 1, target)


def _quickselect_inplace(
    arr: np.ndarray, lo: int, hi: int, target: int
) -> float:
    """Randomised in-place quickselect."""
    while lo < hi:
        pivot_idx = np.random.randint(lo, hi + 1)
        arr[[pivot_idx, hi]] = arr[[hi, pivot_idx]]
        pivot_val = arr[hi]
        store = lo
        for i in range(lo, hi):
            if arr[i] < pivot_val:
                arr[[store, i]] = arr[[i, store]]
                store += 1
        arr[[store, hi]] = arr[[hi, store]]
        if   store == target: return float(arr[store])
        elif store  < target: lo = store + 1
        else:                 hi = store - 1
    return float(arr[lo])

def binary_search(
    sorted_array: np.ndarray,
    x: float,
) -> Tuple[int, bool]:
    """Search for x in a sorted 1-D array.

    Parameters
    ----------
    sorted_array : np.ndarray, shape (n,) — ascending
    x            : float

    Returns
    -------
    index : int   — insertion point
    found : bool  — True if x is present

    Raises
    ------
    ValueError
        If sorted_array is not 1-D.

    """
    sorted_array = np.asarray(sorted_array)
    if sorted_array.ndim != 1:
        raise ValueError(f"sorted_array must be 1-D, got shape {sorted_array.shape}.")
    idx   = int(np.searchsorted(sorted_array, x, side="left"))
    found = bool(idx < len(sorted_array) and sorted_array[idx] == x)
    return idx, found

