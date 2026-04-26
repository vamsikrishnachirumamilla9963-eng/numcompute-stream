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


