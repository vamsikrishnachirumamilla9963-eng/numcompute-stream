"""
rank.py — Ranking and percentile utilities.

Provides rank() with five tie-handling methods (average, dense, ordinal,
min, max) and percentile() with five interpolation strategies.
All routines are vectorised via NumPy.
"""

from __future__ import annotations
import numpy as np
from typing import Union


def rank(
    data: np.ndarray,
    method: str = "average",
    ascending: bool = True,
) -> np.ndarray:
    """Rank elements of a 1-D array with configurable tie handling.

    Parameters
    ----------
    data      : np.ndarray, shape (n,) — NaN values receive NaN rank
    method    : {'average','dense','ordinal','min','max'}
    ascending : bool, default True

    Returns
    -------
    np.ndarray, shape (n,), dtype float — ranks from 1

    Raises
    ------
    ValueError
        If data is not 1-D or method is invalid.

    """
    valid = ("average", "dense", "ordinal", "min", "max")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'.")
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}.")

    n     = len(data)
    ranks = np.full(n, np.nan)
    mask  = ~np.isnan(data)
    vdata = data[mask]
    nv    = vdata.size
    if nv == 0:
        return ranks

    order = np.argsort(vdata, kind="stable")
    if not ascending:
        order = order[::-1]
    svals = vdata[order]

    diff = np.empty(nv, dtype=bool)
    diff[0]  = True
    diff[1:] = svals[1:] != svals[:-1]
    gids     = np.cumsum(diff)

    ordinal  = np.arange(1, nv + 1, dtype=float)

    if   method == "ordinal": r = ordinal
    elif method == "dense":   r = gids.astype(float)
    else:
        buf = np.empty(nv, dtype=float)
        _fill_group_stat(ordinal, gids, buf,
                         "mean" if method == "average" else method)
        r = buf

    result = np.empty(nv, dtype=float)
    result[order] = r
    ranks[np.where(mask)[0]] = result
    return ranks


def _fill_group_stat(
    values: np.ndarray,
    group_ids: np.ndarray,
    out: np.ndarray,
    stat: str,
) -> None:
    """Vectorised per-group mean/min/max via np.add.at and np.minimum.at."""
    n_groups = int(group_ids[-1])
    counts = np.bincount(group_ids - 1, minlength=n_groups)
    sums   = np.bincount(group_ids - 1, weights=values, minlength=n_groups)
    mins   = np.full(n_groups,  np.inf)
    maxs   = np.full(n_groups, -np.inf)
    np.minimum.at(mins, group_ids - 1, values)
    np.maximum.at(maxs, group_ids - 1, values)
    if   stat == "mean": gs = sums / counts
    elif stat == "min":  gs = mins
    else:                gs = maxs
    out[:] = gs[group_ids - 1]

