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

def percentile(
    data: np.ndarray,
    q: Union[float, np.ndarray],
    interpolation: str = "linear",
) -> Union[float, np.ndarray]:
    """Compute percentile(s) of a 1-D array, ignoring NaN.

    Parameters
    ----------
    data          : np.ndarray, shape (n,)
    q             : float or array-like — percentile(s) in [0, 100]
    interpolation : {'linear','lower','higher','midpoint','nearest'}

    Returns
    -------
    float or np.ndarray

    Raises
    ------
    ValueError
        If data is not 1-D, q is outside [0,100], or interpolation invalid.

    """
    valid_interp = ("linear","lower","higher","midpoint","nearest")
    if interpolation not in valid_interp:
        raise ValueError(f"interpolation must be one of {valid_interp}.")
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {data.shape}.")
    q       = np.asarray(q, dtype=float)
    scalar  = q.ndim == 0
    q       = np.atleast_1d(q)
    if np.any((q < 0) | (q > 100)):
        raise ValueError("All values of q must be in [0, 100].")
    clean = data[~np.isnan(data)]
    n     = len(clean)
    if n == 0:
        result = np.full(len(q), np.nan)
        return float(result[0]) if scalar else result
    sd    = np.sort(clean, kind="stable")
    vidx  = (q / 100.0) * (n - 1)
    lo    = np.clip(np.floor(vidx).astype(int), 0, n-1)
    hi    = np.clip(np.ceil(vidx).astype(int),  0, n-1)
    frac  = vidx - lo
    if   interpolation == "lower":    result = sd[lo].astype(float)
    elif interpolation == "higher":   result = sd[hi].astype(float)
    elif interpolation == "nearest":  result = sd[np.where(frac < 0.5, lo, hi)].astype(float)
    elif interpolation == "midpoint": result = (sd[lo] + sd[hi]) / 2.0
    else:                             result = sd[lo] + frac * (sd[hi] - sd[lo])
    return float(result[0]) if scalar else result

