"""
optim.py — Gradient and Jacobian estimation via finite differences.
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def grad(
    f: Callable,
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """Numerical gradient of scalar f via finite differences.

    Parameters
    ----------
    f      : callable — f(x) -> float
    x      : np.ndarray, shape (n,)
    h      : float, default 1e-5
    method : {'central','forward','backward'}
             central  O(h^2), forward/backward O(h)

    Returns
    -------
    np.ndarray, shape (n,)

    Raises
    ------
    ValueError — non-1D x or invalid method

    """
    valid = ("central","forward","backward")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'.")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}.")
    n  = len(x)
    g  = np.empty(n, dtype=float)
    if method == "central":
        for i in range(n):
            xp=x.copy(); xp[i]+=h
            xm=x.copy(); xm[i]-=h
            g[i] = (f(xp)-f(xm))/(2.*h)
    elif method == "forward":
        f0 = f(x)
        for i in range(n):
            xp=x.copy(); xp[i]+=h
            g[i] = (f(xp)-f0)/h
    else:
        f0 = f(x)
        for i in range(n):
            xm=x.copy(); xm[i]-=h
            g[i] = (f0-f(xm))/h
    return g

