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

def jacobian(
    F: Callable,
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """Jacobian matrix of vector-valued F via finite differences.

    Parameters
    ----------
    F      : callable — F(x) -> np.ndarray, shape (m,)
    x      : np.ndarray, shape (n,)
    h      : float, default 1e-5
    method : {'central','forward','backward'}

    Returns
    -------
    np.ndarray, shape (m, n)

    Raises
    ------
    ValueError — non-1D x or invalid method

    """
    valid = ("central","forward","backward")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'.")
    x  = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}.")
    n  = len(x)
    F0 = np.asarray(F(x), dtype=float)
    m  = F0.size
    J  = np.empty((m, n), dtype=float)
    if method == "central":
        for j in range(n):
            xp=x.copy(); xp[j]+=h
            xm=x.copy(); xm[j]-=h
            J[:,j] = (np.asarray(F(xp),dtype=float)-np.asarray(F(xm),dtype=float))/(2.*h)
    elif method == "forward":
        for j in range(n):
            xp=x.copy(); xp[j]+=h
            J[:,j] = (np.asarray(F(xp),dtype=float)-F0)/h
    else:
        for j in range(n):
            xm=x.copy(); xm[j]-=h
            J[:,j] = (F0-np.asarray(F(xm),dtype=float))/h
    return J


def backtracking_line_search(
    f: Callable,
    x: np.ndarray,
    direction: np.ndarray,
    grad_x: np.ndarray,
    alpha: float = 1.,
    rho: float = .5,
    c: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Backtracking Armijo line search.

    Finds step size satisfying: f(x + a*d) <= f(x) + c*a*grad_x.d

    Parameters
    ----------
    f, x, direction, grad_x : standard gradient descent inputs
    alpha : float — initial step size
    rho   : float — reduction factor (0 < rho < 1)
    c     : float — Armijo constant
    max_iter : int

    Returns
    -------
    float — accepted step size
    """
    f0    = f(x)
    slope = float(np.dot(grad_x.ravel(), direction.ravel()))
    for _ in range(max_iter):
        if f(x + alpha*direction) <= f0 + c*alpha*slope:
            return alpha
        alpha *= rho
    return alpha

