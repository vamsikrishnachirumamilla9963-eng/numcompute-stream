"""
utils.py — Shared mathematical utilities for NumCompute.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Iterator, Optional, Union


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    return float(np.sqrt(np.sum((a-b)**2)))

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two vectors."""
    return float(np.sum(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 for zero-norm vectors."""
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0. if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))

def pairwise_euclidean(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pairwise Euclidean distance matrix via broadcasting — no Python loops.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    Y : np.ndarray, shape (m, d) or None — if None, compute X vs X

    Returns
    -------
    np.ndarray, shape (n, m)

    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(XX + YY - 2.*X@Y.T, 0.))
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid — two-branch to avoid overflow."""
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x)
    out[ pos] = 1./(1. + np.exp(-x[ pos]))
    ex        = np.exp(x[~pos])
    out[~pos] = ex/(1. + ex)
    return out

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-shifted softmax — prevents overflow."""
    x = np.asarray(x, dtype=float)
    s = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=axis, keepdims=True)

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, x)."""
    return np.maximum(0., np.asarray(x, dtype=float))

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: x if x > 0 else alpha * x."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)

def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(np.asarray(x, dtype=float))

def logsumexp(
    x: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    x   = np.asarray(x, dtype=float)
    c   = np.max(x, axis=axis, keepdims=True)
    res = np.log(np.sum(np.exp(x-c), axis=axis, keepdims=True)) + c
    if axis is None:
        return float(res.ravel()[0])
    return np.squeeze(res, axis=axis)
