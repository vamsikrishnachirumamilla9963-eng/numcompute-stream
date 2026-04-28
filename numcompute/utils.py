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

