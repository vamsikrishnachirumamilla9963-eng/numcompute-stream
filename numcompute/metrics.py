"""
metrics.py — Evaluation metrics for NumCompute.

All routines are fully vectorised — no Python loops over data elements.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def _check_labels(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}."
        )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (n,)

    Returns
    -------
    float in [0, 1]

    Raises
    ------
    ValueError — shape mismatch or not 1-D

    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_labels(y_true, y_pred)
    return 0.0 if len(y_true) == 0 else float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confusion matrix — fully vectorised via np.searchsorted + np.add.at.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (n,)
    labels         : array-like or None

    Returns
    -------
    matrix : np.ndarray, shape (n_classes, n_classes)
    labels : np.ndarray

    Raises
    ------
    ValueError — shape mismatch or not 1-D

    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_labels(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels    = np.asarray(labels)
    nc        = len(labels)
    so        = np.argsort(labels, kind="stable")
    ls        = labels[so]                        # sorted labels
    tp        = np.searchsorted(ls, y_true)
    pp        = np.searchsorted(ls, y_pred)
    safe_t    = np.clip(tp, 0, nc-1)
    safe_p    = np.clip(pp, 0, nc-1)
    mask      = (ls[safe_t] == y_true) & (ls[safe_p] == y_pred)
    ti        = so[tp[mask]]
    pi        = so[pp[mask]]
    matrix    = np.zeros((nc, nc), dtype=int)
    np.add.at(matrix, (ti, pi), 1)
    return matrix, labels

