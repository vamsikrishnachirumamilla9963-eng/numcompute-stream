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

def _prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    average: str,
    pos_label: int,
) -> float:
    """Vectorised P/R/F1: (n,1)x(1,C) broadcast — no loop over samples."""
    valid = ("macro","micro","binary")
    if average not in valid:
        raise ValueError(f"average must be one of {valid}, got '{average}'.")
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_labels(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        classes = np.array([pos_label])
    y_t = y_true[:, np.newaxis]; y_p = y_pred[:, np.newaxis]; c = classes[np.newaxis,:]
    tp  = np.sum((y_p==c)&(y_t==c), axis=0).astype(float)
    fp  = np.sum((y_p==c)&(y_t!=c), axis=0).astype(float)
    fn  = np.sum((y_p!=c)&(y_t==c), axis=0).astype(float)
    p   = np.where((tp+fp)>0, tp/(tp+fp), 0.)
    r   = np.where((tp+fn)>0, tp/(tp+fn), 0.)
    if average == "micro":
        tp_s=tp.sum(); fp_s=fp.sum(); fn_s=fn.sum()
        pm = float(tp_s/(tp_s+fp_s)) if (tp_s+fp_s)>0 else 0.
        rm = float(tp_s/(tp_s+fn_s)) if (tp_s+fn_s)>0 else 0.
        if metric=="precision": return pm
        if metric=="recall":    return rm
        d  = pm+rm; return float(2*pm*rm/d) if d>0 else 0.
    if metric=="precision": return float(np.mean(p))
    if metric=="recall":    return float(np.mean(r))
    denom = p+r
    return float(np.mean(np.where(denom>0, 2*p*r/denom, 0.)))


def precision(y_true, y_pred, average="macro", pos_label=1) -> float:
    """Precision score. average: {'macro','micro','binary'}"""
    return _prf(y_true, y_pred, "precision", average, pos_label)

def recall(y_true, y_pred, average="macro", pos_label=1) -> float:
    """Recall score. average: {'macro','micro','binary'}"""
    return _prf(y_true, y_pred, "recall", average, pos_label)

def f1(y_true, y_pred, average="macro", pos_label=1) -> float:
    """F1 score. average: {'macro','micro','binary'}"""
    return _prf(y_true, y_pred, "f1", average, pos_label)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Complexity: Time O(n), Space O(1)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_labels(y_true, y_pred)
    return float(np.mean((y_true - y_pred)**2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_labels(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2.

    Returns 1.0 if SS_tot=0 and SS_res=0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_labels(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    if ss_tot == 0.:
        return 1. if ss_res == 0. else 0.
    return float(1. - ss_res/ss_tot)

def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve via cumulative TP/FP counts.

    Parameters
    ----------
    y_true   : np.ndarray, shape (n,) — binary labels
    y_score  : np.ndarray, shape (n,) — continuous scores
    pos_label: int, default 1

    Returns
    -------
    fpr, tpr, thresholds : np.ndarray

    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}.")
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}.")
    desc        = np.argsort(y_score, kind="stable")[::-1]
    ys          = (y_true[desc] == pos_label).astype(int)
    thresh      = y_score[desc]
    tp_c        = np.cumsum(ys)
    fp_c        = np.cumsum(1 - ys)
    n_pos       = int(tp_c[-1]) if len(tp_c) else 0
    n_neg       = int(fp_c[-1]) if len(fp_c) else 0
    tpr         = tp_c / max(n_pos, 1)
    fpr         = fp_c / max(n_neg, 1)
    tpr         = np.concatenate([[0.], tpr])
    fpr         = np.concatenate([[0.], fpr])
    thresh      = np.concatenate([[thresh[0]+1.], thresh])
    return fpr, tpr, thresh


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve via trapezoidal rule.

    """
    fpr    = np.asarray(fpr, dtype=float)
    tpr    = np.asarray(tpr, dtype=float)
    order  = np.argsort(fpr, kind="stable")
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(tpr[order], fpr[order]))



# Streaming Metrics


class StreamingAccuracy:
    """Streaming accuracy metric with update/reset/result API."""

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        _check_labels(y_true, y_pred)

        self.correct_ += int(np.sum(y_true == y_pred))
        self.total_ += int(len(y_true))
        return self

    def result(self):
        return 0.0 if self.total_ == 0 else float(self.correct_ / self.total_)

    def reset(self):
        self.correct_ = 0
        self.total_ = 0
        return self


class StreamingClassificationMetrics:
    """
    Streaming classification metrics.

    Maintains accumulated y_true and y_pred values so accuracy, precision,
    recall, f1, and confusion matrix can be computed over all seen chunks.
    """

    def __init__(self, average="macro", pos_label=1, rolling_window=None):
        self.average = average
        self.pos_label = pos_label
        self.rolling_window = rolling_window
        self.reset()

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        _check_labels(y_true, y_pred)

        self.y_true_ = np.concatenate([self.y_true_, y_true])
        self.y_pred_ = np.concatenate([self.y_pred_, y_pred])

        if self.rolling_window is not None and len(self.y_true_) > self.rolling_window:
            self.y_true_ = self.y_true_[-self.rolling_window:]
            self.y_pred_ = self.y_pred_[-self.rolling_window:]

        return self

    def result(self):
        if len(self.y_true_) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": np.zeros((0, 0), dtype=int),
                "labels": np.array([]),
            }

        cm, labels = confusion_matrix(self.y_true_, self.y_pred_)

        return {
            "accuracy": accuracy(self.y_true_, self.y_pred_),
            "precision": precision(
                self.y_true_,
                self.y_pred_,
                average=self.average,
                pos_label=self.pos_label,
            ),
            "recall": recall(
                self.y_true_,
                self.y_pred_,
                average=self.average,
                pos_label=self.pos_label,
            ),
            "f1": f1(
                self.y_true_,
                self.y_pred_,
                average=self.average,
                pos_label=self.pos_label,
            ),
            "confusion_matrix": cm,
            "labels": labels,
        }

    def reset(self):
        self.y_true_ = np.array([])
        self.y_pred_ = np.array([])
        return self