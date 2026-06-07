"""
stream.py — Streaming trainer for chunk-based learning in NumCompute.
"""

from __future__ import annotations
import sys
import numpy as np

from .metrics import accuracy


class StreamTrainer:
    def __init__(self, model):
        self.model = model
        self.logs_ = []
        self.total_correct_ = 0
        self.total_seen_ = 0
        self.chunk_index_ = 0

    def fit_chunk(self, X, y):
        X, y = self._validate_xy(X, y)

        if not hasattr(self.model, "partial_fit"):
            raise TypeError("model must implement partial_fit().")

        self.model.partial_fit(X, y)

        y_pred = self.model.predict(X)
        chunk_accuracy = accuracy(y, y_pred)

        correct = int(np.sum(y == y_pred))
        self.total_correct_ += correct
        self.total_seen_ += len(y)
        self.chunk_index_ += 1

        cumulative_accuracy = (
            self.total_correct_ / self.total_seen_
            if self.total_seen_ > 0
            else 0.0
        )

        log = {
            "chunk": self.chunk_index_,
            "chunk_accuracy": float(chunk_accuracy),
            "cumulative_accuracy": float(cumulative_accuracy),
            "n_samples": int(len(y)),
            "memory_bytes": self._memory_footprint(),
        }

        self.logs_.append(log)
        return log

    def score_chunk(self, X, y):
        X, y = self._validate_xy(X, y)

        if not hasattr(self.model, "predict"):
            raise TypeError("model must implement predict().")

        y_pred = self.model.predict(X)
        return accuracy(y, y_pred)

    def get_logs(self):
        return list(self.logs_)

    def reset_logs(self):
        self.logs_ = []
        self.total_correct_ = 0
        self.total_seen_ = 0
        self.chunk_index_ = 0
        return self

    def _validate_xy(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        return X, y

    def _memory_footprint(self):
        size = sys.getsizeof(self.model)

        for attr in ("X_seen_", "y_seen_"):
            value = getattr(self.model, attr, None)
            if isinstance(value, np.ndarray):
                size += value.nbytes

        return int(size)