"""
ensemble.py — Streaming-compatible ensemble classifier for NumCompute.
"""

from __future__ import annotations
import numpy as np

from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 3,
        min_samples_split: int = 2,
        max_features="sqrt",
        criterion: str = "gini",
        bootstrap: bool = True,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_ = []
        self.X_seen_ = None
        self.y_seen_ = None
        self.classes_ = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X, y):
        X, y = self._validate_xy(X, y)
        self.X_seen_ = X.copy()
        self.y_seen_ = y.copy()
        self.classes_ = np.unique(y)
        self._fit_trees(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._validate_xy(X_chunk, y_chunk)

        if self.X_seen_ is None:
            self.X_seen_ = X_chunk.copy()
            self.y_seen_ = y_chunk.copy()
        else:
            self.X_seen_ = np.vstack([self.X_seen_, X_chunk])
            self.y_seen_ = np.concatenate([self.y_seen_, y_chunk])

        self.classes_ = np.unique(self.y_seen_)
        self._fit_trees(self.X_seen_, self.y_seen_)
        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("Call fit() or partial_fit() before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return np.array([self._majority_vote(all_preds[:, i]) for i in range(X.shape[0])])

    def _fit_trees(self, X, y):
        self.trees_ = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng_.integers(0, n_samples, size=n_samples)
                X_train = X[idx]
                y_train = y[idx]
            else:
                X_train = X
                y_train = y

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            tree.fit(X_train, y_train)
            self.trees_.append(tree)

    def _majority_vote(self, predictions):
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

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