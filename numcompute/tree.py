"""
tree.py — Streaming-compatible Decision Tree Classifier for NumCompute.
"""

from __future__ import annotations
import numpy as np


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        max_features=None,
        criterion: str = "gini",
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.root_ = None
        self.X_seen_ = None
        self.y_seen_ = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = self._validate_xy(X, y)
        self.X_seen_ = X.copy()
        self.y_seen_ = y.copy()
        self.classes_ = np.unique(y)
        self.root_ = self._build_tree(X, y, depth=0)
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
        self.root_ = self._build_tree(self.X_seen_, self.y_seen_, depth=0)
        return self

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Call fit() or partial_fit() before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return np.array([self._predict_one(row, self.root_) for row in X])

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

    def _build_tree(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return self._make_leaf(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._make_leaf(y)

        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _make_leaf(self, y):
        values, counts = np.unique(y, return_counts=True)
        majority = values[np.argmax(counts)]
        return {"type": "leaf", "class": majority}

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        features = np.arange(n_features)

        if self.max_features is not None:
            if isinstance(self.max_features, int):
                k = min(self.max_features, n_features)
            elif self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            else:
                k = n_features
            features = np.random.choice(features, size=k, replace=False)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        parent_impurity = self._impurity(y)

        for feature in features:
            col = X[:, feature]
            col = col[~np.isnan(col)]

            if col.size == 0:
                continue

            thresholds = np.unique(col)

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                weighted_impurity = (
                    len(left_y) / n_samples * self._impurity(left_y)
                    + len(right_y) / n_samples * self._impurity(right_y)
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)

        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _predict_one(self, row, node):
        while node["type"] != "leaf":
            feature = node["feature"]
            threshold = node["threshold"]

            value = row[feature]

            if np.isnan(value):
                node = node["left"]
            elif value <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        return node["class"]