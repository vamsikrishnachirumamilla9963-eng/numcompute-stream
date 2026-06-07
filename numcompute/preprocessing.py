"""
preprocessing.py — Data preprocessing transformers for NumCompute.

All transformers follow the fit / transform / fit_transform API.
Implementations are fully vectorised; no Python-level loops over data.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class _BaseTransformer:
    """Mixin providing default fit_transform and input validation."""

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit to X and return transformed result."""
        return self.fit(X, y).transform(X)

    def _validate(self, X: np.ndarray, *, fitted: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}.")
        if fitted and not hasattr(self, "_fitted"):
            raise RuntimeError("Call fit() before transform().")
        return X


class StandardScaler(_BaseTransformer):
    """Standardise features to zero mean and unit variance (z-score).

    Attributes
    ----------
    mean_  : np.ndarray, shape (n_features,)
    scale_ : np.ndarray, shape (n_features,)
        Zero-variance features get scale_=1 to avoid division by zero.
    """

    def fit(self, X: np.ndarray, y=None) -> "StandardScaler":
        """Compute per-feature mean and std.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self

        Complexity
        ----------
        Time : O(n_samples x n_features)
        Space: O(n_features)
        """
        X = self._validate(X)
        self.mean_  = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self._fitted = True
        return self
    def partial_fit(self, X: np.ndarray, y=None) -> "StandardScaler":
        """Update running mean and variance from a new chunk."""
        X = self._validate(X)

        chunk_count = X.shape[0]
        chunk_mean = np.nanmean(X, axis=0)
        chunk_var = np.nanvar(X, axis=0)

        if not hasattr(self, "n_seen_"):
            self.n_seen_ = 0
            self.mean_ = np.zeros(X.shape[1])
            self.var_ = np.zeros(X.shape[1])

        total_count = self.n_seen_ + chunk_count
        delta = chunk_mean - self.mean_

        new_mean = self.mean_ + delta * (chunk_count / total_count)
        new_var = (
            (self.n_seen_ * self.var_)
            + (chunk_count * chunk_var)
            + (delta ** 2) * self.n_seen_ * chunk_count / total_count
        ) / total_count

        self.n_seen_ = total_count
        self.mean_ = new_mean
        self.var_ = new_var
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score standardisation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            NaN values are preserved.

        Complexity
        ----------
        Time : O(n_samples x n_features)
        Space: O(n_samples x n_features)
        """
        X = self._validate(X, fitted=True)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardisation."""
        X = self._validate(X, fitted=True)
        return X * self.scale_ + self.mean_
    
class MinMaxScaler(_BaseTransformer):
    """The function helps with maintaining the range for the array [feature_range[0], feature_range[1]].

    Attributes
    ----------
    data_min_ : np.ndarray
    data_max_ : np.ndarray
    scale_    : np.ndarray
    min_      : np.ndarray
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must satisfy lo < hi, got {feature_range}.")
        self.feature_range = feature_range

    def fit(self, X: np.ndarray, y=None) -> "MinMaxScaler":
        """Compute per-feature min and max.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range  = np.where(data_range == 0, 1.0, data_range)
        lo, hi = self.feature_range
        self.scale_ = (hi - lo) / data_range
        self.min_   = lo - self.data_min_ * self.scale_
        self._fitted = True
        return self
    def partial_fit(self, X: np.ndarray, y=None) -> "MinMaxScaler":
        """Update running min/max from a new chunk."""
        X = self._validate(X)

        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if not hasattr(self, "data_min_"):
            self.data_min_ = chunk_min
            self.data_max_ = chunk_max
        else:
            self.data_min_ = np.minimum(self.data_min_, chunk_min)
            self.data_max_ = np.maximum(self.data_max_, chunk_max)

        data_range = self.data_max_ - self.data_min_
        data_range = np.where(data_range == 0, 1.0, data_range)

        lo, hi = self.feature_range
        self.scale_ = (hi - lo) / data_range
        self.min_ = lo - self.data_min_ * self.scale_
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to the fitted range.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        """
        X = self._validate(X, fitted=True)
        return X * self.scale_ + self.min_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the min-max scaling."""
        X = self._validate(X, fitted=True)
        return (X - self.min_) / self.scale_
    
class SimpleImputer(_BaseTransformer):
    """Replace NaN values with a per-column statistic.

    Parameters
    ----------
    strategy   : {'mean', 'median', 'constant'}
    fill_value : float, default 0.0  (used only when strategy='constant')

    Attributes
    ----------
    statistics_ : np.ndarray, shape (n_features,)
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(
                f"strategy must be 'mean', 'median', or 'constant'; got '{strategy}'."
            )
        self.strategy   = strategy
        self.fill_value = fill_value

    def fit(self, X: np.ndarray, y=None) -> "SimpleImputer":
        """Compute per-feature fill statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        with np.errstate(all="ignore"):   # suppress all-NaN column warning
            if self.strategy == "mean":
                self.statistics_ = np.nanmean(X, axis=0)
            elif self.strategy == "median":
                self.statistics_ = np.nanmedian(X, axis=0)
            else:
                self.statistics_ = np.full(X.shape[1], self.fill_value)
        self._fitted = True
        return self
    def partial_fit(self, X: np.ndarray, y=None) -> "SimpleImputer":
        """Update imputation statistics from a new chunk."""
        X = self._validate(X)

        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value)
            self._fitted = True
            return self

        if not hasattr(self, "_stored_values"):
            self._stored_values = [[] for _ in range(X.shape[1])]

        for j in range(X.shape[1]):
            clean = X[:, j][~np.isnan(X[:, j])]
            if clean.size > 0:
                self._stored_values[j].extend(clean.tolist())

        stats = []
        for values in self._stored_values:
            if len(values) == 0:
                stats.append(np.nan)
            elif self.strategy == "mean":
                stats.append(float(np.mean(values)))
            else:
                stats.append(float(np.median(values)))

        self.statistics_ = np.asarray(stats)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN values with fitted statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features) — copy with NaNs replaced.

        """
        X = self._validate(X, fitted=True).copy()
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return X
    
class OneHotEncoder(_BaseTransformer):
    """Encode integer categorical features as one-hot binary columns.

    Attributes
    ----------
    categories_ : list of np.ndarray
        Sorted unique categories per input feature column.
    """

    def fit(self, X: np.ndarray, y=None) -> "OneHotEncoder":
        """Learn categories for each feature column.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self

        """
        X = self._validate(X)
        self.categories_ = [
            np.unique(X[:, j][~np.isnan(X[:, j])])
            for j in range(X.shape[1])
        ]
        self._fitted = True
        return self
    def partial_fit(self, X: np.ndarray, y=None) -> "OneHotEncoder":
        """Update learned categories from a new chunk."""
        X = self._validate(X)

        if not hasattr(self, "categories_"):
            self.categories_ = [
                np.unique(X[:, j][~np.isnan(X[:, j])])
                for j in range(X.shape[1])
            ]
        else:
            for j in range(X.shape[1]):
                new_cats = np.unique(X[:, j][~np.isnan(X[:, j])])
                self.categories_[j] = np.unique(
                    np.concatenate([self.categories_[j], new_cats])
                )

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, sum(len(cats)))
            Unknown categories produce an all-zero row segment.

        """
        X = self._validate(X, fitted=True)
        parts = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j:j+1]
            parts.append((col == cats[np.newaxis, :]).astype(float))
        return np.hstack(parts)

    def get_feature_names_out(self) -> list:
        """Return output feature names as list of strings."""
        names = []
        for j, cats in enumerate(self.categories_):
            names.extend([f"x{j}_{c}" for c in cats])
        return names
    
