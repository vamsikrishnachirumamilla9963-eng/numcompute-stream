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