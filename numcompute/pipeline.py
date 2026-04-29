"""
pipeline.py — Pipeline abstraction for NumCompute.
"""

from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, Tuple


class Transformer:
    """Base class for transformers (fit + transform protocol)."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Transformer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class Estimator:
    """Base class for estimators (fit + predict protocol)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError