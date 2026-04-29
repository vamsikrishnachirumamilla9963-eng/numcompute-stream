"""
pipeline.py — Pipeline abstraction for NumCompute.
"""

from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, Tuple

class Pipeline:

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self._validate_steps(steps)
        self.steps = steps

    def _validate_steps(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [n for n,_ in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique; got {names}.")
        for i,(name,step) in enumerate(steps[:-1]):
            if not (hasattr(step,"fit") and hasattr(step,"transform")):
                raise TypeError(
                    f"Intermediate step '{name}' (index {i}) must implement "
                    f"fit() and transform()."
                )

    def __getitem__(self, key: str) -> Any:
        for name,step in self.steps:
            if name==key: return step
        raise KeyError(f"No step named '{key}'.")

    def named_steps(self) -> dict:
        return {n: s for n,s in self.steps}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pipeline":
        """Fit all steps sequentially.
        """
        x_cur = X
        for _,step in self.steps[:-1]:
            x_cur = step.fit_transform(x_cur, y)
        _,last = self.steps[-1]
        if hasattr(last,"fit"):
            last.fit(x_cur, y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        last = self.steps[-1][1]
        if hasattr(last,"predict") and not hasattr(last,"transform"):
            raise RuntimeError("Last step is an Estimator; use predict() instead.")
        x_cur = X
        for _,step in self.steps:
            x_cur = step.transform(x_cur)
        return x_cur

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform then predict via last step."""
        last = self.steps[-1][1]
        if not hasattr(last,"predict"):
            raise RuntimeError("Last step does not implement predict().")
        x_cur = X
        for _,step in self.steps[:-1]:
            x_cur = step.transform(x_cur)
        return last.predict(x_cur)
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