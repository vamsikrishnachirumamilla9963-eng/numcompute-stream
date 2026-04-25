"""
io.py — Data I/O utilities for NumCompute.

Provides CSV loading with support for custom delimiters, missing-value
handling, streaming/chunked reads, and dtype coercion.
"""

from __future__ import annotations
import numpy as np
from typing import Iterator, Optional

def load_csv(
    filepath: str,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: type = float,
    fill_value: float = np.nan,
    encoding: str = "utf-8",
) -> np.ndarray:
    """Load a CSV file into a 2-D NumPy array.

    Missing cells are replaced with fill_value.

    Parameters
    ----------
    filepath  : str
    delimiter : str, default ','
    skip_header : bool, default True
    dtype     : type, default float
    fill_value : float, default np.nan
    encoding  : str, default 'utf-8'

    Returns
    -------
    np.ndarray, shape (n_rows, n_cols)

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If the file is empty after skipping the header.

    Complexity
    ----------
    Time : O(n_rows x n_cols)
    Space: O(n_rows x n_cols)
    """
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=int(skip_header),
        filling_values=fill_value,
        dtype=dtype,
        encoding=encoding,
        autostrip=True,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise ValueError(f"File '{filepath}' contains no data rows.")
    return data
