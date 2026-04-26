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

def save_csv(
    array: np.ndarray,
    filepath: str,
    delimiter: str = ",",
    header: Optional[str] = None,
    fmt: str = "%.8g",
) -> None:
    """Save a 2-D NumPy array to a CSV file.

    Parameters
    ----------
    array     : np.ndarray, shape (n_rows, n_cols)
    filepath  : str
    delimiter : str, default ','
    header    : str or None
    fmt       : str, default '%.8g'

    Raises
    ------
    ValueError
        If array is not 1-D or 2-D.

    """
    if array.ndim not in (1, 2):
        raise ValueError(f"array must be 1-D or 2-D, got shape {array.shape}.")
    np.savetxt(filepath, array, delimiter=delimiter,
               header=header or "", fmt=fmt, comments="")
    
def load_csv_chunks(
    filepath: str,
    chunk_size: int = 1000,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: type = float,
    fill_value: float = np.nan,
    encoding: str = "utf-8",
) -> Iterator[np.ndarray]:
    """Yield successive chunk_size-row chunks from a CSV file.

    Parameters
    ----------
    filepath   : str
    chunk_size : int, default 1000
    delimiter  : str, default ','
    skip_header: bool, default True
    dtype      : type, default float
    fill_value : float, default np.nan
    encoding   : str, default 'utf-8'

    Yields
    ------
    np.ndarray, shape (<=chunk_size, n_cols)
    """
    with open(filepath, "r", encoding=encoding) as fh:
        if skip_header:
            next(fh)
        buffer: list[str] = []
        for line in fh:
            buffer.append(line)
            if len(buffer) >= chunk_size:
                yield _parse_lines(buffer, delimiter, dtype, fill_value)
                buffer = []
        if buffer:
            yield _parse_lines(buffer, delimiter, dtype, fill_value)


def _parse_lines(
    lines: list[str],
    delimiter: str,
    dtype: type,
    fill_value: float,
) -> np.ndarray:
    """Parse a list of raw CSV text lines into a 2-D array."""
    import io as _io
    text = "".join(lines)
    arr = np.genfromtxt(
        _io.StringIO(text),
        delimiter=delimiter,
        filling_values=fill_value,
        dtype=dtype,
        autostrip=True,
    )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

