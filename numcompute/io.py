"""
io.py — Data I/O utilities for NumCompute.

Provides CSV loading with support for custom delimiters, missing-value
handling, streaming/chunked reads, and dtype coercion.
"""

from __future__ import annotations
import numpy as np
from typing import Iterator, Optional