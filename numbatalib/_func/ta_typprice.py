from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _typprice_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray) -> None:
    n = high.shape[0]
    for i in range(n):
        out[i] = (high[i] + low[i] + close[i]) / 3.0


def TYPPRICE(high, low, close):
    """
    Typical Price
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    out = np.empty(n, dtype=np.float64)
    _typprice_kernel(h, l, c, out)
    return out

