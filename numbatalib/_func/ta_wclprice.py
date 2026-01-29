from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _wclprice_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray) -> None:
    n = high.shape[0]
    for i in range(n):
        out[i] = (high[i] + low[i] + 2.0 * close[i]) * 0.25


def WCLPRICE(high, low, close):
    """
    Weighted Close Price
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    out = np.empty(n, dtype=np.float64)
    _wclprice_kernel(h, l, c, out)
    return out

