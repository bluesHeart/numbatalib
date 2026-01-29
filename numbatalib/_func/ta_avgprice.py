from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _avgprice_kernel(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray) -> None:
    n = open_.shape[0]
    for i in range(n):
        out[i] = (open_[i] + high[i] + low[i] + close[i]) * 0.25


def AVGPRICE(open, high, low, close):
    """
    Average Price
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = o.shape[0]
    if h.shape[0] != n or l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    out = np.empty(n, dtype=np.float64)
    _avgprice_kernel(o, h, l, c, out)
    return out

