from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like


@njit(cache=True)
def _bop_kernel(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray) -> None:
    n = open_.shape[0]
    for i in range(n):
        denom = high[i] - low[i]
        if denom != 0.0:
            out[i] = (close[i] - open_[i]) / denom
        else:
            out[i] = 0.0


def BOP(open, high, low, close):
    """
    Balance Of Power
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = o.shape[0]
    if h.shape[0] != n or l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.float64)
    _bop_kernel(o, h, l, c, out)
    return out

