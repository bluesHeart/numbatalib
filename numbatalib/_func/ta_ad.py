from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like


@njit(cache=True)
def _ad_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, out: np.ndarray) -> None:
    n = high.shape[0]
    ad = 0.0
    for i in range(n):
        tmp = high[i] - low[i]
        if tmp > 0.0:
            ad += (((close[i] - low[i]) - (high[i] - close[i])) / tmp) * volume[i]
        out[i] = ad


def AD(high, low, close, volume):
    """
    Chaikin A/D Line
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    v = as_1d_float64(volume)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n or v.shape[0] != n:
        raise ValueError("inputs must have the same length")

    out = nan_like(h, dtype=np.float64)
    _ad_kernel(h, l, c, v, out)
    return out

