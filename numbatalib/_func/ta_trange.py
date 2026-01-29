from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like


@njit(cache=True)
def _trange_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray) -> None:
    n = high.shape[0]
    if n <= 1:
        return

    i = 1
    while i < n:
        temp_lt = low[i]
        temp_ht = high[i]
        temp_cy = close[i - 1]

        greatest = temp_ht - temp_lt

        val2 = abs(temp_cy - temp_ht)
        if val2 > greatest:
            greatest = val2

        val3 = abs(temp_cy - temp_lt)
        if val3 > greatest:
            greatest = val3

        out[i] = greatest
        i += 1


def TRANGE(high, low, close):
    """
    True Range
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    out = nan_like(h, dtype=np.float64)
    _trange_kernel(h, l, c, out)
    return out

