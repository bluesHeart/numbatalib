from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _natr_kernel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int, out: np.ndarray
) -> None:
    n = high.shape[0]
    if n <= timeperiod:
        return

    # Initial ATR.
    tr_sum = 0.0
    i = 1
    while i <= timeperiod:
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

        tr_sum += greatest
        i += 1

    atr = tr_sum / timeperiod
    out[timeperiod] = atr / close[timeperiod] * 100.0

    # Wilder smoothing.
    i = timeperiod + 1
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

        atr = ((atr * (timeperiod - 1)) + greatest) / timeperiod
        out[i] = atr / close[i] * 100.0
        i += 1


def NATR(high, low, close, timeperiod: int = 14):
    """
    Normalized Average True Range
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))

    out = nan_like(h, dtype=np.float64)
    _natr_kernel(h, l, c, tp, out)
    return out

