from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _minus_dm_kernel(high: np.ndarray, low: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = high.shape[0]
    if n == 0:
        return

    if timeperiod > 1:
        lookback_total = timeperiod - 1
    else:
        lookback_total = 1

    if lookback_total >= n:
        return

    if timeperiod <= 1:
        prev_high = high[lookback_total - 1]
        prev_low = low[lookback_total - 1]
        for today in range(lookback_total, n):
            diff_p = high[today] - prev_high
            diff_m = prev_low - low[today]
            prev_high = high[today]
            prev_low = low[today]
            if (diff_m > 0.0) and (diff_p < diff_m):
                out[today] = diff_m
            else:
                out[today] = 0.0
        return

    prev_minus_dm = 0.0
    today = 0
    prev_high = high[today]
    prev_low = low[today]
    for _ in range(timeperiod - 1):
        today += 1
        diff_p = high[today] - prev_high
        diff_m = prev_low - low[today]
        prev_high = high[today]
        prev_low = low[today]
        if (diff_m > 0.0) and (diff_p < diff_m):
            prev_minus_dm += diff_m

    out[lookback_total] = prev_minus_dm

    for today in range(lookback_total + 1, n):
        diff_p = high[today] - prev_high
        diff_m = prev_low - low[today]
        prev_high = high[today]
        prev_low = low[today]

        prev_minus_dm -= prev_minus_dm / timeperiod
        if (diff_m > 0.0) and (diff_p < diff_m):
            prev_minus_dm += diff_m
        out[today] = prev_minus_dm


def MINUS_DM(high, low, timeperiod: int = 14):
    """
    Minus Directional Movement
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    n = h.shape[0]
    if l.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))
    out = nan_like(h, dtype=np.float64)
    _minus_dm_kernel(h, l, tp, out)
    return out

