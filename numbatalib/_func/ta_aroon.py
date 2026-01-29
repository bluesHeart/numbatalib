from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _aroon_kernel(high: np.ndarray, low: np.ndarray, timeperiod: int, out_down: np.ndarray, out_up: np.ndarray) -> None:
    n = high.shape[0]
    if n <= timeperiod:
        return

    trailing_idx = 0
    lowest_idx = -1
    highest_idx = -1
    lowest = 0.0
    highest = 0.0
    factor = 100.0 / timeperiod

    for today in range(timeperiod, n):
        # Lowest low
        tmp = low[today]
        if lowest_idx < trailing_idx:
            lowest_idx = trailing_idx
            lowest = low[lowest_idx]
            i = lowest_idx
            while i < today:
                i += 1
                tmp2 = low[i]
                if tmp2 <= lowest:
                    lowest_idx = i
                    lowest = tmp2
        elif tmp <= lowest:
            lowest_idx = today
            lowest = tmp

        # Highest high
        tmp = high[today]
        if highest_idx < trailing_idx:
            highest_idx = trailing_idx
            highest = high[highest_idx]
            i = highest_idx
            while i < today:
                i += 1
                tmp2 = high[i]
                if tmp2 >= highest:
                    highest_idx = i
                    highest = tmp2
        elif tmp >= highest:
            highest_idx = today
            highest = tmp

        out_up[today] = factor * (timeperiod - (today - highest_idx))
        out_down[today] = factor * (timeperiod - (today - lowest_idx))

        trailing_idx += 1


def AROON(high, low, timeperiod: int = 14):
    """
    Aroon
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    if l.shape[0] != h.shape[0]:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    out_down = nan_like(h, dtype=np.float64)
    out_up = nan_like(h, dtype=np.float64)
    _aroon_kernel(h, l, tp, out_down, out_up)
    return out_down, out_up

