from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _midprice_kernel(high: np.ndarray, low: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = high.shape[0]
    if timeperiod > n:
        return

    nb_initial = timeperiod - 1
    today = nb_initial
    trailing = 0

    while today < n:
        lowest = low[trailing]
        highest = high[trailing]
        i = trailing + 1
        while i <= today:
            tmp = low[i]
            if tmp < lowest:
                lowest = tmp
            tmp = high[i]
            if tmp > highest:
                highest = tmp
            i += 1

        out[today] = (highest + lowest) * 0.5
        trailing += 1
        today += 1


def MIDPRICE(high, low, timeperiod: int = 14):
    """
    Midpoint Price over period.
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    if h.shape[0] != l.shape[0]:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(h, dtype=np.float64)
    _midprice_kernel(h, l, tp, out)
    return out
