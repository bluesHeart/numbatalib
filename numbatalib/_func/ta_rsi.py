from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


TA_EPSILON = 1e-14


@njit(cache=True)
def _rsi_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    lookback = timeperiod
    if n <= lookback:
        return

    prev_value = real[0]
    prev_gain = 0.0
    prev_loss = 0.0

    for i in range(1, timeperiod + 1):
        v = real[i]
        diff = v - prev_value
        prev_value = v
        if diff < 0.0:
            prev_loss -= diff
        else:
            prev_gain += diff

    prev_loss /= timeperiod
    prev_gain /= timeperiod

    denom = prev_gain + prev_loss
    if math.fabs(denom) >= TA_EPSILON:
        out[timeperiod] = 100.0 * (prev_gain / denom)
    else:
        out[timeperiod] = 0.0

    for i in range(timeperiod + 1, n):
        v = real[i]
        diff = v - prev_value
        prev_value = v

        prev_loss *= timeperiod - 1
        prev_gain *= timeperiod - 1
        if diff < 0.0:
            prev_loss -= diff
        else:
            prev_gain += diff
        prev_loss /= timeperiod
        prev_gain /= timeperiod

        denom = prev_gain + prev_loss
        if math.fabs(denom) >= TA_EPSILON:
            out[i] = 100.0 * (prev_gain / denom)
        else:
            out[i] = 0.0


def RSI(real, timeperiod: int = 14):
    """
    Relative Strength Index
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    out = nan_like(real_arr, dtype=np.float64)
    _rsi_kernel(real_arr, tp, out)
    return out

