from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


TA_EPSILON = 1e-14


@njit(cache=True)
def _correl_kernel(real0: np.ndarray, real1: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real0.shape[0]
    if timeperiod > n:
        return
    lookback = timeperiod - 1

    # Initial window.
    sum_xy = 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    for i in range(timeperiod):
        x = real0[i]
        y = real1[i]
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y

    trailing_idx = 0
    today = lookback
    n_f = float(timeperiod)

    trailing_x = real0[trailing_idx]
    trailing_y = real1[trailing_idx]
    trailing_idx += 1
    temp = (sum_x2 - ((sum_x * sum_x) / n_f)) * (sum_y2 - ((sum_y * sum_y) / n_f))
    if temp > TA_EPSILON:
        out[today] = (sum_xy - ((sum_x * sum_y) / n_f)) / math.sqrt(temp)
    else:
        out[today] = 0.0

    today += 1
    while today < n:
        sum_x -= trailing_x
        sum_x2 -= trailing_x * trailing_x
        sum_xy -= trailing_x * trailing_y
        sum_y -= trailing_y
        sum_y2 -= trailing_y * trailing_y

        x = real0[today]
        y = real1[today]
        sum_x += x
        sum_x2 += x * x
        sum_y += y
        sum_y2 += y * y
        sum_xy += x * y

        trailing_x = real0[trailing_idx]
        trailing_y = real1[trailing_idx]
        trailing_idx += 1

        temp = (sum_x2 - ((sum_x * sum_x) / n_f)) * (sum_y2 - ((sum_y * sum_y) / n_f))
        if temp > TA_EPSILON:
            out[today] = (sum_xy - ((sum_x * sum_y) / n_f)) / math.sqrt(temp)
        else:
            out[today] = 0.0

        today += 1


def CORREL(real0, real1, timeperiod: int = 30):
    """
    Pearson's Correlation Coefficient (r)
    """
    x = as_1d_float64(real0)
    y = as_1d_float64(real1)
    if y.shape[0] != x.shape[0]:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))
    out = nan_like(x, dtype=np.float64)
    _correl_kernel(x, y, tp, out)
    return out

