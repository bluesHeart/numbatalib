from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _avgdev_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    lookback = timeperiod - 1
    for today in range(lookback, n):
        today_sum = 0.0
        for i in range(timeperiod):
            today_sum += real[today - i]
        mean = today_sum / timeperiod
        today_dev = 0.0
        for i in range(timeperiod):
            today_dev += math.fabs(real[today - i] - mean)
        out[today] = today_dev / timeperiod


def AVGDEV(real, timeperiod: int = 14):
    """
    Average Deviation
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _avgdev_kernel(real_arr, tp, out)
    return out

