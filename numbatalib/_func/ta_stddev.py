from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import (
    Range,
    as_1d_float64,
    nan_like,
    validate_float_param,
    validate_int_param,
)


TA_REAL_MIN = -3e37
TA_REAL_MAX = 3e37


@njit(cache=True)
def _stddev_kernel(real: np.ndarray, timeperiod: int, nbdev: float, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    lookback = timeperiod - 1
    trailing = 0

    sum1 = 0.0
    sum2 = 0.0

    i = 0
    while i < lookback:
        temp = real[i]
        sum1 += temp
        sum2 += temp * temp
        i += 1

    while i < n:
        temp = real[i]
        sum1 += temp
        sum2 += temp * temp

        mean1 = sum1 / timeperiod
        mean2 = sum2 / timeperiod

        temp = real[trailing]
        sum1 -= temp
        sum2 -= temp * temp
        trailing += 1

        var = mean2 - mean1 * mean1
        if var > 0.0:
            if nbdev != 1.0:
                out[i] = math.sqrt(var) * nbdev
            else:
                out[i] = math.sqrt(var)
        else:
            out[i] = 0.0
        i += 1


def STDDEV(real, timeperiod: int = 5, nbdev: float = 1.0):
    """
    Standard Deviation
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    nb = validate_float_param("nbdev", nbdev, Range(min=TA_REAL_MIN, max=TA_REAL_MAX))

    out = nan_like(real_arr, dtype=np.float64)
    _stddev_kernel(real_arr, tp, nb, out)
    return out

