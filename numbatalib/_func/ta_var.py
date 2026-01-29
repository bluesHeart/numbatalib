from __future__ import annotations

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
def _var_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
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

        out[i] = mean2 - mean1 * mean1
        i += 1


def VAR(real, timeperiod: int = 5, nbdev: float = 1.0):
    """
    Variance

    Note: `nbdev` is accepted for API parity, but ignored by TA-Lib's VAR implementation.
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))
    _ = validate_float_param("nbdev", nbdev, Range(min=TA_REAL_MIN, max=TA_REAL_MAX))

    out = nan_like(real_arr, dtype=np.float64)
    _var_kernel(real_arr, tp, out)
    return out

