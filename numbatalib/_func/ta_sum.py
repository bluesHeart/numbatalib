from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _sum_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    lookback = timeperiod - 1
    trailing = 0

    period_total = 0.0
    i = 0
    while i < lookback:
        period_total += real[i]
        i += 1

    while i < n:
        period_total += real[i]
        out[i] = period_total
        period_total -= real[trailing]
        trailing += 1
        i += 1


def SUM(real, timeperiod: int = 30):
    """
    Summation over a specified period.
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _sum_kernel(real_arr, tp, out)
    return out

