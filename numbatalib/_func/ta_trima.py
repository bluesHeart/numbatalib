from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_sma import _sma_kernel


def TRIMA(real, timeperiod: int = 30):
    """
    Triangular Moving Average
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    n = real_arr.shape[0]
    out = nan_like(real_arr, dtype=np.float64)
    if n == 0 or tp > n:
        return out

    if tp % 2 == 0:
        p1 = tp // 2
        p2 = p1 + 1
    else:
        p1 = (tp + 1) // 2
        p2 = p1

    if p1 == 1:
        first_valid = np.ascontiguousarray(real_arr)
        p1_lb = 0
    else:
        first_full = nan_like(real_arr, dtype=np.float64)
        _sma_kernel(real_arr, p1, first_full)
        p1_lb = p1 - 1
        first_valid = np.ascontiguousarray(first_full[p1_lb:])

    second_full = nan_like(first_valid, dtype=np.float64)
    _sma_kernel(first_valid, p2, second_full)
    p2_lb = p2 - 1
    second_valid = second_full[p2_lb:]

    out[(p1_lb + p2_lb) :] = second_valid
    return out

