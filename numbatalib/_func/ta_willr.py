from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_max import _max_kernel
from numbatalib._func.ta_min import _min_kernel


TA_EPSILON = 1e-14


@njit(cache=True)
def _willr_kernel(highest: np.ndarray, lowest: np.ndarray, close: np.ndarray, out: np.ndarray) -> None:
    n = close.shape[0]
    for i in range(n):
        hh = highest[i]
        ll = lowest[i]
        if math.isnan(hh) or math.isnan(ll):
            continue
        rng = hh - ll
        if math.fabs(rng) < TA_EPSILON:
            out[i] = 0.0
        else:
            out[i] = (-100.0) * ((hh - close[i]) / rng)


def WILLR(high, low, close, timeperiod: int = 14):
    """
    Williams' %R
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    highest = nan_like(h, dtype=np.float64)
    lowest = nan_like(h, dtype=np.float64)
    _max_kernel(h, tp, highest)
    _min_kernel(l, tp, lowest)

    out = nan_like(h, dtype=np.float64)
    _willr_kernel(highest, lowest, c, out)
    return out

