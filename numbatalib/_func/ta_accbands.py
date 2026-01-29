from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, validate_int_param
from numbatalib._func.ta_sma import SMA

TA_EPSILON = 1e-14


@njit(cache=True)
def _accbands_transform_kernel(high: np.ndarray, low: np.ndarray, out_high: np.ndarray, out_low: np.ndarray) -> None:
    n = high.shape[0]
    for i in range(n):
        tmp = high[i] + low[i]
        if abs(tmp) >= TA_EPSILON:
            t = 4.0 * (high[i] - low[i]) / tmp
            out_high[i] = high[i] * (1.0 + t)
            out_low[i] = low[i] * (1.0 - t)
        else:
            out_high[i] = high[i]
            out_low[i] = low[i]


def ACCBANDS(high, low, close, timeperiod: int = 20):
    """
    Acceleration Bands
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    t_high = np.empty(n, dtype=np.float64)
    t_low = np.empty(n, dtype=np.float64)
    _accbands_transform_kernel(h, l, t_high, t_low)

    middle = SMA(c, timeperiod=tp)
    upper = SMA(t_high, timeperiod=tp)
    lower = SMA(t_low, timeperiod=tp)
    return upper, middle, lower
