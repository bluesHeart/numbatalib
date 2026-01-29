from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_avgdev import AVGDEV
from numbatalib._func.ta_sma import SMA


TA_EPSILON = 1e-14


@njit(cache=True)
def _cci_kernel(tp: np.ndarray, ma: np.ndarray, dev: np.ndarray, out: np.ndarray) -> None:
    n = tp.shape[0]
    for i in range(n):
        d = dev[i]
        m = ma[i]
        if math.isnan(d) or math.isnan(m):
            continue
        denom = 0.015 * d
        if math.fabs(denom) < TA_EPSILON:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - m) / denom


def CCI(high, low, close, timeperiod: int = 14):
    """
    Commodity Channel Index
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    typical = (h + l + c) / 3.0
    ma = SMA(typical, timeperiod=tp)
    dev = AVGDEV(typical, timeperiod=tp)

    out = nan_like(typical, dtype=np.float64)
    _cci_kernel(typical, ma, dev, out)
    return out

