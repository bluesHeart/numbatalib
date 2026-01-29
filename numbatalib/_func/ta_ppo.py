from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ma import MA, _validate_matype


TA_EPSILON = 1e-14


@njit(cache=True)
def _ppo_kernel(fast_ma: np.ndarray, slow_ma: np.ndarray, out: np.ndarray) -> None:
    n = fast_ma.shape[0]
    for i in range(n):
        s = slow_ma[i]
        if math.isnan(s):
            continue
        if math.fabs(s) < TA_EPSILON:
            out[i] = 0.0
        else:
            out[i] = ((fast_ma[i] - s) / s) * 100.0


def PPO(real, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0):
    """
    Percentage Price Oscillator
    """
    real_arr = as_1d_float64(real)
    fp = validate_int_param("fastperiod", fastperiod, Range(min=2, max=100000))
    sp = validate_int_param("slowperiod", slowperiod, Range(min=2, max=100000))
    mt = _validate_matype(matype)

    if sp < fp:
        fp, sp = sp, fp

    fast_ma = MA(real_arr, timeperiod=fp, matype=mt)
    slow_ma = MA(real_arr, timeperiod=sp, matype=mt)

    out = nan_like(real_arr, dtype=np.float64)
    _ppo_kernel(fast_ma, slow_ma, out)
    return out

