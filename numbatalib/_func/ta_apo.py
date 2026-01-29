from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ma import MA, _validate_matype


def APO(real, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0):
    """
    Absolute Price Oscillator
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
    out[:] = fast_ma - slow_ma
    return out

