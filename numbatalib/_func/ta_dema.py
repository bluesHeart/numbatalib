from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ema import EMA


def DEMA(real, timeperiod: int = 30):
    """
    Double Exponential Moving Average
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    n = real_arr.shape[0]

    lb = tp - 1
    lookback = 2 * lb
    out = nan_like(real_arr, dtype=np.float64)
    if n <= lookback:
        return out

    ema1 = EMA(real_arr, timeperiod=tp)
    ema1_valid = np.ascontiguousarray(ema1[lb:])

    ema2_full = EMA(ema1_valid, timeperiod=tp)
    ema2_valid = np.ascontiguousarray(ema2_full[lb:])

    dema_valid = (2.0 * ema1_valid[lb:]) - ema2_valid
    out[lookback:] = dema_valid
    return out

