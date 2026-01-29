from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ema import EMA


def TEMA(real, timeperiod: int = 30):
    """
    Triple Exponential Moving Average
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    n = real_arr.shape[0]

    lb = tp - 1
    lookback = 3 * lb
    out = nan_like(real_arr, dtype=np.float64)
    if n <= lookback:
        return out

    ema1 = EMA(real_arr, timeperiod=tp)
    ema1_valid = np.ascontiguousarray(ema1[lb:])

    ema2_full = EMA(ema1_valid, timeperiod=tp)
    ema2_valid = np.ascontiguousarray(ema2_full[lb:])

    ema3_full = EMA(ema2_valid, timeperiod=tp)
    ema3_valid = np.ascontiguousarray(ema3_full[lb:])

    ema1_for = ema1_valid[2 * lb :]
    ema2_for = ema2_valid[lb:]
    tema_valid = (3.0 * ema1_for) - (3.0 * ema2_for) + ema3_valid

    out[lookback:] = tema_valid
    return out

