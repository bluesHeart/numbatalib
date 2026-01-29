from __future__ import annotations

import numpy as np

from numbatalib._core._validation import (
    Range,
    as_1d_float64,
    validate_float_param,
    validate_int_param,
)
from numbatalib._func.ta_ma import MA
from numbatalib._func.ta_stddev import STDDEV, TA_REAL_MAX, TA_REAL_MIN


def BBANDS(
    real,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
):
    """
    Bollinger Bands
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    up = validate_float_param("nbdevup", nbdevup, Range(min=TA_REAL_MIN, max=TA_REAL_MAX))
    dn = validate_float_param("nbdevdn", nbdevdn, Range(min=TA_REAL_MIN, max=TA_REAL_MAX))

    middle = MA(real_arr, timeperiod=tp, matype=matype)
    std = STDDEV(real_arr, timeperiod=tp, nbdev=1.0)

    upper = middle + (up * std)
    lower = middle - (dn * std)
    return upper, middle, lower

