from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ema import EMA


TA_EPSILON = 1e-14


@njit(cache=True)
def _trix_roc_kernel(series: np.ndarray, offset: int, out: np.ndarray) -> None:
    n = series.shape[0]
    # series is aligned to original index `offset` (series[0] corresponds to original[offset]).
    for i in range(1, n):
        prev = series[i - 1]
        if math.fabs(prev) < TA_EPSILON:
            out[offset + i] = 0.0
        else:
            out[offset + i] = ((series[i] - prev) / prev) * 100.0


def TRIX(real, timeperiod: int = 30):
    """
    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))
    out = nan_like(real_arr, dtype=np.float64)
    n = real_arr.shape[0]
    if n == 0:
        return out
    if tp == 1:
        # 3x EMA(1) == input; TRIX is just ROC1 of input.
        series = real_arr
        if n > 1:
            _trix_roc_kernel(series, 0, out)
        return out

    lb = tp - 1
    lookback = 3 * lb + 1
    if n <= lookback:
        return out

    ema1 = EMA(real_arr, timeperiod=tp)
    ema1_valid = np.ascontiguousarray(ema1[lb:])
    ema2_full = EMA(ema1_valid, timeperiod=tp)
    ema2_valid = np.ascontiguousarray(ema2_full[lb:])
    ema3_full = EMA(ema2_valid, timeperiod=tp)
    ema3_valid = np.ascontiguousarray(ema3_full[lb:])

    # ema3_valid aligns to original index 3*lb.
    _trix_roc_kernel(ema3_valid, 3 * lb, out)
    return out

