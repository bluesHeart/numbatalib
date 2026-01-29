from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


TA_EPSILON = 1e-14


@njit(cache=True)
def _kama_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    lookback = timeperiod
    if n <= lookback:
        return

    const_max = 2.0 / (30.0 + 1.0)
    const_diff = 2.0 / (2.0 + 1.0) - const_max

    sum_roc1 = 0.0
    today = 0
    trailing_idx = 0
    for _ in range(timeperiod):
        temp = real[today] - real[today + 1]
        sum_roc1 += math.fabs(temp)
        today += 1

    prev_kama = real[today - 1]

    temp_real = real[today]
    temp_real2 = real[trailing_idx]
    period_roc = temp_real - temp_real2
    trailing_idx += 1
    trailing_value = temp_real2

    if (sum_roc1 <= period_roc) or (math.fabs(sum_roc1) < TA_EPSILON):
        er = 1.0
    else:
        er = math.fabs(period_roc / sum_roc1)

    sc = (er * const_diff) + const_max
    sc *= sc
    prev_kama = ((real[today] - prev_kama) * sc) + prev_kama
    today += 1

    # Unstable period = 0; compute first output at index `lookback`.
    idx = lookback
    out[idx] = prev_kama

    while today < n:
        temp_real = real[today]
        temp_real2 = real[trailing_idx]
        period_roc = temp_real - temp_real2
        trailing_idx += 1

        sum_roc1 -= math.fabs(trailing_value - temp_real2)
        sum_roc1 += math.fabs(temp_real - real[today - 1])
        trailing_value = temp_real2

        if (sum_roc1 <= period_roc) or (math.fabs(sum_roc1) < TA_EPSILON):
            er = 1.0
        else:
            er = math.fabs(period_roc / sum_roc1)

        sc = (er * const_diff) + const_max
        sc *= sc

        prev_kama = ((temp_real - prev_kama) * sc) + prev_kama
        out[today] = prev_kama
        today += 1


def KAMA(real, timeperiod: int = 30):
    """
    Kaufman Adaptive Moving Average
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    out = nan_like(real_arr, dtype=np.float64)
    _kama_kernel(real_arr, tp, out)
    return out

