from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _wma_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    lookback = timeperiod - 1
    start = lookback

    divider = (timeperiod * (timeperiod + 1)) / 2.0

    trailing_idx = 0
    in_idx = trailing_idx

    # Evaluate the initial periodSum/periodSub (except the last bar).
    period_sum = 0.0
    period_sub = 0.0
    weight = 1.0
    while in_idx < start:
        temp = real[in_idx]
        in_idx += 1
        period_sub += temp
        period_sum += temp * weight
        weight += 1.0

    trailing_value = 0.0

    # Tight loop for the remaining range.
    while in_idx < n:
        temp = real[in_idx]
        in_idx += 1

        period_sub += temp
        period_sub -= trailing_value
        period_sum += temp * timeperiod

        trailing_value = real[trailing_idx]
        trailing_idx += 1

        today = in_idx - 1
        out[today] = period_sum / divider

        period_sum -= period_sub


def WMA(real, timeperiod: int = 30):
    """
    Weighted Moving Average
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _wma_kernel(real_arr, tp, out)
    return out

