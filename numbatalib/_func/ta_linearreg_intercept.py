from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _linearreg_intercept_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    lookback = timeperiod - 1
    sum_x = timeperiod * (timeperiod - 1) * 0.5
    sum_xsqr = timeperiod * (timeperiod - 1) * (2 * timeperiod - 1) / 6.0
    divisor = sum_x * sum_x - timeperiod * sum_xsqr

    for today in range(lookback, n):
        sum_xy = 0.0
        sum_y = 0.0
        for i in range(timeperiod - 1, -1, -1):
            temp = real[today - i]
            sum_y += temp
            sum_xy += float(i) * temp
        m = (timeperiod * sum_xy - sum_x * sum_y) / divisor
        b = (sum_y - m * sum_x) / timeperiod
        out[today] = b


def LINEARREG_INTERCEPT(real, timeperiod: int = 14):
    """
    Linear Regression Intercept
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    out = nan_like(real_arr, dtype=np.float64)
    _linearreg_intercept_kernel(real_arr, tp, out)
    return out

