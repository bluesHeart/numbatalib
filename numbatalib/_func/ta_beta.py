from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


TA_EPSILON = 1e-14


@njit(cache=True)
def _beta_kernel(real0: np.ndarray, real1: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real0.shape[0]
    if n == 0 or timeperiod > n - 1:
        return

    start = timeperiod
    if n <= start:
        return

    s_xx = 0.0
    s_xy = 0.0
    s_x = 0.0
    s_y = 0.0

    trailing_idx = 0
    last_x = real0[trailing_idx]
    last_y = real1[trailing_idx]
    trailing_last_x = last_x
    trailing_last_y = last_y

    i = trailing_idx + 1
    while i < start:
        tmp = real0[i]
        if abs(last_x) >= TA_EPSILON:
            x = (tmp - last_x) / last_x
        else:
            x = 0.0
        last_x = tmp

        tmp = real1[i]
        if abs(last_y) >= TA_EPSILON:
            y = (tmp - last_y) / last_y
        else:
            y = 0.0
        last_y = tmp

        s_xx += x * x
        s_xy += x * y
        s_x += x
        s_y += y
        i += 1

    out_idx = start
    n_f = float(timeperiod)
    trailing_idx = 1
    while i < n:
        tmp = real0[i]
        if abs(last_x) >= TA_EPSILON:
            x = (tmp - last_x) / last_x
        else:
            x = 0.0
        last_x = tmp

        tmp = real1[i]
        if abs(last_y) >= TA_EPSILON:
            y = (tmp - last_y) / last_y
        else:
            y = 0.0
        last_y = tmp

        s_xx += x * x
        s_xy += x * y
        s_x += x
        s_y += y

        # trailing returns (read before write).
        tmp = real0[trailing_idx]
        if abs(trailing_last_x) >= TA_EPSILON:
            x_t = (tmp - trailing_last_x) / trailing_last_x
        else:
            x_t = 0.0
        trailing_last_x = tmp

        tmp = real1[trailing_idx]
        if abs(trailing_last_y) >= TA_EPSILON:
            y_t = (tmp - trailing_last_y) / trailing_last_y
        else:
            y_t = 0.0
        trailing_last_y = tmp
        trailing_idx += 1

        denom = (n_f * s_xx) - (s_x * s_x)
        if abs(denom) >= TA_EPSILON:
            out[out_idx] = ((n_f * s_xy) - (s_x * s_y)) / denom
        else:
            out[out_idx] = 0.0
        out_idx += 1

        s_xx -= x_t * x_t
        s_xy -= x_t * y_t
        s_x -= x_t
        s_y -= y_t
        i += 1


def BETA(real0, real1, timeperiod: int = 5):
    """
    Beta
    """
    x = as_1d_float64(real0)
    y = as_1d_float64(real1)
    if y.shape[0] != x.shape[0]:
        raise ValueError("inputs must have the same length")
    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))

    out = nan_like(x, dtype=np.float64)
    _beta_kernel(x, y, tp, out)
    return out
