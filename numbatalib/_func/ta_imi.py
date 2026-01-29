from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _imi_kernel(open_: np.ndarray, close: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = open_.shape[0]
    lookback = timeperiod - 1
    if n <= lookback:
        return

    for today in range(lookback, n):
        upsum = 0.0
        downsum = 0.0
        start = today - lookback
        for i in range(start, today + 1):
            c = close[i]
            o = open_[i]
            if c > o:
                upsum += c - o
            else:
                downsum += o - c
        out[today] = 100.0 * (upsum / (upsum + downsum))


def IMI(open, close, timeperiod: int = 14):
    """
    Intraday Momentum Index
    """
    o = as_1d_float64(open)
    c = as_1d_float64(close)
    if c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    out = nan_like(o, dtype=np.float64)
    _imi_kernel(o, c, tp, out)
    return out

