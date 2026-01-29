from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import candle_color


@njit(cache=True)
def _cdl3outside_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 3
    if n <= lookback_total:
        return

    for i in range(lookback_total, n):
        if (
            candle_color(open_, close, i - 1) == 1
            and candle_color(open_, close, i - 2) == -1
            and close[i - 1] > open_[i - 2]
            and open_[i - 1] < close[i - 2]
            and close[i] > close[i - 1]
        ) or (
            candle_color(open_, close, i - 1) == -1
            and candle_color(open_, close, i - 2) == 1
            and open_[i - 1] > close[i - 2]
            and close[i - 1] < open_[i - 2]
            and close[i] < close[i - 1]
        ):
            out[i] = candle_color(open_, close, i - 1) * 100
        else:
            out[i] = 0


def CDL3OUTSIDE(open, high, low, close):
    """
    Three Outside Up/Down

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdl3outside_kernel(o, h, l, c, out)
    return out

