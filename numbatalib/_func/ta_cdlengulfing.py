from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import candle_color


@njit(cache=True)
def _cdlengulfing_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 2
    if n <= lookback_total:
        return

    for i in range(lookback_total, n):
        c0 = candle_color(open_, close, i)
        c1 = candle_color(open_, close, i - 1)

        bullish = (
            c0 == 1
            and c1 == -1
            and close[i] > open_[i - 1]
            and open_[i] < close[i - 1]
        )
        bearish = (
            c0 == -1
            and c1 == 1
            and open_[i] > close[i - 1]
            and close[i] < open_[i - 1]
        )

        if bullish or bearish:
            out[i] = c0 * 100
        else:
            out[i] = 0


def CDLENGULFING(open, high, low, close):
    """
    Engulfing Pattern

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlengulfing_kernel(o, h, l, c, out)
    return out
