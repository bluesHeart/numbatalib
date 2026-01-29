from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import NEAR, candle_average, candle_color, candle_range, real_body, real_body_gap_down, real_body_gap_up


@njit(cache=True)
def _cdltasukigap_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 7  # TA_CANDLEAVGPERIOD(Near) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    near_total = 0.0
    near_trailing = start_idx - 5  # avgPeriod(Near)=5

    i = near_trailing
    while i < start_idx:
        near_total += candle_range(NEAR, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        near_avg = candle_average(NEAR, near_total, open_, high, low, close, i - 1)
        rb1 = real_body(open_, close, i - 1)
        rb0 = real_body(open_, close, i)

        upside = (
            real_body_gap_up(open_, close, i - 1, i - 2)
            and candle_color(open_, close, i - 1) == 1
            and candle_color(open_, close, i) == -1
            and open_[i] < close[i - 1]
            and open_[i] > open_[i - 1]
            and close[i] < open_[i - 1]
            and close[i] > (close[i - 2] if close[i - 2] > open_[i - 2] else open_[i - 2])
            and abs(rb1 - rb0) < near_avg
        )
        downside = (
            real_body_gap_down(open_, close, i - 1, i - 2)
            and candle_color(open_, close, i - 1) == -1
            and candle_color(open_, close, i) == 1
            and open_[i] < open_[i - 1]
            and open_[i] > close[i - 1]
            and close[i] > open_[i - 1]
            and close[i] < (close[i - 2] if close[i - 2] < open_[i - 2] else open_[i - 2])
            and abs(rb1 - rb0) < near_avg
        )

        if upside or downside:
            out[i] = candle_color(open_, close, i - 1) * 100
        else:
            out[i] = 0

        near_total += candle_range(NEAR, open_, high, low, close, i - 1) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 1
        )
        near_trailing += 1


def CDLTASUKIGAP(open, high, low, close):
    """
    Tasuki Gap

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdltasukigap_kernel(o, h, l, c, out)
    return out

