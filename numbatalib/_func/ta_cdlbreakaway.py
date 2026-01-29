from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    candle_average,
    candle_color,
    candle_range,
    real_body,
    real_body_gap_down,
    real_body_gap_up,
)


@njit(cache=True)
def _cdlbreakaway_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 14  # avgPeriod(BodyLong) + 4
    if n <= lookback_total:
        return

    start_idx = lookback_total
    bodylong_total = 0.0
    bodylong_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = bodylong_trailing
    while i < start_idx:
        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 4)
        i += 1

    for i in range(start_idx, n):
        c4 = candle_color(open_, close, i - 4)
        if (
            real_body(open_, close, i - 4)
            > candle_average(BODY_LONG, bodylong_total, open_, high, low, close, i - 4)
            and c4 == candle_color(open_, close, i - 3)
            and candle_color(open_, close, i - 3) == candle_color(open_, close, i - 1)
            and candle_color(open_, close, i - 1) == -candle_color(open_, close, i)
            and (
                (
                    c4 == -1
                    and real_body_gap_down(open_, close, i - 3, i - 4)
                    and high[i - 2] < high[i - 3]
                    and low[i - 2] < low[i - 3]
                    and high[i - 1] < high[i - 2]
                    and low[i - 1] < low[i - 2]
                    and close[i] > open_[i - 3]
                    and close[i] < close[i - 4]
                )
                or (
                    c4 == 1
                    and real_body_gap_up(open_, close, i - 3, i - 4)
                    and high[i - 2] > high[i - 3]
                    and low[i - 2] > low[i - 3]
                    and high[i - 1] > high[i - 2]
                    and low[i - 1] > low[i - 2]
                    and close[i] < open_[i - 3]
                    and close[i] > close[i - 4]
                )
            )
        ):
            out[i] = candle_color(open_, close, i) * 100
        else:
            out[i] = 0

        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 4) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 4
        )
        bodylong_trailing += 1


def CDLBREAKAWAY(open, high, low, close):
    """
    Breakaway

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlbreakaway_kernel(o, h, l, c, out)
    return out

