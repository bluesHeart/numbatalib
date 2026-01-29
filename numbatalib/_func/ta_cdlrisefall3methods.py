from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    BODY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    real_body,
)


@njit(cache=True)
def _cdlrisefall3methods_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 14  # max(BodyShort, BodyLong) + 4
    if n <= lookback_total:
        return

    start_idx = lookback_total
    bodylong_total4 = 0.0
    bodylong_total0 = 0.0
    bodyshort_total3 = 0.0
    bodyshort_total2 = 0.0
    bodyshort_total1 = 0.0

    bodyshort_trailing = start_idx - 10  # avgPeriod(BodyShort)=10
    bodylong_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = bodyshort_trailing
    while i < start_idx:
        bodyshort_total3 += candle_range(BODY_SHORT, open_, high, low, close, i - 3)
        bodyshort_total2 += candle_range(BODY_SHORT, open_, high, low, close, i - 2)
        bodyshort_total1 += candle_range(BODY_SHORT, open_, high, low, close, i - 1)
        i += 1

    i = bodylong_trailing
    while i < start_idx:
        bodylong_total4 += candle_range(BODY_LONG, open_, high, low, close, i - 4)
        bodylong_total0 += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        c4 = candle_color(open_, close, i - 4)
        if (
            real_body(open_, close, i - 4)
            > candle_average(BODY_LONG, bodylong_total4, open_, high, low, close, i - 4)
            and real_body(open_, close, i - 3)
            < candle_average(BODY_SHORT, bodyshort_total3, open_, high, low, close, i - 3)
            and real_body(open_, close, i - 2)
            < candle_average(BODY_SHORT, bodyshort_total2, open_, high, low, close, i - 2)
            and real_body(open_, close, i - 1)
            < candle_average(BODY_SHORT, bodyshort_total1, open_, high, low, close, i - 1)
            and real_body(open_, close, i)
            > candle_average(BODY_LONG, bodylong_total0, open_, high, low, close, i)
            and c4 == -candle_color(open_, close, i - 3)
            and candle_color(open_, close, i - 3) == candle_color(open_, close, i - 2)
            and candle_color(open_, close, i - 2) == candle_color(open_, close, i - 1)
            and candle_color(open_, close, i - 1) == -candle_color(open_, close, i)
            and (open_[i - 3] if open_[i - 3] < close[i - 3] else close[i - 3]) < high[i - 4]
            and (open_[i - 3] if open_[i - 3] > close[i - 3] else close[i - 3]) > low[i - 4]
            and (open_[i - 2] if open_[i - 2] < close[i - 2] else close[i - 2]) < high[i - 4]
            and (open_[i - 2] if open_[i - 2] > close[i - 2] else close[i - 2]) > low[i - 4]
            and (open_[i - 1] if open_[i - 1] < close[i - 1] else close[i - 1]) < high[i - 4]
            and (open_[i - 1] if open_[i - 1] > close[i - 1] else close[i - 1]) > low[i - 4]
            and close[i - 2] * c4 < close[i - 3] * c4
            and close[i - 1] * c4 < close[i - 2] * c4
            and open_[i] * c4 > close[i - 1] * c4
            and close[i] * c4 > close[i - 4] * c4
        ):
            out[i] = 100 * c4
        else:
            out[i] = 0

        bodylong_total4 += candle_range(BODY_LONG, open_, high, low, close, i - 4) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 4
        )
        bodyshort_total3 += candle_range(BODY_SHORT, open_, high, low, close, i - 3) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing - 3
        )
        bodyshort_total2 += candle_range(BODY_SHORT, open_, high, low, close, i - 2) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing - 2
        )
        bodyshort_total1 += candle_range(BODY_SHORT, open_, high, low, close, i - 1) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing - 1
        )
        bodylong_total0 += candle_range(BODY_LONG, open_, high, low, close, i) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing
        )

        bodyshort_trailing += 1
        bodylong_trailing += 1


def CDLRISEFALL3METHODS(open, high, low, close):
    """
    Rising/Falling Three Methods

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlrisefall3methods_kernel(o, h, l, c, out)
    return out

