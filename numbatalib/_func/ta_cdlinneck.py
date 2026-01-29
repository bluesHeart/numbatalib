from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import BODY_LONG, EQUAL, candle_average, candle_color, candle_range, real_body


@njit(cache=True)
def _cdlinneck_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 11  # max(Equal, BodyLong) + 1
    if n <= lookback_total:
        return

    start_idx = lookback_total
    equal_total = 0.0
    body_long_total = 0.0
    equal_trailing = start_idx - 5  # avgPeriod(Equal)=5
    body_long_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = equal_trailing
    while i < start_idx:
        equal_total += candle_range(EQUAL, open_, high, low, close, i - 1)
        i += 1

    i = body_long_trailing
    while i < start_idx:
        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        if (
            candle_color(open_, close, i - 1) == -1
            and real_body(open_, close, i - 1)
            > candle_average(BODY_LONG, body_long_total, open_, high, low, close, i - 1)
            and candle_color(open_, close, i) == 1
            and open_[i] < low[i - 1]
            and close[i]
            <= close[i - 1]
            + candle_average(EQUAL, equal_total, open_, high, low, close, i - 1)
            and close[i] >= close[i - 1]
        ):
            out[i] = -100
        else:
            out[i] = 0

        equal_total += candle_range(EQUAL, open_, high, low, close, i - 1) - candle_range(
            EQUAL, open_, high, low, close, equal_trailing - 1
        )
        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i - 1) - candle_range(
            BODY_LONG, open_, high, low, close, body_long_trailing - 1
        )
        equal_trailing += 1
        body_long_trailing += 1


def CDLINNECK(open, high, low, close):
    """
    In-Neck Pattern

    Output is an int array with values in {0, -100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlinneck_kernel(o, h, l, c, out)
    return out

