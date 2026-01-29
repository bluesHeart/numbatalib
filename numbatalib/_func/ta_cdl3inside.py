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
def _cdl3inside_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # max(BodyShort, BodyLong) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    body_long_total = 0.0
    body_short_total = 0.0
    body_long_trailing = start_idx - 2 - 10  # avgPeriod(BodyLong)=10
    body_short_trailing = start_idx - 1 - 10  # avgPeriod(BodyShort)=10

    i = body_long_trailing
    while i < start_idx - 2:
        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    i = body_short_trailing
    while i < start_idx - 1:
        body_short_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        oc1_max = close[i - 1] if close[i - 1] > open_[i - 1] else open_[i - 1]
        oc1_min = open_[i - 1] if close[i - 1] > open_[i - 1] else close[i - 1]
        oc2_max = close[i - 2] if close[i - 2] > open_[i - 2] else open_[i - 2]
        oc2_min = open_[i - 2] if close[i - 2] > open_[i - 2] else close[i - 2]

        if (
            real_body(open_, close, i - 2)
            > candle_average(BODY_LONG, body_long_total, open_, high, low, close, i - 2)
            and real_body(open_, close, i - 1)
            <= candle_average(BODY_SHORT, body_short_total, open_, high, low, close, i - 1)
            and oc1_max < oc2_max
            and oc1_min > oc2_min
            and (
                (
                    candle_color(open_, close, i - 2) == 1
                    and candle_color(open_, close, i) == -1
                    and close[i] < open_[i - 2]
                )
                or (
                    candle_color(open_, close, i - 2) == -1
                    and candle_color(open_, close, i) == 1
                    and close[i] > open_[i - 2]
                )
            )
        ):
            out[i] = -candle_color(open_, close, i - 2) * 100
        else:
            out[i] = 0

        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i - 2) - candle_range(
            BODY_LONG, open_, high, low, close, body_long_trailing
        )
        body_short_total += candle_range(
            BODY_SHORT, open_, high, low, close, i - 1
        ) - candle_range(BODY_SHORT, open_, high, low, close, body_short_trailing)
        body_long_trailing += 1
        body_short_trailing += 1


def CDL3INSIDE(open, high, low, close):
    """
    Three Inside Up/Down

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdl3inside_kernel(o, h, l, c, out)
    return out

