from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlspinningtop_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 10  # TA_CANDLEAVGPERIOD(BodyShort)
    if n <= lookback_total:
        return

    body_total = 0.0
    trailing = 0
    i = 0
    while i < lookback_total:
        body_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        i += 1

    for i in range(lookback_total, n):
        rb = real_body(open_, close, i)
        if (
            rb < candle_average(BODY_SHORT, body_total, open_, high, low, close, i)
            and upper_shadow(open_, high, close, i) > rb
            and lower_shadow(open_, low, close, i) > rb
        ):
            out[i] = candle_color(open_, close, i) * 100
        else:
            out[i] = 0

        body_total += candle_range(BODY_SHORT, open_, high, low, close, i) - candle_range(
            BODY_SHORT, open_, high, low, close, trailing
        )
        trailing += 1


def CDLSPINNINGTOP(open, high, low, close):
    """
    Spinning Top

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlspinningtop_kernel(o, h, l, c, out)
    return out

