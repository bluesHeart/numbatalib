from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_SHORT,
    SHADOW_SHORT,
    candle_average,
    candle_color,
    candle_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlshortline_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 10  # max(BodyShort, ShadowShort)
    if n <= lookback_total:
        return

    body_total = 0.0
    shadow_total = 0.0
    body_trailing = 0
    shadow_trailing = 0

    i = 0
    while i < lookback_total:
        body_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        shadow_total += candle_range(SHADOW_SHORT, open_, high, low, close, i)
        i += 1

    for i in range(lookback_total, n):
        if (
            real_body(open_, close, i) < candle_average(BODY_SHORT, body_total, open_, high, low, close, i)
            and upper_shadow(open_, high, close, i)
            < candle_average(SHADOW_SHORT, shadow_total, open_, high, low, close, i)
            and lower_shadow(open_, low, close, i)
            < candle_average(SHADOW_SHORT, shadow_total, open_, high, low, close, i)
        ):
            out[i] = candle_color(open_, close, i) * 100
        else:
            out[i] = 0

        body_total += candle_range(BODY_SHORT, open_, high, low, close, i) - candle_range(
            BODY_SHORT, open_, high, low, close, body_trailing
        )
        shadow_total += candle_range(SHADOW_SHORT, open_, high, low, close, i) - candle_range(
            SHADOW_SHORT, open_, high, low, close, shadow_trailing
        )
        body_trailing += 1
        shadow_trailing += 1


def CDLSHORTLINE(open, high, low, close):
    """
    Short Line Candle

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlshortline_kernel(o, h, l, c, out)
    return out

