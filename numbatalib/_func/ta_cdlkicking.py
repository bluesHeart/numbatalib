from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    SHADOW_VERY_SHORT,
    candle_average,
    candle_color,
    candle_gap_down,
    candle_gap_up,
    candle_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlkicking_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 11  # max(ShadowVeryShort, BodyLong) + 1
    if n <= lookback_total:
        return

    start_idx = lookback_total
    shadow_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10
    body_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    sh1 = 0.0
    sh0 = 0.0
    bd1 = 0.0
    bd0 = 0.0

    i = shadow_trailing
    while i < start_idx:
        sh1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1)
        sh0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i)
        i += 1

    i = body_trailing
    while i < start_idx:
        bd1 += candle_range(BODY_LONG, open_, high, low, close, i - 1)
        bd0 += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        c1 = candle_color(open_, close, i - 1)
        c0 = candle_color(open_, close, i)

        rb1 = real_body(open_, close, i - 1)
        rb0 = real_body(open_, close, i)

        is_marubozu_1 = (
            rb1 > candle_average(BODY_LONG, bd1, open_, high, low, close, i - 1)
            and upper_shadow(open_, high, close, i - 1)
            < candle_average(SHADOW_VERY_SHORT, sh1, open_, high, low, close, i - 1)
            and lower_shadow(open_, low, close, i - 1)
            < candle_average(SHADOW_VERY_SHORT, sh1, open_, high, low, close, i - 1)
        )
        is_marubozu_0 = (
            rb0 > candle_average(BODY_LONG, bd0, open_, high, low, close, i)
            and upper_shadow(open_, high, close, i)
            < candle_average(SHADOW_VERY_SHORT, sh0, open_, high, low, close, i)
            and lower_shadow(open_, low, close, i)
            < candle_average(SHADOW_VERY_SHORT, sh0, open_, high, low, close, i)
        )

        gap_ok = (c1 == -1 and candle_gap_up(high, low, i, i - 1)) or (
            c1 == 1 and candle_gap_down(high, low, i, i - 1)
        )

        if c1 == -c0 and is_marubozu_1 and is_marubozu_0 and gap_ok:
            out[i] = c0 * 100
        else:
            out[i] = 0

        bd1 += candle_range(BODY_LONG, open_, high, low, close, i - 1) - candle_range(
            BODY_LONG, open_, high, low, close, body_trailing - 1
        )
        bd0 += candle_range(BODY_LONG, open_, high, low, close, i) - candle_range(
            BODY_LONG, open_, high, low, close, body_trailing
        )
        sh1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_trailing - 1
        )
        sh0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_trailing
        )
        body_trailing += 1
        shadow_trailing += 1


def CDLKICKING(open, high, low, close):
    """
    Kicking

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlkicking_kernel(o, h, l, c, out)
    return out

