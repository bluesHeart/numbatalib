from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    EQUAL,
    SHADOW_VERY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlseparatinglines_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 11
    if n <= lookback_total:
        return

    start_idx = lookback_total
    shadow_total = 0.0
    body_total = 0.0
    eq_total = 0.0

    shadow_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10
    body_trailing = start_idx - 10  # avgPeriod(BodyLong)=10
    eq_trailing = start_idx - 5  # avgPeriod(Equal)=5

    i = shadow_trailing
    while i < start_idx:
        shadow_total += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i)
        i += 1

    i = body_trailing
    while i < start_idx:
        body_total += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    i = eq_trailing
    while i < start_idx:
        eq_total += candle_range(EQUAL, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        eq = candle_average(EQUAL, eq_total, open_, high, low, close, i - 1)
        col0 = candle_color(open_, close, i)
        if (
            candle_color(open_, close, i - 1) == -col0
            and open_[i] <= open_[i - 1] + eq
            and open_[i] >= open_[i - 1] - eq
            and real_body(open_, close, i)
            > candle_average(BODY_LONG, body_total, open_, high, low, close, i)
            and (
                (
                    col0 == 1
                    and lower_shadow(open_, low, close, i)
                    < candle_average(SHADOW_VERY_SHORT, shadow_total, open_, high, low, close, i)
                )
                or (
                    col0 == -1
                    and upper_shadow(open_, high, close, i)
                    < candle_average(SHADOW_VERY_SHORT, shadow_total, open_, high, low, close, i)
                )
            )
        ):
            out[i] = col0 * 100
        else:
            out[i] = 0

        shadow_total += candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, i
        ) - candle_range(SHADOW_VERY_SHORT, open_, high, low, close, shadow_trailing)
        body_total += candle_range(BODY_LONG, open_, high, low, close, i) - candle_range(
            BODY_LONG, open_, high, low, close, body_trailing
        )
        eq_total += candle_range(EQUAL, open_, high, low, close, i - 1) - candle_range(
            EQUAL, open_, high, low, close, eq_trailing - 1
        )
        shadow_trailing += 1
        body_trailing += 1
        eq_trailing += 1


def CDLSEPARATINGLINES(open, high, low, close):
    """
    Separating Lines

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlseparatinglines_kernel(o, h, l, c, out)
    return out

