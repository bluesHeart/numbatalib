from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    EQUAL,
    SHADOW_VERY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    lower_shadow,
)


@njit(cache=True)
def _cdlidentical3crows_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # max(ShadowVeryShort, Equal) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    shadow_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10
    eq_trailing = start_idx - 5  # avgPeriod(Equal)=5

    sh2 = 0.0
    sh1 = 0.0
    sh0 = 0.0
    eq2 = 0.0
    eq1 = 0.0

    i = shadow_trailing
    while i < start_idx:
        sh2 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 2)
        sh1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1)
        sh0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i)
        i += 1

    i = eq_trailing
    while i < start_idx:
        eq2 += candle_range(EQUAL, open_, high, low, close, i - 2)
        eq1 += candle_range(EQUAL, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        eq_2_avg = candle_average(EQUAL, eq2, open_, high, low, close, i - 2)
        eq_1_avg = candle_average(EQUAL, eq1, open_, high, low, close, i - 1)

        if (
            candle_color(open_, close, i - 2) == -1
            and lower_shadow(open_, low, close, i - 2)
            < candle_average(SHADOW_VERY_SHORT, sh2, open_, high, low, close, i - 2)
            and candle_color(open_, close, i - 1) == -1
            and lower_shadow(open_, low, close, i - 1)
            < candle_average(SHADOW_VERY_SHORT, sh1, open_, high, low, close, i - 1)
            and candle_color(open_, close, i) == -1
            and lower_shadow(open_, low, close, i)
            < candle_average(SHADOW_VERY_SHORT, sh0, open_, high, low, close, i)
            and close[i - 2] > close[i - 1]
            and close[i - 1] > close[i]
            and open_[i - 1] <= close[i - 2] + eq_2_avg
            and open_[i - 1] >= close[i - 2] - eq_2_avg
            and open_[i] <= close[i - 1] + eq_1_avg
            and open_[i] >= close[i - 1] - eq_1_avg
        ):
            out[i] = -100
        else:
            out[i] = 0

        sh2 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 2) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_trailing - 2
        )
        sh1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_trailing - 1
        )
        sh0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_trailing
        )
        eq2 += candle_range(EQUAL, open_, high, low, close, i - 2) - candle_range(
            EQUAL, open_, high, low, close, eq_trailing - 2
        )
        eq1 += candle_range(EQUAL, open_, high, low, close, i - 1) - candle_range(
            EQUAL, open_, high, low, close, eq_trailing - 1
        )
        shadow_trailing += 1
        eq_trailing += 1


def CDLIDENTICAL3CROWS(open, high, low, close):
    """
    Identical Three Crows

    Output is an int array with values in {0, -100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlidentical3crows_kernel(o, h, l, c, out)
    return out

