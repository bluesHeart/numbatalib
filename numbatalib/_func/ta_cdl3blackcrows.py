from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    SHADOW_VERY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    lower_shadow,
)


@njit(cache=True)
def _cdl3blackcrows_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 13  # TA_CANDLEAVGPERIOD(ShadowVeryShort) + 3
    if n <= lookback_total:
        return

    start_idx = lookback_total
    shadow_vs_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10

    tot2 = 0.0
    tot1 = 0.0
    tot0 = 0.0

    i = shadow_vs_trailing
    while i < start_idx:
        tot2 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 2)
        tot1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1)
        tot0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        if (
            candle_color(open_, close, i - 3) == 1
            and candle_color(open_, close, i - 2) == -1
            and lower_shadow(open_, low, close, i - 2)
            < candle_average(SHADOW_VERY_SHORT, tot2, open_, high, low, close, i - 2)
            and candle_color(open_, close, i - 1) == -1
            and lower_shadow(open_, low, close, i - 1)
            < candle_average(SHADOW_VERY_SHORT, tot1, open_, high, low, close, i - 1)
            and candle_color(open_, close, i) == -1
            and lower_shadow(open_, low, close, i)
            < candle_average(SHADOW_VERY_SHORT, tot0, open_, high, low, close, i)
            and open_[i - 1] < open_[i - 2]
            and open_[i - 1] > close[i - 2]
            and open_[i] < open_[i - 1]
            and open_[i] > close[i - 1]
            and high[i - 3] > close[i - 2]
            and close[i - 2] > close[i - 1]
            and close[i - 1] > close[i]
        ):
            out[i] = -100
        else:
            out[i] = 0

        tot2 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 2) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_vs_trailing - 2
        )
        tot1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_vs_trailing - 1
        )
        tot0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadow_vs_trailing
        )
        shadow_vs_trailing += 1


def CDL3BLACKCROWS(open, high, low, close):
    """
    Three Black Crows

    Output is an int array with values in {0, -100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdl3blackcrows_kernel(o, h, l, c, out)
    return out

