from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    BODY_SHORT,
    SHADOW_LONG,
    SHADOW_VERY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdl3starsinsouth_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # max(ShadowVeryShort, ShadowLong, BodyLong, BodyShort) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    bodylong_total = 0.0
    shadowlong_total = 0.0
    svs_total1 = 0.0
    svs_total0 = 0.0
    bodyshort_total = 0.0

    bodylong_trailing = start_idx - 10  # avgPeriod(BodyLong)=10
    shadowlong_trailing = start_idx  # avgPeriod(ShadowLong)=0
    svs_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10
    bodyshort_trailing = start_idx - 10  # avgPeriod(BodyShort)=10

    i = bodylong_trailing
    while i < start_idx:
        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 2)
        i += 1

    i = shadowlong_trailing
    while i < start_idx:
        shadowlong_total += candle_range(SHADOW_LONG, open_, high, low, close, i - 2)
        i += 1

    i = svs_trailing
    while i < start_idx:
        svs_total1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1)
        svs_total0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i)
        i += 1

    i = bodyshort_trailing
    while i < start_idx:
        bodyshort_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        if (
            candle_color(open_, close, i - 2) == -1
            and candle_color(open_, close, i - 1) == -1
            and candle_color(open_, close, i) == -1
            and real_body(open_, close, i - 2)
            > candle_average(BODY_LONG, bodylong_total, open_, high, low, close, i - 2)
            and lower_shadow(open_, low, close, i - 2)
            > candle_average(SHADOW_LONG, shadowlong_total, open_, high, low, close, i - 2)
            and real_body(open_, close, i - 1) < real_body(open_, close, i - 2)
            and open_[i - 1] > close[i - 2]
            and open_[i - 1] <= high[i - 2]
            and low[i - 1] < close[i - 2]
            and low[i - 1] >= low[i - 2]
            and lower_shadow(open_, low, close, i - 1)
            > candle_average(SHADOW_VERY_SHORT, svs_total1, open_, high, low, close, i - 1)
            and real_body(open_, close, i)
            < candle_average(BODY_SHORT, bodyshort_total, open_, high, low, close, i)
            and lower_shadow(open_, low, close, i)
            < candle_average(SHADOW_VERY_SHORT, svs_total0, open_, high, low, close, i)
            and upper_shadow(open_, high, close, i)
            < candle_average(SHADOW_VERY_SHORT, svs_total0, open_, high, low, close, i)
            and low[i] > low[i - 1]
            and high[i] < high[i - 1]
        ):
            out[i] = 100
        else:
            out[i] = 0

        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 2) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 2
        )
        shadowlong_total += candle_range(
            SHADOW_LONG, open_, high, low, close, i - 2
        ) - candle_range(SHADOW_LONG, open_, high, low, close, shadowlong_trailing - 2)
        svs_total1 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, svs_trailing - 1
        )
        svs_total0 += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, svs_trailing
        )
        bodyshort_total += candle_range(BODY_SHORT, open_, high, low, close, i) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing
        )

        bodylong_trailing += 1
        shadowlong_trailing += 1
        svs_trailing += 1
        bodyshort_trailing += 1


def CDL3STARSINSOUTH(open, high, low, close):
    """
    Three Stars In The South

    Output is an int array with values in {0, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdl3starsinsouth_kernel(o, h, l, c, out)
    return out

