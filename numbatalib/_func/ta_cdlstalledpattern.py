from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    BODY_SHORT,
    NEAR,
    SHADOW_VERY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlstalledpattern_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # max(BodyLong, BodyShort, ShadowVeryShort, Near) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    bodylong_total2 = 0.0
    bodylong_total1 = 0.0
    bodyshort_total = 0.0
    shadowvs_total = 0.0
    near_total2 = 0.0
    near_total1 = 0.0

    bodylong_trailing = start_idx - 10  # avgPeriod(BodyLong)=10
    bodyshort_trailing = start_idx - 10  # avgPeriod(BodyShort)=10
    shadowvs_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10
    near_trailing = start_idx - 5  # avgPeriod(Near)=5

    i = bodylong_trailing
    while i < start_idx:
        bodylong_total2 += candle_range(BODY_LONG, open_, high, low, close, i - 2)
        bodylong_total1 += candle_range(BODY_LONG, open_, high, low, close, i - 1)
        i += 1

    i = bodyshort_trailing
    while i < start_idx:
        bodyshort_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        i += 1

    i = shadowvs_trailing
    while i < start_idx:
        shadowvs_total += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1)
        i += 1

    i = near_trailing
    while i < start_idx:
        near_total2 += candle_range(NEAR, open_, high, low, close, i - 2)
        near_total1 += candle_range(NEAR, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        if (
            candle_color(open_, close, i - 2) == 1
            and candle_color(open_, close, i - 1) == 1
            and candle_color(open_, close, i) == 1
            and close[i] > close[i - 1]
            and close[i - 1] > close[i - 2]
            and real_body(open_, close, i - 2)
            > candle_average(BODY_LONG, bodylong_total2, open_, high, low, close, i - 2)
            and real_body(open_, close, i - 1)
            > candle_average(BODY_LONG, bodylong_total1, open_, high, low, close, i - 1)
            and upper_shadow(open_, high, close, i - 1)
            < candle_average(SHADOW_VERY_SHORT, shadowvs_total, open_, high, low, close, i - 1)
            and open_[i - 1] > open_[i - 2]
            and open_[i - 1]
            <= close[i - 2] + candle_average(NEAR, near_total2, open_, high, low, close, i - 2)
            and real_body(open_, close, i)
            < candle_average(BODY_SHORT, bodyshort_total, open_, high, low, close, i)
            and open_[i]
            >= close[i - 1]
            - real_body(open_, close, i)
            - candle_average(NEAR, near_total1, open_, high, low, close, i - 1)
        ):
            out[i] = -100
        else:
            out[i] = 0

        bodylong_total2 += candle_range(BODY_LONG, open_, high, low, close, i - 2) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 2
        )
        bodylong_total1 += candle_range(BODY_LONG, open_, high, low, close, i - 1) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 1
        )
        near_total2 += candle_range(NEAR, open_, high, low, close, i - 2) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 2
        )
        near_total1 += candle_range(NEAR, open_, high, low, close, i - 1) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 1
        )
        bodyshort_total += candle_range(BODY_SHORT, open_, high, low, close, i) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing
        )
        shadowvs_total += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, shadowvs_trailing - 1
        )

        bodylong_trailing += 1
        bodyshort_trailing += 1
        shadowvs_trailing += 1
        near_trailing += 1


def CDLSTALLEDPATTERN(open, high, low, close):
    """
    Stalled Pattern

    Output is an int array with values in {0, -100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlstalledpattern_kernel(o, h, l, c, out)
    return out

