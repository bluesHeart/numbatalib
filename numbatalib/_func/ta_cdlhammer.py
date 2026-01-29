from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_SHORT,
    NEAR,
    SHADOW_LONG,
    SHADOW_VERY_SHORT,
    candle_average,
    candle_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlhammer_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 11
    if n <= lookback_total:
        return

    start_idx = lookback_total
    body_total = 0.0
    shadow_long_total = 0.0
    shadow_vs_total = 0.0
    near_total = 0.0

    body_trailing = start_idx - 10  # avgPeriod(BodyShort)=10
    shadow_long_trailing = start_idx  # avgPeriod(ShadowLong)=0
    shadow_vs_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10
    near_trailing = start_idx - 1 - 5  # avgPeriod(Near)=5

    i = body_trailing
    while i < start_idx:
        body_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        i += 1

    i = shadow_long_trailing
    while i < start_idx:
        shadow_long_total += candle_range(SHADOW_LONG, open_, high, low, close, i)
        i += 1

    i = shadow_vs_trailing
    while i < start_idx:
        shadow_vs_total += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i)
        i += 1

    i = near_trailing
    while i < start_idx - 1:
        near_total += candle_range(NEAR, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        oc_min = open_[i] if open_[i] < close[i] else close[i]
        if (
            real_body(open_, close, i)
            < candle_average(BODY_SHORT, body_total, open_, high, low, close, i)
            and lower_shadow(open_, low, close, i)
            > candle_average(SHADOW_LONG, shadow_long_total, open_, high, low, close, i)
            and upper_shadow(open_, high, close, i)
            < candle_average(SHADOW_VERY_SHORT, shadow_vs_total, open_, high, low, close, i)
            and oc_min <= low[i - 1] + candle_average(NEAR, near_total, open_, high, low, close, i - 1)
        ):
            out[i] = 100
        else:
            out[i] = 0

        body_total += candle_range(BODY_SHORT, open_, high, low, close, i) - candle_range(
            BODY_SHORT, open_, high, low, close, body_trailing
        )
        shadow_long_total += candle_range(
            SHADOW_LONG, open_, high, low, close, i
        ) - candle_range(SHADOW_LONG, open_, high, low, close, shadow_long_trailing)
        shadow_vs_total += candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, i
        ) - candle_range(SHADOW_VERY_SHORT, open_, high, low, close, shadow_vs_trailing)
        near_total += candle_range(NEAR, open_, high, low, close, i - 1) - candle_range(
            NEAR, open_, high, low, close, near_trailing
        )

        body_trailing += 1
        shadow_long_trailing += 1
        shadow_vs_trailing += 1
        near_trailing += 1


def CDLHAMMER(open, high, low, close):
    """
    Hammer

    Output is an int array with values in {0, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlhammer_kernel(o, h, l, c, out)
    return out

