from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_DOJI,
    NEAR,
    SHADOW_LONG,
    candle_average,
    candle_range,
    high_low_range,
    lower_shadow,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdlrickshawman_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 10  # max(max(BodyDoji, ShadowLong), Near)
    if n <= lookback_total:
        return

    body_doji_total = 0.0
    shadow_long_total = 0.0
    near_total = 0.0
    body_trailing = 0
    shadow_trailing = 0
    near_trailing = 0

    i = 0
    while i < lookback_total:
        body_doji_total += candle_range(BODY_DOJI, open_, high, low, close, i)
        shadow_long_total += candle_range(SHADOW_LONG, open_, high, low, close, i)
        near_total += candle_range(NEAR, open_, high, low, close, i)
        i += 1

    for i in range(lookback_total, n):
        is_doji = real_body(open_, close, i) <= candle_average(
            BODY_DOJI, body_doji_total, open_, high, low, close, i
        )
        sh_long_avg = candle_average(SHADOW_LONG, shadow_long_total, open_, high, low, close, i)
        near_avg = candle_average(NEAR, near_total, open_, high, low, close, i)

        oc_min = open_[i] if open_[i] < close[i] else close[i]
        oc_max = open_[i] if open_[i] > close[i] else close[i]
        midpoint = low[i] + high_low_range(high, low, i) * 0.5

        if (
            is_doji
            and lower_shadow(open_, low, close, i) > sh_long_avg
            and upper_shadow(open_, high, close, i) > sh_long_avg
            and (oc_min <= midpoint + near_avg and oc_max >= midpoint - near_avg)
        ):
            out[i] = 100
        else:
            out[i] = 0

        body_doji_total += candle_range(BODY_DOJI, open_, high, low, close, i) - candle_range(
            BODY_DOJI, open_, high, low, close, body_trailing
        )
        shadow_long_total += candle_range(
            SHADOW_LONG, open_, high, low, close, i
        ) - candle_range(SHADOW_LONG, open_, high, low, close, shadow_trailing)
        near_total += candle_range(NEAR, open_, high, low, close, i) - candle_range(
            NEAR, open_, high, low, close, near_trailing
        )

        body_trailing += 1
        shadow_trailing += 1
        near_trailing += 1


def CDLRICKSHAWMAN(open, high, low, close):
    """
    Rickshaw Man

    Output is an int array with values in {0, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlrickshawman_kernel(o, h, l, c, out)
    return out

