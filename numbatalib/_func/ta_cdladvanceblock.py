from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_LONG,
    FAR,
    NEAR,
    SHADOW_LONG,
    SHADOW_SHORT,
    candle_average,
    candle_color,
    candle_range,
    real_body,
    upper_shadow,
)


@njit(cache=True)
def _cdladvanceblock_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # max(ShadowLong, ShadowShort, Far, Near, BodyLong) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    shadowshort_total2 = 0.0
    shadowshort_total1 = 0.0
    shadowshort_total0 = 0.0
    shadowlong_total1 = 0.0
    shadowlong_total0 = 0.0
    near_total2 = 0.0
    near_total1 = 0.0
    far_total2 = 0.0
    far_total1 = 0.0
    bodylong_total = 0.0

    shadowshort_trailing = start_idx - 10  # avgPeriod(ShadowShort)=10
    shadowlong_trailing = start_idx  # avgPeriod(ShadowLong)=0
    near_trailing = start_idx - 5  # avgPeriod(Near)=5
    far_trailing = start_idx - 5  # avgPeriod(Far)=5
    bodylong_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = shadowshort_trailing
    while i < start_idx:
        shadowshort_total2 += candle_range(SHADOW_SHORT, open_, high, low, close, i - 2)
        shadowshort_total1 += candle_range(SHADOW_SHORT, open_, high, low, close, i - 1)
        shadowshort_total0 += candle_range(SHADOW_SHORT, open_, high, low, close, i)
        i += 1

    i = shadowlong_trailing
    while i < start_idx:
        shadowlong_total1 += candle_range(SHADOW_LONG, open_, high, low, close, i - 1)
        shadowlong_total0 += candle_range(SHADOW_LONG, open_, high, low, close, i)
        i += 1

    i = near_trailing
    while i < start_idx:
        near_total2 += candle_range(NEAR, open_, high, low, close, i - 2)
        near_total1 += candle_range(NEAR, open_, high, low, close, i - 1)
        i += 1

    i = far_trailing
    while i < start_idx:
        far_total2 += candle_range(FAR, open_, high, low, close, i - 2)
        far_total1 += candle_range(FAR, open_, high, low, close, i - 1)
        i += 1

    i = bodylong_trailing
    while i < start_idx:
        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 2)
        i += 1

    for i in range(start_idx, n):
        if (
            candle_color(open_, close, i - 2) == 1
            and candle_color(open_, close, i - 1) == 1
            and candle_color(open_, close, i) == 1
            and close[i] > close[i - 1]
            and close[i - 1] > close[i - 2]
            and open_[i - 1] > open_[i - 2]
            and open_[i - 1]
            <= close[i - 2] + candle_average(NEAR, near_total2, open_, high, low, close, i - 2)
            and open_[i] > open_[i - 1]
            and open_[i]
            <= close[i - 1] + candle_average(NEAR, near_total1, open_, high, low, close, i - 1)
            and real_body(open_, close, i - 2)
            > candle_average(BODY_LONG, bodylong_total, open_, high, low, close, i - 2)
            and upper_shadow(open_, high, close, i - 2)
            < candle_average(SHADOW_SHORT, shadowshort_total2, open_, high, low, close, i - 2)
            and (
                (
                    real_body(open_, close, i - 1)
                    < real_body(open_, close, i - 2)
                    - candle_average(FAR, far_total2, open_, high, low, close, i - 2)
                    and real_body(open_, close, i)
                    < real_body(open_, close, i - 1)
                    + candle_average(NEAR, near_total1, open_, high, low, close, i - 1)
                )
                or (
                    real_body(open_, close, i)
                    < real_body(open_, close, i - 1)
                    - candle_average(FAR, far_total1, open_, high, low, close, i - 1)
                )
                or (
                    real_body(open_, close, i) < real_body(open_, close, i - 1)
                    and real_body(open_, close, i - 1) < real_body(open_, close, i - 2)
                    and (
                        upper_shadow(open_, high, close, i)
                        > candle_average(SHADOW_SHORT, shadowshort_total0, open_, high, low, close, i)
                        or upper_shadow(open_, high, close, i - 1)
                        > candle_average(SHADOW_SHORT, shadowshort_total1, open_, high, low, close, i - 1)
                    )
                )
                or (
                    real_body(open_, close, i) < real_body(open_, close, i - 1)
                    and upper_shadow(open_, high, close, i)
                    > candle_average(SHADOW_LONG, shadowlong_total0, open_, high, low, close, i)
                )
            )
        ):
            out[i] = -100
        else:
            out[i] = 0

        shadowshort_total2 += candle_range(SHADOW_SHORT, open_, high, low, close, i - 2) - candle_range(
            SHADOW_SHORT, open_, high, low, close, shadowshort_trailing - 2
        )
        shadowshort_total1 += candle_range(SHADOW_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_SHORT, open_, high, low, close, shadowshort_trailing - 1
        )
        shadowshort_total0 += candle_range(SHADOW_SHORT, open_, high, low, close, i) - candle_range(
            SHADOW_SHORT, open_, high, low, close, shadowshort_trailing
        )

        shadowlong_total1 += candle_range(SHADOW_LONG, open_, high, low, close, i - 1) - candle_range(
            SHADOW_LONG, open_, high, low, close, shadowlong_trailing - 1
        )
        shadowlong_total0 += candle_range(SHADOW_LONG, open_, high, low, close, i) - candle_range(
            SHADOW_LONG, open_, high, low, close, shadowlong_trailing
        )

        far_total2 += candle_range(FAR, open_, high, low, close, i - 2) - candle_range(
            FAR, open_, high, low, close, far_trailing - 2
        )
        far_total1 += candle_range(FAR, open_, high, low, close, i - 1) - candle_range(
            FAR, open_, high, low, close, far_trailing - 1
        )

        near_total2 += candle_range(NEAR, open_, high, low, close, i - 2) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 2
        )
        near_total1 += candle_range(NEAR, open_, high, low, close, i - 1) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 1
        )

        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 2) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 2
        )

        shadowshort_trailing += 1
        shadowlong_trailing += 1
        near_trailing += 1
        far_trailing += 1
        bodylong_trailing += 1


def CDLADVANCEBLOCK(open, high, low, close):
    """
    Advance Block

    Output is an int array with values in {0, -100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdladvanceblock_kernel(o, h, l, c, out)
    return out

