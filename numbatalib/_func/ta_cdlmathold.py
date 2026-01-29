from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_float_param
from numbatalib._func._candles import (
    BODY_LONG,
    BODY_SHORT,
    candle_average,
    candle_color,
    candle_range,
    real_body,
    real_body_gap_up,
)


TA_REAL_MAX = 3e37


@njit(cache=True)
def _cdlmathold_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float,
    out: np.ndarray,
) -> None:
    n = open_.shape[0]
    lookback_total = 14  # max(BodyShort, BodyLong) + 4
    if n <= lookback_total:
        return

    start_idx = lookback_total
    bodylong_total4 = 0.0
    bodyshort_total3 = 0.0
    bodyshort_total2 = 0.0
    bodyshort_total1 = 0.0

    bodyshort_trailing = start_idx - 10  # avgPeriod(BodyShort)=10
    bodylong_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = bodyshort_trailing
    while i < start_idx:
        bodyshort_total3 += candle_range(BODY_SHORT, open_, high, low, close, i - 3)
        bodyshort_total2 += candle_range(BODY_SHORT, open_, high, low, close, i - 2)
        bodyshort_total1 += candle_range(BODY_SHORT, open_, high, low, close, i - 1)
        i += 1

    i = bodylong_trailing
    while i < start_idx:
        bodylong_total4 += candle_range(BODY_LONG, open_, high, low, close, i - 4)
        i += 1

    for i in range(start_idx, n):
        i2_min_oc = open_[i - 2] if open_[i - 2] < close[i - 2] else close[i - 2]
        i2_max_oc = open_[i - 2] if open_[i - 2] > close[i - 2] else close[i - 2]
        i1_min_oc = open_[i - 1] if open_[i - 1] < close[i - 1] else close[i - 1]
        i1_max_oc = open_[i - 1] if open_[i - 1] > close[i - 1] else close[i - 1]
        i3_high = high[i - 3]
        i2_high = high[i - 2]
        i1_high = high[i - 1]

        if (
            real_body(open_, close, i - 4)
            > candle_average(BODY_LONG, bodylong_total4, open_, high, low, close, i - 4)
            and real_body(open_, close, i - 3)
            < candle_average(BODY_SHORT, bodyshort_total3, open_, high, low, close, i - 3)
            and real_body(open_, close, i - 2)
            < candle_average(BODY_SHORT, bodyshort_total2, open_, high, low, close, i - 2)
            and real_body(open_, close, i - 1)
            < candle_average(BODY_SHORT, bodyshort_total1, open_, high, low, close, i - 1)
            and candle_color(open_, close, i - 4) == 1
            and candle_color(open_, close, i - 3) == -1
            and candle_color(open_, close, i) == 1
            and real_body_gap_up(open_, close, i - 3, i - 4)
            and i2_min_oc < close[i - 4]
            and i1_min_oc < close[i - 4]
            and i2_min_oc > close[i - 4] - real_body(open_, close, i - 4) * penetration
            and i1_min_oc > close[i - 4] - real_body(open_, close, i - 4) * penetration
            and i2_max_oc < open_[i - 3]
            and i1_max_oc < i2_max_oc
            and open_[i] > close[i - 1]
            and close[i] > (i3_high if i3_high >= i2_high else i2_high if i2_high >= i1_high else i1_high)
        ):
            out[i] = 100
        else:
            out[i] = 0

        bodylong_total4 += candle_range(BODY_LONG, open_, high, low, close, i - 4) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing - 4
        )
        bodyshort_total3 += candle_range(BODY_SHORT, open_, high, low, close, i - 3) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing - 3
        )
        bodyshort_total2 += candle_range(BODY_SHORT, open_, high, low, close, i - 2) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing - 2
        )
        bodyshort_total1 += candle_range(BODY_SHORT, open_, high, low, close, i - 1) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing - 1
        )

        bodyshort_trailing += 1
        bodylong_trailing += 1


def CDLMATHOLD(open, high, low, close, penetration=0.5):
    """
    Mat Hold

    Output is an int array with values in {0, 100}.
    """
    pen = validate_float_param("penetration", penetration, Range(min=0.0, max=TA_REAL_MAX))
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlmathold_kernel(o, h, l, c, pen, out)
    return out

