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
def _cdleveningstar_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float,
    out: np.ndarray,
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # max(BodyShort, BodyLong) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    bodylong_total = 0.0
    bodyshort_total = 0.0
    bodyshort_total2 = 0.0

    bodylong_trailing = start_idx - 2 - 10  # avgPeriod(BodyLong)=10
    bodyshort_trailing = start_idx - 1 - 10  # avgPeriod(BodyShort)=10

    i = bodylong_trailing
    while i < start_idx - 2:
        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    i = bodyshort_trailing
    while i < start_idx - 1:
        bodyshort_total += candle_range(BODY_SHORT, open_, high, low, close, i)
        bodyshort_total2 += candle_range(BODY_SHORT, open_, high, low, close, i + 1)
        i += 1

    for i in range(start_idx, n):
        if (
            real_body(open_, close, i - 2)
            > candle_average(BODY_LONG, bodylong_total, open_, high, low, close, i - 2)
            and candle_color(open_, close, i - 2) == 1
            and real_body(open_, close, i - 1)
            <= candle_average(BODY_SHORT, bodyshort_total, open_, high, low, close, i - 1)
            and real_body_gap_up(open_, close, i - 1, i - 2)
            and real_body(open_, close, i)
            > candle_average(BODY_SHORT, bodyshort_total2, open_, high, low, close, i)
            and candle_color(open_, close, i) == -1
            and close[i] < close[i - 2] - real_body(open_, close, i - 2) * penetration
        ):
            out[i] = -100
        else:
            out[i] = 0

        bodylong_total += candle_range(BODY_LONG, open_, high, low, close, i - 2) - candle_range(
            BODY_LONG, open_, high, low, close, bodylong_trailing
        )
        bodyshort_total += candle_range(BODY_SHORT, open_, high, low, close, i - 1) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing
        )
        bodyshort_total2 += candle_range(BODY_SHORT, open_, high, low, close, i) - candle_range(
            BODY_SHORT, open_, high, low, close, bodyshort_trailing + 1
        )

        bodylong_trailing += 1
        bodyshort_trailing += 1


def CDLEVENINGSTAR(open, high, low, close, penetration=0.3):
    """
    Evening Star

    Output is an int array with values in {0, -100}.
    """
    pen = validate_float_param("penetration", penetration, Range(min=0.0, max=TA_REAL_MAX))
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdleveningstar_kernel(o, h, l, c, pen, out)
    return out

