from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_float_param
from numbatalib._func._candles import BODY_LONG, candle_average, candle_color, candle_range, real_body


TA_REAL_MAX = 3e37


@njit(cache=True)
def _cdldarkcloudcover_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float,
    out: np.ndarray,
) -> None:
    n = open_.shape[0]
    lookback_total = 11  # TA_CANDLEAVGPERIOD(BodyLong) + 1
    if n <= lookback_total:
        return

    start_idx = lookback_total
    body_long_total = 0.0
    body_long_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = body_long_trailing
    while i < start_idx:
        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        rb1 = real_body(open_, close, i - 1)
        if (
            candle_color(open_, close, i - 1) == 1
            and rb1 > candle_average(BODY_LONG, body_long_total, open_, high, low, close, i - 1)
            and candle_color(open_, close, i) == -1
            and open_[i] > high[i - 1]
            and close[i] > open_[i - 1]
            and close[i] < close[i - 1] - rb1 * penetration
        ):
            out[i] = -100
        else:
            out[i] = 0

        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i - 1) - candle_range(
            BODY_LONG, open_, high, low, close, body_long_trailing - 1
        )
        body_long_trailing += 1


def CDLDARKCLOUDCOVER(open, high, low, close, penetration: float = 0.5):
    """
    Dark Cloud Cover

    Output is an int array with values in {0, -100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    pen = validate_float_param("penetration", penetration, Range(min=0.0, max=TA_REAL_MAX))
    out = nan_like(o, dtype=np.int32)
    _cdldarkcloudcover_kernel(o, h, l, c, pen, out)
    return out

