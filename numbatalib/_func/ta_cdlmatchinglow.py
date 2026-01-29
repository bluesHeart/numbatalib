from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import EQUAL, candle_average, candle_color, candle_range


@njit(cache=True)
def _cdlmatchinglow_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 6  # TA_CANDLEAVGPERIOD(Equal) + 1
    if n <= lookback_total:
        return

    start_idx = lookback_total
    eq_total = 0.0
    trailing = start_idx - 5  # avgPeriod(Equal)=5

    i = trailing
    while i < start_idx:
        eq_total += candle_range(EQUAL, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        eq = candle_average(EQUAL, eq_total, open_, high, low, close, i - 1)
        if (
            candle_color(open_, close, i - 1) == -1
            and candle_color(open_, close, i) == -1
            and close[i] <= close[i - 1] + eq
            and close[i] >= close[i - 1] - eq
        ):
            out[i] = 100
        else:
            out[i] = 0

        eq_total += candle_range(EQUAL, open_, high, low, close, i - 1) - candle_range(
            EQUAL, open_, high, low, close, trailing - 1
        )
        trailing += 1


def CDLMATCHINGLOW(open, high, low, close):
    """
    Matching Low

    Output is an int array with values in {0, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlmatchinglow_kernel(o, h, l, c, out)
    return out

