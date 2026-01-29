from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import NEAR, candle_average, candle_color, candle_range


@njit(cache=True)
def _cdl3linestrike_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 8  # TA_CANDLEAVGPERIOD(Near) + 3
    if n <= lookback_total:
        return

    start_idx = lookback_total
    near_trailing = start_idx - 5  # avgPeriod(Near)=5
    tot3 = 0.0
    tot2 = 0.0

    i = near_trailing
    while i < start_idx:
        tot3 += candle_range(NEAR, open_, high, low, close, i - 3)
        tot2 += candle_range(NEAR, open_, high, low, close, i - 2)
        i += 1

    for i in range(start_idx, n):
        c1 = candle_color(open_, close, i - 3)
        c2 = candle_color(open_, close, i - 2)
        c3 = candle_color(open_, close, i - 1)
        c4 = candle_color(open_, close, i)

        oc13_min = open_[i - 3] if open_[i - 3] < close[i - 3] else close[i - 3]
        oc13_max = open_[i - 3] if open_[i - 3] > close[i - 3] else close[i - 3]
        oc12_min = open_[i - 2] if open_[i - 2] < close[i - 2] else close[i - 2]
        oc12_max = open_[i - 2] if open_[i - 2] > close[i - 2] else close[i - 2]

        near1 = candle_average(NEAR, tot3, open_, high, low, close, i - 3)
        near2 = candle_average(NEAR, tot2, open_, high, low, close, i - 2)

        if (
            c1 == c2
            and c2 == c3
            and c4 == -c3
            and open_[i - 2] >= oc13_min - near1
            and open_[i - 2] <= oc13_max + near1
            and open_[i - 1] >= oc12_min - near2
            and open_[i - 1] <= oc12_max + near2
            and (
                (
                    c3 == 1
                    and close[i - 1] > close[i - 2]
                    and close[i - 2] > close[i - 3]
                    and open_[i] > close[i - 1]
                    and close[i] < open_[i - 3]
                )
                or (
                    c3 == -1
                    and close[i - 1] < close[i - 2]
                    and close[i - 2] < close[i - 3]
                    and open_[i] < close[i - 1]
                    and close[i] > open_[i - 3]
                )
            )
        ):
            out[i] = c3 * 100
        else:
            out[i] = 0

        tot3 += candle_range(NEAR, open_, high, low, close, i - 3) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 3
        )
        tot2 += candle_range(NEAR, open_, high, low, close, i - 2) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 2
        )
        near_trailing += 1


def CDL3LINESTRIKE(open, high, low, close):
    """
    Three-Line Strike

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdl3linestrike_kernel(o, h, l, c, out)
    return out

