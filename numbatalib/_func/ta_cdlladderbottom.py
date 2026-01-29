from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import SHADOW_VERY_SHORT, candle_average, candle_color, candle_range, upper_shadow


@njit(cache=True)
def _cdlladderbottom_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 14  # avgPeriod(ShadowVeryShort) + 4
    if n <= lookback_total:
        return

    start_idx = lookback_total
    svs_total = 0.0
    svs_trailing = start_idx - 10  # avgPeriod(ShadowVeryShort)=10

    i = svs_trailing
    while i < start_idx:
        svs_total += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1)
        i += 1

    for i in range(start_idx, n):
        if (
            candle_color(open_, close, i - 4) == -1
            and candle_color(open_, close, i - 3) == -1
            and candle_color(open_, close, i - 2) == -1
            and open_[i - 4] > open_[i - 3]
            and open_[i - 3] > open_[i - 2]
            and close[i - 4] > close[i - 3]
            and close[i - 3] > close[i - 2]
            and candle_color(open_, close, i - 1) == -1
            and upper_shadow(open_, high, close, i - 1)
            > candle_average(SHADOW_VERY_SHORT, svs_total, open_, high, low, close, i - 1)
            and candle_color(open_, close, i) == 1
            and open_[i] > open_[i - 1]
            and close[i] > high[i - 1]
        ):
            out[i] = 100
        else:
            out[i] = 0

        svs_total += candle_range(SHADOW_VERY_SHORT, open_, high, low, close, i - 1) - candle_range(
            SHADOW_VERY_SHORT, open_, high, low, close, svs_trailing - 1
        )
        svs_trailing += 1


def CDLLADDERBOTTOM(open, high, low, close):
    """
    Ladder Bottom

    Output is an int array with values in {0, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlladderbottom_kernel(o, h, l, c, out)
    return out

