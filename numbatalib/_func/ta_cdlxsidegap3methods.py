from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import candle_color, real_body_gap_down, real_body_gap_up


@njit(cache=True)
def _cdlxsidegap3methods_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 2
    if n <= lookback_total:
        return

    for i in range(lookback_total, n):
        oc1_max = close[i - 1] if close[i - 1] > open_[i - 1] else open_[i - 1]
        oc1_min = open_[i - 1] if close[i - 1] > open_[i - 1] else close[i - 1]
        oc2_max = close[i - 2] if close[i - 2] > open_[i - 2] else open_[i - 2]
        oc2_min = open_[i - 2] if close[i - 2] > open_[i - 2] else close[i - 2]

        if (
            candle_color(open_, close, i - 2) == candle_color(open_, close, i - 1)
            and candle_color(open_, close, i - 1) == -candle_color(open_, close, i)
            and open_[i] < oc1_max
            and open_[i] > oc1_min
            and close[i] < oc2_max
            and close[i] > oc2_min
            and (
                (
                    candle_color(open_, close, i - 2) == 1
                    and real_body_gap_up(open_, close, i - 1, i - 2)
                )
                or (
                    candle_color(open_, close, i - 2) == -1
                    and real_body_gap_down(open_, close, i - 1, i - 2)
                )
            )
        ):
            out[i] = candle_color(open_, close, i - 2) * 100
        else:
            out[i] = 0


def CDLXSIDEGAP3METHODS(open, high, low, close):
    """
    Upside/Downside Gap Three Methods

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlxsidegap3methods_kernel(o, h, l, c, out)
    return out

