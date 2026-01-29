from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import BODY_DOJI, candle_average, candle_range, real_body


@njit(cache=True)
def _cdldoji_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 10  # TA_CANDLEAVGPERIOD(BodyDoji)
    if n <= lookback_total:
        return

    body_doji_total = 0.0
    trailing_idx = 0
    i = 0
    while i < lookback_total:
        body_doji_total += candle_range(BODY_DOJI, open_, high, low, close, i)
        i += 1

    for i in range(lookback_total, n):
        if real_body(open_, close, i) <= candle_average(
            BODY_DOJI, body_doji_total, open_, high, low, close, i
        ):
            out[i] = 100
        else:
            out[i] = 0

        body_doji_total += candle_range(BODY_DOJI, open_, high, low, close, i) - candle_range(
            BODY_DOJI, open_, high, low, close, trailing_idx
        )
        trailing_idx += 1


def CDLDOJI(open, high, low, close):
    """
    Doji

    Output is an int array with values in {0, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdldoji_kernel(o, h, l, c, out)
    return out

