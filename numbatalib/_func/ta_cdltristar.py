from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_DOJI,
    candle_average,
    candle_range,
    real_body,
    real_body_gap_down,
    real_body_gap_up,
)


@njit(cache=True)
def _cdltristar_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 12  # avgPeriod(BodyDoji) + 2
    if n <= lookback_total:
        return

    start_idx = lookback_total
    body_total = 0.0
    body_trailing = start_idx - 2 - 10  # avgPeriod(BodyDoji)=10

    i = body_trailing
    while i < start_idx - 2:
        body_total += candle_range(BODY_DOJI, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        if (
            real_body(open_, close, i - 2)
            <= candle_average(BODY_DOJI, body_total, open_, high, low, close, i - 2)
            and real_body(open_, close, i - 1)
            <= candle_average(BODY_DOJI, body_total, open_, high, low, close, i - 2)
            and real_body(open_, close, i)
            <= candle_average(BODY_DOJI, body_total, open_, high, low, close, i - 2)
        ):
            v = 0
            if real_body_gap_up(open_, close, i - 1, i - 2) and (
                (open_[i] if open_[i] > close[i] else close[i])
                < (open_[i - 1] if open_[i - 1] > close[i - 1] else close[i - 1])
            ):
                v = -100
            if real_body_gap_down(open_, close, i - 1, i - 2) and (
                (open_[i] if open_[i] < close[i] else close[i])
                > (open_[i - 1] if open_[i - 1] < close[i - 1] else close[i - 1])
            ):
                v = 100
            out[i] = v
        else:
            out[i] = 0

        body_total += candle_range(BODY_DOJI, open_, high, low, close, i - 2) - candle_range(
            BODY_DOJI, open_, high, low, close, body_trailing
        )
        body_trailing += 1


def CDLTRISTAR(open, high, low, close):
    """
    Tristar Pattern

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdltristar_kernel(o, h, l, c, out)
    return out

