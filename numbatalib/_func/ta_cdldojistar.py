from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import (
    BODY_DOJI,
    BODY_LONG,
    candle_average,
    candle_color,
    candle_range,
    real_body,
    real_body_gap_down,
    real_body_gap_up,
)


@njit(cache=True)
def _cdldojistar_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 11  # max(BodyDoji, BodyLong) + 1
    if n <= lookback_total:
        return

    start_idx = lookback_total
    body_long_total = 0.0
    body_doji_total = 0.0

    body_long_trailing = start_idx - 1 - 10  # avgPeriod(BodyLong)=10
    body_doji_trailing = start_idx - 10  # avgPeriod(BodyDoji)=10

    i = body_long_trailing
    while i < start_idx - 1:
        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    i = body_doji_trailing
    while i < start_idx:
        body_doji_total += candle_range(BODY_DOJI, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        if (
            real_body(open_, close, i - 1)
            > candle_average(BODY_LONG, body_long_total, open_, high, low, close, i - 1)
            and real_body(open_, close, i)
            <= candle_average(BODY_DOJI, body_doji_total, open_, high, low, close, i)
            and (
                (
                    candle_color(open_, close, i - 1) == 1
                    and real_body_gap_up(open_, close, i, i - 1)
                )
                or (
                    candle_color(open_, close, i - 1) == -1
                    and real_body_gap_down(open_, close, i, i - 1)
                )
            )
        ):
            out[i] = -candle_color(open_, close, i - 1) * 100
        else:
            out[i] = 0

        body_long_total += candle_range(BODY_LONG, open_, high, low, close, i - 1) - candle_range(
            BODY_LONG, open_, high, low, close, body_long_trailing
        )
        body_doji_total += candle_range(BODY_DOJI, open_, high, low, close, i) - candle_range(
            BODY_DOJI, open_, high, low, close, body_doji_trailing
        )
        body_long_trailing += 1
        body_doji_trailing += 1


def CDLDOJISTAR(open, high, low, close):
    """
    Doji Star

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdldojistar_kernel(o, h, l, c, out)
    return out

