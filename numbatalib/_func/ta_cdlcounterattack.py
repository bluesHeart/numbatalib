from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import BODY_LONG, EQUAL, candle_average, candle_color, candle_range, real_body


@njit(cache=True)
def _cdlcounterattack_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = open_.shape[0]
    lookback_total = 11  # max(Equal, BodyLong) + 1
    if n <= lookback_total:
        return

    start_idx = lookback_total
    equal_total = 0.0
    eq_trailing = start_idx - 5  # avgPeriod(Equal)=5

    tot1 = 0.0
    tot0 = 0.0
    body_trailing = start_idx - 10  # avgPeriod(BodyLong)=10

    i = eq_trailing
    while i < start_idx:
        equal_total += candle_range(EQUAL, open_, high, low, close, i - 1)
        i += 1

    i = body_trailing
    while i < start_idx:
        tot1 += candle_range(BODY_LONG, open_, high, low, close, i - 1)
        tot0 += candle_range(BODY_LONG, open_, high, low, close, i)
        i += 1

    for i in range(start_idx, n):
        eq = candle_average(EQUAL, equal_total, open_, high, low, close, i - 1)
        if (
            candle_color(open_, close, i - 1) == -candle_color(open_, close, i)
            and real_body(open_, close, i - 1)
            > candle_average(BODY_LONG, tot1, open_, high, low, close, i - 1)
            and real_body(open_, close, i)
            > candle_average(BODY_LONG, tot0, open_, high, low, close, i)
            and close[i] <= close[i - 1] + eq
            and close[i] >= close[i - 1] - eq
        ):
            out[i] = candle_color(open_, close, i) * 100
        else:
            out[i] = 0

        equal_total += candle_range(EQUAL, open_, high, low, close, i - 1) - candle_range(
            EQUAL, open_, high, low, close, eq_trailing - 1
        )
        tot1 += candle_range(BODY_LONG, open_, high, low, close, i - 1) - candle_range(
            BODY_LONG, open_, high, low, close, body_trailing - 1
        )
        tot0 += candle_range(BODY_LONG, open_, high, low, close, i) - candle_range(
            BODY_LONG, open_, high, low, close, body_trailing
        )
        eq_trailing += 1
        body_trailing += 1


def CDLCOUNTERATTACK(open, high, low, close):
    """
    Counterattack

    Output is an int array with values in {0, -100, 100}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlcounterattack_kernel(o, h, l, c, out)
    return out

