from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like
from numbatalib._func._candles import NEAR, candle_average, candle_range


@njit(cache=True)
def _cdlhikkakemod_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = high.shape[0]
    lookback_total = 10  # max(1, avgPeriod(Near)) + 5
    if n <= lookback_total:
        return

    start_idx = lookback_total
    near_total = 0.0
    near_trailing = start_idx - 3 - 5  # avgPeriod(Near)=5

    i = near_trailing
    while i < start_idx - 3:
        near_total += candle_range(NEAR, open_, high, low, close, i - 2)
        i += 1

    pattern_idx = 0
    pattern_result = 0

    i = start_idx - 3
    while i < start_idx:
        if (
            high[i - 2] < high[i - 3]
            and low[i - 2] > low[i - 3]
            and high[i - 1] < high[i - 2]
            and low[i - 1] > low[i - 2]
            and (
                (
                    high[i] < high[i - 1]
                    and low[i] < low[i - 1]
                    and close[i - 2]
                    <= low[i - 2]
                    + candle_average(NEAR, near_total, open_, high, low, close, i - 2)
                )
                or (
                    high[i] > high[i - 1]
                    and low[i] > low[i - 1]
                    and close[i - 2]
                    >= high[i - 2]
                    - candle_average(NEAR, near_total, open_, high, low, close, i - 2)
                )
            )
        ):
            pattern_result = 100 * (1 if high[i] < high[i - 1] else -1)
            pattern_idx = i
        else:
            if pattern_idx != 0 and i <= pattern_idx + 3 and (
                (pattern_result > 0 and close[i] > high[pattern_idx - 1])
                or (pattern_result < 0 and close[i] < low[pattern_idx - 1])
            ):
                pattern_idx = 0

        near_total += candle_range(NEAR, open_, high, low, close, i - 2) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 2
        )
        near_trailing += 1
        i += 1

    for i in range(start_idx, n):
        if (
            high[i - 2] < high[i - 3]
            and low[i - 2] > low[i - 3]
            and high[i - 1] < high[i - 2]
            and low[i - 1] > low[i - 2]
            and (
                (
                    high[i] < high[i - 1]
                    and low[i] < low[i - 1]
                    and close[i - 2]
                    <= low[i - 2]
                    + candle_average(NEAR, near_total, open_, high, low, close, i - 2)
                )
                or (
                    high[i] > high[i - 1]
                    and low[i] > low[i - 1]
                    and close[i - 2]
                    >= high[i - 2]
                    - candle_average(NEAR, near_total, open_, high, low, close, i - 2)
                )
            )
        ):
            pattern_result = 100 * (1 if high[i] < high[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        else:
            if pattern_idx != 0 and i <= pattern_idx + 3 and (
                (pattern_result > 0 and close[i] > high[pattern_idx - 1])
                or (pattern_result < 0 and close[i] < low[pattern_idx - 1])
            ):
                out[i] = pattern_result + 100 * (1 if pattern_result > 0 else -1)
                pattern_idx = 0
            else:
                out[i] = 0

        near_total += candle_range(NEAR, open_, high, low, close, i - 2) - candle_range(
            NEAR, open_, high, low, close, near_trailing - 2
        )
        near_trailing += 1


def CDLHIKKAKEMOD(open, high, low, close):
    """
    Modified Hikkake Pattern

    Output is an int array with values in {0, -200, -100, 100, 200}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlhikkakemod_kernel(o, h, l, c, out)
    return out

