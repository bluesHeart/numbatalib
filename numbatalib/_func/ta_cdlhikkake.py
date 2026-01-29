from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64, nan_like


@njit(cache=True)
def _cdlhikkake_kernel(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, out: np.ndarray
) -> None:
    n = high.shape[0]
    lookback_total = 5
    if n <= lookback_total:
        return

    start_idx = lookback_total
    pattern_idx = 0
    pattern_result = 0

    i = start_idx - 3
    while i < start_idx:
        if high[i - 1] < high[i - 2] and low[i - 1] > low[i - 2] and (
            (high[i] < high[i - 1] and low[i] < low[i - 1])
            or (high[i] > high[i - 1] and low[i] > low[i - 1])
        ):
            pattern_result = 100 * (1 if high[i] < high[i - 1] else -1)
            pattern_idx = i
        else:
            if pattern_idx != 0 and i <= pattern_idx + 3 and (
                (pattern_result > 0 and close[i] > high[pattern_idx - 1])
                or (pattern_result < 0 and close[i] < low[pattern_idx - 1])
            ):
                pattern_idx = 0
        i += 1

    for i in range(start_idx, n):
        if high[i - 1] < high[i - 2] and low[i - 1] > low[i - 2] and (
            (high[i] < high[i - 1] and low[i] < low[i - 1])
            or (high[i] > high[i - 1] and low[i] > low[i - 1])
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


def CDLHIKKAKE(open, high, low, close):
    """
    Hikkake Pattern

    Output is an int array with values in {0, -200, -100, 100, 200}.
    """
    o = as_1d_float64(open)
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    if h.shape[0] != o.shape[0] or l.shape[0] != o.shape[0] or c.shape[0] != o.shape[0]:
        raise ValueError("inputs must have the same length")

    out = nan_like(o, dtype=np.int32)
    _cdlhikkake_kernel(o, h, l, c, out)
    return out

