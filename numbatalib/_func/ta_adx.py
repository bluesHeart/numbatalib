from __future__ import annotations

import math

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func._dmi_shared import _dm_deltas, _ta_is_zero, _true_range


@njit(cache=True)
def _adx_kernel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int, out: np.ndarray
) -> None:
    n = high.shape[0]
    if n == 0:
        return

    lookback_total = (2 * timeperiod) - 1
    if lookback_total >= n:
        return

    prev_minus_dm = 0.0
    prev_plus_dm = 0.0
    prev_tr = 0.0

    today = 0
    prev_high = high[today]
    prev_low = low[today]
    prev_close = close[today]

    # Initial DM and TR (timeperiod-1 bars).
    for _ in range(timeperiod - 1):
        today += 1
        curr_high = high[today]
        curr_low = low[today]
        diff_p, diff_m = _dm_deltas(curr_high, curr_low, prev_high, prev_low)
        prev_high = curr_high
        prev_low = curr_low

        if (diff_m > 0.0) and (diff_p < diff_m):
            prev_minus_dm += diff_m
        elif (diff_p > 0.0) and (diff_p > diff_m):
            prev_plus_dm += diff_p

        tr = _true_range(curr_high, curr_low, prev_close)
        prev_tr += tr
        prev_close = close[today]

    # Sum first `timeperiod` DX values.
    sum_dx = 0.0
    for _ in range(timeperiod):
        today += 1
        curr_high = high[today]
        curr_low = low[today]
        diff_p, diff_m = _dm_deltas(curr_high, curr_low, prev_high, prev_low)
        prev_high = curr_high
        prev_low = curr_low

        prev_minus_dm -= prev_minus_dm / timeperiod
        prev_plus_dm -= prev_plus_dm / timeperiod
        if (diff_m > 0.0) and (diff_p < diff_m):
            prev_minus_dm += diff_m
        elif (diff_p > 0.0) and (diff_p > diff_m):
            prev_plus_dm += diff_p

        tr = _true_range(curr_high, curr_low, prev_close)
        prev_tr = prev_tr - (prev_tr / timeperiod) + tr
        prev_close = close[today]

        if not _ta_is_zero(prev_tr):
            minus_di = 100.0 * (prev_minus_dm / prev_tr)
            plus_di = 100.0 * (prev_plus_dm / prev_tr)
            s = minus_di + plus_di
            if not _ta_is_zero(s):
                sum_dx += 100.0 * (math.fabs(minus_di - plus_di) / s)

    prev_adx = sum_dx / timeperiod
    out[lookback_total] = prev_adx

    # Subsequent ADX.
    for today in range(lookback_total + 1, n):
        curr_high = high[today]
        curr_low = low[today]
        diff_p, diff_m = _dm_deltas(curr_high, curr_low, prev_high, prev_low)
        prev_high = curr_high
        prev_low = curr_low

        prev_minus_dm -= prev_minus_dm / timeperiod
        prev_plus_dm -= prev_plus_dm / timeperiod
        if (diff_m > 0.0) and (diff_p < diff_m):
            prev_minus_dm += diff_m
        elif (diff_p > 0.0) and (diff_p > diff_m):
            prev_plus_dm += diff_p

        tr = _true_range(curr_high, curr_low, prev_close)
        prev_tr = prev_tr - (prev_tr / timeperiod) + tr
        prev_close = close[today]

        if not _ta_is_zero(prev_tr):
            minus_di = 100.0 * (prev_minus_dm / prev_tr)
            plus_di = 100.0 * (prev_plus_dm / prev_tr)
            s = minus_di + plus_di
            if not _ta_is_zero(s):
                dx = 100.0 * (math.fabs(minus_di - plus_di) / s)
                prev_adx = ((prev_adx * (timeperiod - 1)) + dx) / timeperiod

        out[today] = prev_adx


def ADX(high, low, close, timeperiod: int = 14):
    """
    Average Directional Movement Index
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    out = nan_like(h, dtype=np.float64)
    _adx_kernel(h, l, c, tp, out)
    return out

