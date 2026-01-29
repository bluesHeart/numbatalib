from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _mfi_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    timeperiod: int,
    pos_buf: np.ndarray,
    neg_buf: np.ndarray,
    out: np.ndarray,
) -> None:
    n = high.shape[0]
    lookback = timeperiod
    if n <= lookback:
        return

    prev_typ = (high[0] + low[0] + close[0]) / 3.0
    pos_sum = 0.0
    neg_sum = 0.0
    buf_idx = 0

    day = 1
    for _ in range(timeperiod):
        typ = (high[day] + low[day] + close[day]) / 3.0
        diff = typ - prev_typ
        prev_typ = typ
        mf = typ * volume[day]
        if diff < 0.0:
            neg_buf[buf_idx] = mf
            pos_buf[buf_idx] = 0.0
            neg_sum += mf
        elif diff > 0.0:
            pos_buf[buf_idx] = mf
            neg_buf[buf_idx] = 0.0
            pos_sum += mf
        else:
            pos_buf[buf_idx] = 0.0
            neg_buf[buf_idx] = 0.0
        buf_idx += 1
        if buf_idx == timeperiod:
            buf_idx = 0
        day += 1

    total = pos_sum + neg_sum
    if total < 1.0:
        out[timeperiod] = 0.0
    else:
        out[timeperiod] = 100.0 * (pos_sum / total)

    for day in range(timeperiod + 1, n):
        pos_sum -= pos_buf[buf_idx]
        neg_sum -= neg_buf[buf_idx]

        typ = (high[day] + low[day] + close[day]) / 3.0
        diff = typ - prev_typ
        prev_typ = typ
        mf = typ * volume[day]
        if diff < 0.0:
            neg_buf[buf_idx] = mf
            pos_buf[buf_idx] = 0.0
            neg_sum += mf
        elif diff > 0.0:
            pos_buf[buf_idx] = mf
            neg_buf[buf_idx] = 0.0
            pos_sum += mf
        else:
            pos_buf[buf_idx] = 0.0
            neg_buf[buf_idx] = 0.0

        buf_idx += 1
        if buf_idx == timeperiod:
            buf_idx = 0

        total = pos_sum + neg_sum
        if total < 1.0:
            out[day] = 0.0
        else:
            out[day] = 100.0 * (pos_sum / total)


def MFI(high, low, close, volume, timeperiod: int = 14):
    """
    Money Flow Index
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    v = as_1d_float64(volume)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n or v.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(h, dtype=np.float64)
    pos_buf = np.zeros(tp, dtype=np.float64)
    neg_buf = np.zeros(tp, dtype=np.float64)
    _mfi_kernel(h, l, c, v, tp, pos_buf, neg_buf, out)
    return out

