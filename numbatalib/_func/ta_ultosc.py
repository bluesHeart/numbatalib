from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


TA_EPSILON = 1e-14


@njit(cache=True)
def _ultosc_terms(high: np.ndarray, low: np.ndarray, close: np.ndarray, day: int) -> tuple[float, float]:
    lt = low[day]
    ht = high[day]
    cy = close[day - 1] if day > 0 else 0.0
    true_low = lt if lt < cy else cy
    close_minus_true_low = close[day] - true_low

    true_range = ht - lt
    tmp = math.fabs(cy - ht)
    if tmp > true_range:
        true_range = tmp
    tmp = math.fabs(cy - lt)
    if tmp > true_range:
        true_range = tmp
    return close_minus_true_low, true_range


@njit(cache=True)
def _ultosc_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    p1: int,
    p2: int,
    p3: int,
    out: np.ndarray,
) -> None:
    n = high.shape[0]
    start = 0 if p3 == 1 else p3
    if n <= start:
        return

    # Prime running totals (exclude start day).
    a1 = 0.0
    b1 = 0.0
    for i in range(start - p1 + 1, start):
        cm, tr = _ultosc_terms(high, low, close, i)
        a1 += cm
        b1 += tr

    a2 = 0.0
    b2 = 0.0
    for i in range(start - p2 + 1, start):
        cm, tr = _ultosc_terms(high, low, close, i)
        a2 += cm
        b2 += tr

    a3 = 0.0
    b3 = 0.0
    for i in range(start - p3 + 1, start):
        cm, tr = _ultosc_terms(high, low, close, i)
        a3 += cm
        b3 += tr

    trailing1 = start - p1 + 1
    trailing2 = start - p2 + 1
    trailing3 = start - p3 + 1

    for today in range(start, n):
        cm, tr = _ultosc_terms(high, low, close, today)
        a1 += cm
        a2 += cm
        a3 += cm
        b1 += tr
        b2 += tr
        b3 += tr

        output = 0.0
        if math.fabs(b1) >= TA_EPSILON:
            output += 4.0 * (a1 / b1)
        if math.fabs(b2) >= TA_EPSILON:
            output += 2.0 * (a2 / b2)
        if math.fabs(b3) >= TA_EPSILON:
            output += a3 / b3
        out[today] = 100.0 * (output / 7.0)

        cm, tr = _ultosc_terms(high, low, close, trailing1)
        a1 -= cm
        b1 -= tr
        cm, tr = _ultosc_terms(high, low, close, trailing2)
        a2 -= cm
        b2 -= tr
        cm, tr = _ultosc_terms(high, low, close, trailing3)
        a3 -= cm
        b3 -= tr

        trailing1 += 1
        trailing2 += 1
        trailing3 += 1


def ULTOSC(high, low, close, timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28):
    """
    Ultimate Oscillator
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    p1 = validate_int_param("timeperiod1", timeperiod1, Range(min=1, max=100000))
    p2 = validate_int_param("timeperiod2", timeperiod2, Range(min=1, max=100000))
    p3 = validate_int_param("timeperiod3", timeperiod3, Range(min=1, max=100000))

    # TA-Lib sorts the periods so p1 <= p2 <= p3.
    periods = [p1, p2, p3]
    periods.sort()
    p1, p2, p3 = periods[0], periods[1], periods[2]

    out = nan_like(h, dtype=np.float64)
    _ultosc_kernel(h, l, c, p1, p2, p3, out)
    return out
