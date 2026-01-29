from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_float_param, validate_int_param


@njit(cache=True)
def _t3_kernel(real: np.ndarray, timeperiod: int, vfactor: float, out: np.ndarray) -> None:
    n = real.shape[0]
    if n == 0:
        return

    lookback = 6 * (timeperiod - 1)
    if n <= lookback:
        return

    k = 2.0 / (timeperiod + 1.0)
    one_minus_k = 1.0 - k

    today = 0

    # Initialize e1 with SMA of first period values.
    temp = real[today]
    today += 1
    for _ in range(timeperiod - 1):
        temp += real[today]
        today += 1
    e1 = temp / timeperiod

    # Initialize e2.
    temp = e1
    for _ in range(timeperiod - 1):
        e1 = (k * real[today]) + (one_minus_k * e1)
        today += 1
        temp += e1
    e2 = temp / timeperiod

    # Initialize e3.
    temp = e2
    for _ in range(timeperiod - 1):
        e1 = (k * real[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        temp += e2
    e3 = temp / timeperiod

    # Initialize e4.
    temp = e3
    for _ in range(timeperiod - 1):
        e1 = (k * real[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        temp += e3
    e4 = temp / timeperiod

    # Initialize e5.
    temp = e4
    for _ in range(timeperiod - 1):
        e1 = (k * real[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        temp += e4
    e5 = temp / timeperiod

    # Initialize e6.
    temp = e5
    for _ in range(timeperiod - 1):
        e1 = (k * real[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        temp += e5
    e6 = temp / timeperiod

    # Constants.
    temp2 = vfactor * vfactor
    c1 = -(temp2 * vfactor)
    c2 = 3.0 * (temp2 - c1)
    c3 = (-6.0 * temp2) - 3.0 * (vfactor - c1)
    c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * temp2

    # First output is at index lookback, which corresponds to the last value consumed so far.
    out[lookback] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    idx = lookback + 1
    while idx < n:
        x = real[idx]
        e1 = (k * x) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        e6 = (k * e5) + (one_minus_k * e6)
        out[idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        idx += 1


def T3(real, timeperiod: int = 5, vfactor: float = 0.7):
    """
    T3 Moving Average
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    vf = validate_float_param("vfactor", vfactor, Range(min=0.0, max=1.0))
    out = nan_like(real_arr, dtype=np.float64)
    _t3_kernel(real_arr, tp, vf, out)
    return out

