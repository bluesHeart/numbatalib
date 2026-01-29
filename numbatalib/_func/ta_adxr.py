from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_adx import ADX


@njit(cache=True)
def _adxr_kernel(adx: np.ndarray, shift: int, start: int, out: np.ndarray) -> None:
    n = adx.shape[0]
    for i in range(start, n):
        out[i] = (adx[i] + adx[i - shift]) / 2.0


def ADXR(high, low, close, timeperiod: int = 14):
    """
    Average Directional Movement Index Rating
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n:
        raise ValueError("inputs must have the same length")

    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    adx = ADX(h, l, c, timeperiod=tp)
    out = nan_like(h, dtype=np.float64)

    start = (3 * tp) - 2
    if start >= n:
        return out

    shift = tp - 1
    _adxr_kernel(adx, shift, start, out)
    return out

