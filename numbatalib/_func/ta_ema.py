from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _ema_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    k = 2.0 / (timeperiod + 1.0)

    # Seed: simple MA of first `timeperiod` values (TA_COMPATIBILITY_DEFAULT).
    s = 0.0
    for i in range(timeperiod):
        s += real[i]
    prev = s / timeperiod

    out_idx = timeperiod - 1
    out[out_idx] = prev

    for i in range(timeperiod, n):
        prev = ((real[i] - prev) * k) + prev
        out[i] = prev


def EMA(real, timeperiod: int = 30):
    """
    Exponential Moving Average

    Mirrors TA-Lib default compatibility behavior (classic seed).

    Notes:
      - TA-Lib's unstable period support is not wired yet; current implementation
        matches the default (unstable period = 0).
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _ema_kernel(real_arr, tp, out)
    return out

