from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _sma_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    # Match TA-Lib's TA_INT_SMA accumulation order as closely as possible.
    lookback_total = timeperiod - 1
    start_idx = lookback_total
    end_idx = n - 1

    period_total = 0.0
    trailing_idx = start_idx - lookback_total  # = 0
    i = trailing_idx

    if timeperiod > 1:
        while i < start_idx:
            period_total += real[i]
            i += 1

    out_pos = start_idx
    while True:
        period_total += real[i]
        i += 1
        temp_real = period_total
        period_total -= real[trailing_idx]
        trailing_idx += 1
        out[out_pos] = temp_real / timeperiod
        out_pos += 1
        if i > end_idx:
            break


def SMA(real, timeperiod: int = 30):
    """
    Simple Moving Average

    Mirrors TA-Lib behavior:
      - output length equals input length
      - leading values are NaN (lookback)
      - `timeperiod` must be in [2, 100000] (TA-Lib range)
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _sma_kernel(real_arr, tp, out)
    return out
