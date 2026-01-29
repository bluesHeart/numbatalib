from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ma import MA, _ma_lookback, _validate_matype


def MAVP(real, periods, minperiod: int = 2, maxperiod: int = 30, matype: int = 0):
    """
    Moving average with variable period
    """
    real_arr = as_1d_float64(real)
    periods_arr = as_1d_float64(periods)
    n = real_arr.shape[0]
    if periods_arr.shape[0] != n:
        raise ValueError("inputs must have the same length")

    minp = validate_int_param("minperiod", minperiod, Range(min=2, max=100000))
    maxp = validate_int_param("maxperiod", maxperiod, Range(min=2, max=100000))
    mt = _validate_matype(matype)

    out = nan_like(real_arr, dtype=np.float64)
    if n == 0:
        return out

    lookback_total = _ma_lookback(maxp, mt)
    if lookback_total >= n:
        return out

    # Clamp requested periods.
    clamped = periods_arr.astype(np.int64, copy=False)
    clamped = np.clip(clamped, minp, maxp).astype(np.int32, copy=False)

    # Compute each distinct period once and scatter.
    # (The number of distinct periods is typically small.)
    unique_periods = np.unique(clamped[lookback_total:]) if lookback_total < n else np.unique(clamped)
    for p in unique_periods:
        ma = MA(real_arr, timeperiod=int(p), matype=mt)
        mask = clamped == p
        if lookback_total > 0:
            mask[:lookback_total] = False
        out[mask] = ma[mask]
    return out
