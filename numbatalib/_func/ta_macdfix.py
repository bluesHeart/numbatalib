from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_macd import _macd_kernel


def MACDFIX(real, signalperiod: int = 9):
    """
    MACD Fix 12/26
    """
    real_arr = as_1d_float64(real)
    sigp = validate_int_param("signalperiod", signalperiod, Range(min=1, max=100000))

    out_macd = nan_like(real_arr, dtype=np.float64)
    out_signal = nan_like(real_arr, dtype=np.float64)
    out_hist = nan_like(real_arr, dtype=np.float64)

    # TA-Lib defines MACDFIX as INT_MACD with (fast=0, slow=0) which triggers
    # fixed k values (12/26 => 0.15/0.075).
    _macd_kernel(real_arr, 0, 0, sigp, out_macd, out_signal, out_hist)
    return out_macd, out_signal, out_hist
