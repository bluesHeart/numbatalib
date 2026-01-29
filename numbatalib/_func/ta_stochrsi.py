from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ma import MA, _ma_lookback, _validate_matype
from numbatalib._func.ta_max import _max_kernel
from numbatalib._func.ta_min import _min_kernel
from numbatalib._func.ta_rsi import RSI


TA_EPSILON = 1e-14


@njit(cache=True)
def _stoch_k_kernel(close: np.ndarray, highest: np.ndarray, lowest: np.ndarray, out: np.ndarray) -> None:
    n = close.shape[0]
    for i in range(n):
        hh = highest[i]
        ll = lowest[i]
        if math.isnan(hh) or math.isnan(ll):
            continue
        denom = hh - ll
        if math.fabs(denom) < TA_EPSILON:
            out[i] = 0.0
        else:
            out[i] = 100.0 * ((close[i] - ll) / denom)


def STOCHRSI(
    real,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
):
    """
    Stochastic Relative Strength Index
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))
    fastk = validate_int_param("fastk_period", fastk_period, Range(min=1, max=100000))
    fastd = validate_int_param("fastd_period", fastd_period, Range(min=1, max=100000))
    fd_mt = _validate_matype(fastd_matype)

    rsi_full = RSI(real_arr, timeperiod=tp)
    rsi_lb = tp

    out_k = nan_like(real_arr, dtype=np.float64)
    out_d = nan_like(real_arr, dtype=np.float64)
    n = real_arr.shape[0]
    if rsi_lb >= n:
        return out_k, out_d

    rsi_valid = np.ascontiguousarray(rsi_full[rsi_lb:])

    highest = nan_like(rsi_valid, dtype=np.float64)
    lowest = nan_like(rsi_valid, dtype=np.float64)
    _max_kernel(rsi_valid, fastk, highest)
    _min_kernel(rsi_valid, fastk, lowest)

    fastk_raw = nan_like(rsi_valid, dtype=np.float64)
    _stoch_k_kernel(rsi_valid, highest, lowest, fastk_raw)

    fastk_lb = fastk - 1
    fastd_lb = _ma_lookback(fastd, fd_mt)
    total_lb = rsi_lb + fastk_lb + fastd_lb
    if total_lb >= n:
        return out_k, out_d

    fastk_valid = np.ascontiguousarray(fastk_raw[fastk_lb:])
    fastd_full = MA(fastk_valid, timeperiod=fastd, matype=fd_mt)
    fastd_valid = np.ascontiguousarray(fastd_full[fastd_lb:])

    out_k[total_lb:] = fastk_raw[fastk_lb + fastd_lb :]
    out_d[total_lb:] = fastd_valid
    return out_k, out_d

