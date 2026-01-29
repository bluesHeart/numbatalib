from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param
from numbatalib._func.ta_ma import MA, _ma_lookback, _validate_matype


def MACDEXT(
    real,
    fastperiod: int = 12,
    fastmatype: int = 0,
    slowperiod: int = 26,
    slowmatype: int = 0,
    signalperiod: int = 9,
    signalmatype: int = 0,
):
    """
    MACD with controllable MA types for each stage.
    """
    real_arr = as_1d_float64(real)
    fp = validate_int_param("fastperiod", fastperiod, Range(min=2, max=100000))
    sp = validate_int_param("slowperiod", slowperiod, Range(min=2, max=100000))
    sigp = validate_int_param("signalperiod", signalperiod, Range(min=1, max=100000))
    fmt = _validate_matype(fastmatype)
    smt = _validate_matype(slowmatype)
    sigmt = _validate_matype(signalmatype)

    # Swap so that slowperiod is always >= fastperiod (TA-Lib behavior).
    if sp < fp:
        fp, sp = sp, fp
        fmt, smt = smt, fmt

    n = real_arr.shape[0]
    out_macd = nan_like(real_arr, dtype=np.float64)
    out_signal = nan_like(real_arr, dtype=np.float64)
    out_hist = nan_like(real_arr, dtype=np.float64)
    if n == 0:
        return out_macd, out_signal, out_hist

    lookback_largest = _ma_lookback(fp, fmt)
    lb_slow = _ma_lookback(sp, smt)
    if lb_slow > lookback_largest:
        lookback_largest = lb_slow
    lookback_signal = _ma_lookback(sigp, sigmt)
    lookback_total = lookback_largest + lookback_signal
    if lookback_total >= n:
        return out_macd, out_signal, out_hist

    fast_ma = MA(real_arr, timeperiod=fp, matype=fmt)
    slow_ma = MA(real_arr, timeperiod=sp, matype=smt)
    macd_line = fast_ma - slow_ma

    macd_valid = np.ascontiguousarray(macd_line[lookback_largest:])
    if sigp == 1:
        signal_valid = macd_valid
    else:
        sig_full = MA(macd_valid, timeperiod=sigp, matype=sigmt)
        signal_valid = np.ascontiguousarray(sig_full[lookback_signal:])

    out_macd[lookback_total:] = macd_line[lookback_total:]
    out_signal[lookback_total:] = signal_valid
    out_hist[lookback_total:] = out_macd[lookback_total:] - signal_valid
    return out_macd, out_signal, out_hist

