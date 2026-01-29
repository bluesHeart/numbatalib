from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _adosc_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    fastperiod: int,
    slowperiod: int,
    out: np.ndarray,
) -> None:
    n = high.shape[0]
    if n == 0:
        return

    slowest = slowperiod if fastperiod < slowperiod else fastperiod
    lookback = slowest - 1
    if n <= lookback:
        return

    today = 0
    start = lookback

    ad = 0.0
    fastk = 2.0 / (fastperiod + 1.0)
    slowk = 2.0 / (slowperiod + 1.0)
    one_minus_fastk = 1.0 - fastk
    one_minus_slowk = 1.0 - slowk

    # Seed with first A/D value.
    tmp = high[today] - low[today]
    if tmp > 0.0:
        ad += (((close[today] - low[today]) - (high[today] - close[today])) / tmp) * volume[today]
    today += 1
    fast_ema = ad
    slow_ema = ad

    # Warm up until `start`.
    while today < start:
        tmp = high[today] - low[today]
        if tmp > 0.0:
            ad += (((close[today] - low[today]) - (high[today] - close[today])) / tmp) * volume[today]
        today += 1
        fast_ema = (fastk * ad) + (one_minus_fastk * fast_ema)
        slow_ema = (slowk * ad) + (one_minus_slowk * slow_ema)

    while today < n:
        tmp = high[today] - low[today]
        if tmp > 0.0:
            ad += (((close[today] - low[today]) - (high[today] - close[today])) / tmp) * volume[today]
        today += 1
        fast_ema = (fastk * ad) + (one_minus_fastk * fast_ema)
        slow_ema = (slowk * ad) + (one_minus_slowk * slow_ema)
        out[today - 1] = fast_ema - slow_ema


def ADOSC(high, low, close, volume, fastperiod: int = 3, slowperiod: int = 10):
    """
    Chaikin A/D Oscillator
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    c = as_1d_float64(close)
    v = as_1d_float64(volume)
    n = h.shape[0]
    if l.shape[0] != n or c.shape[0] != n or v.shape[0] != n:
        raise ValueError("inputs must have the same length")

    fp = validate_int_param("fastperiod", fastperiod, Range(min=2, max=100000))
    sp = validate_int_param("slowperiod", slowperiod, Range(min=2, max=100000))

    out = nan_like(h, dtype=np.float64)
    _adosc_kernel(h, l, c, v, fp, sp, out)
    return out

