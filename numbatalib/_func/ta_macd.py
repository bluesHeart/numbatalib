from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _ema_seeded_from_idx0_kernel(
    real: np.ndarray, start_idx: int, period: int, k: float, out: np.ndarray
) -> None:
    """
    TA-Lib-style EMA starting at `start_idx` (inclusive) and ending at `n-1`.

    This matches TA_INT_EMA behavior for TA_COMPATIBILITY_DEFAULT when called
    with startIdx > lookbackTotal: the seed is an SMA of the `period` values
    ending at startIdx (not the classic EMA seed from the beginning of series).

    `out` length must be `n - start_idx`.
    """
    n = real.shape[0]
    if start_idx >= n:
        return

    lookback = period - 1
    if start_idx < lookback:
        start_idx = lookback
    if start_idx >= n:
        return

    # Seed: SMA of `period` values ending at start_idx.
    s = 0.0
    base = start_idx - lookback
    for i in range(period):
        s += real[base + i]
    prev = s / period
    out[0] = prev

    out_i = 1
    for i in range(start_idx + 1, n):
        prev = ((real[i] - prev) * k) + prev
        out[out_i] = prev
        out_i += 1


@njit(cache=True)
def _macd_kernel(
    real: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    out_macd: np.ndarray,
    out_signal: np.ndarray,
    out_hist: np.ndarray,
) -> None:
    n = real.shape[0]
    if n == 0:
        return

    fp = fast_period
    sp = slow_period
    sigp = signal_period

    # TA-Lib behavior: swap if slow < fast.
    if sp < fp:
        fp, sp = sp, fp

    # TA-Lib special "fixed 12/26" case: use hard-coded k values.
    if sp != 0:
        k_slow = 2.0 / (sp + 1.0)
    else:
        sp = 26
        k_slow = 0.075  # fixed 26

    if fp != 0:
        k_fast = 2.0 / (fp + 1.0)
    else:
        fp = 12
        k_fast = 0.15  # fixed 12

    lookback_signal = sigp - 1 if sigp > 1 else 0
    lookback_total = lookback_signal + (sp - 1)
    if lookback_total >= n:
        return

    start_idx = lookback_total
    # "Move back startIdx by signal lookback" like TA_INT_MACD.
    ema_start = start_idx - lookback_signal  # equals sp-1

    # EMA buffers from `ema_start` to end.
    buf_len = n - ema_start
    fast_buf = np.empty(buf_len, dtype=np.float64)
    slow_buf = np.empty(buf_len, dtype=np.float64)
    _ema_seeded_from_idx0_kernel(real, ema_start, fp, k_fast, fast_buf)
    _ema_seeded_from_idx0_kernel(real, ema_start, sp, k_slow, slow_buf)

    # Compute MACD + signal + histogram.
    if sigp == 1:
        for t in range(buf_len):
            idx = ema_start + t
            macd_val = fast_buf[t] - slow_buf[t]
            out_macd[idx] = macd_val
            out_signal[idx] = macd_val
            out_hist[idx] = 0.0
        return

    k_sig = 2.0 / (sigp + 1.0)

    # Seed signal EMA with SMA of the first `sigp` MACD values in the buffer.
    s = 0.0
    for i in range(sigp):
        s += fast_buf[i] - slow_buf[i]
    prev_sig = s / sigp

    for t in range(buf_len):
        idx = ema_start + t
        macd_val = fast_buf[t] - slow_buf[t]
        if t < lookback_signal:
            # Not output yet, but MACD values still contribute to signal seed.
            continue
        if t == lookback_signal:
            sig_val = prev_sig
        else:
            prev_sig = ((macd_val - prev_sig) * k_sig) + prev_sig
            sig_val = prev_sig

        out_macd[idx] = macd_val
        out_signal[idx] = sig_val
        out_hist[idx] = macd_val - sig_val


def MACD(real, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
    """
    Moving Average Convergence/Divergence
    """
    real_arr = as_1d_float64(real)
    fp = validate_int_param("fastperiod", fastperiod, Range(min=2, max=100000))
    sp = validate_int_param("slowperiod", slowperiod, Range(min=2, max=100000))
    sigp = validate_int_param("signalperiod", signalperiod, Range(min=1, max=100000))

    out_macd = nan_like(real_arr, dtype=np.float64)
    out_signal = nan_like(real_arr, dtype=np.float64)
    out_hist = nan_like(real_arr, dtype=np.float64)

    _macd_kernel(real_arr, fp, sp, sigp, out_macd, out_signal, out_hist)
    return out_macd, out_signal, out_hist
