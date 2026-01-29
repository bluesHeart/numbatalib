from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_float_param


TA_REAL_MAX = 3e37


@njit(cache=True)
def _sar_kernel(
    high: np.ndarray,
    low: np.ndarray,
    acceleration: float,
    maximum: float,
    out: np.ndarray,
) -> None:
    n = high.shape[0]
    if n < 2:
        return

    # Coerce acceleration/max.
    af = acceleration
    if af > maximum:
        af = maximum

    # Determine initial direction using -DM on the first transition.
    up_move = high[1] - high[0]
    down_move = low[0] - low[1]
    is_long = 1
    if down_move > 0.0 and down_move > up_move:
        is_long = 0

    new_high = high[0]
    new_low = low[0]

    if is_long == 1:
        ep = high[1]
        sar = new_low
    else:
        ep = low[1]
        sar = new_high

    # Cheat for first iteration.
    new_low = low[1]
    new_high = high[1]

    idx = 1
    while idx < n:
        prev_low = new_low
        prev_high = new_high

        new_low = low[idx]
        new_high = high[idx]

        if is_long == 1:
            if new_low <= sar:
                # Switch to short.
                is_long = 0
                sar = ep
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
                out[idx] = sar

                af = acceleration
                ep = new_low

                sar = sar + af * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
            else:
                out[idx] = sar

                if new_high > ep:
                    ep = new_high
                    af += acceleration
                    if af > maximum:
                        af = maximum

                sar = sar + af * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
        else:
            if new_high >= sar:
                # Switch to long.
                is_long = 1
                sar = ep
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
                out[idx] = sar

                af = acceleration
                ep = new_high

                sar = sar + af * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
            else:
                out[idx] = sar

                if new_low < ep:
                    ep = new_low
                    af += acceleration
                    if af > maximum:
                        af = maximum

                sar = sar + af * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high

        idx += 1


def SAR(high, low, acceleration: float = 0.02, maximum: float = 0.2):
    """
    Parabolic SAR
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    if l.shape[0] != h.shape[0]:
        raise ValueError("inputs must have the same length")

    acc = validate_float_param("acceleration", acceleration, Range(min=0.0, max=TA_REAL_MAX))
    mx = validate_float_param("maximum", maximum, Range(min=0.0, max=TA_REAL_MAX))
    if acc > mx:
        acc = mx

    out = nan_like(h, dtype=np.float64)
    _sar_kernel(h, l, acc, mx, out)
    return out

