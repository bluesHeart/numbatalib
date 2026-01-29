from __future__ import annotations

import math
import numpy as np
from numba import njit

from numbatalib._core._validation import (
    Range,
    as_1d_float64,
    nan_like,
    validate_float_param,
)


TA_REAL_MAX = 3e37
TA_REAL_MIN = -3e37


@njit(cache=True)
def _sarext_kernel(
    high: np.ndarray,
    low: np.ndarray,
    startvalue: float,
    offsetonreverse: float,
    accelerationinitlong: float,
    accelerationlong: float,
    accelerationmaxlong: float,
    accelerationinitshort: float,
    accelerationshort: float,
    accelerationmaxshort: float,
    out: np.ndarray,
) -> None:
    n = high.shape[0]
    if n < 2:
        return

    # Coerce acceleration init/max relationships.
    if accelerationinitlong > accelerationmaxlong:
        accelerationinitlong = accelerationmaxlong
    if accelerationinitshort > accelerationmaxshort:
        accelerationinitshort = accelerationmaxshort

    af_long = accelerationinitlong
    af_short = accelerationinitshort
    if accelerationlong > accelerationmaxlong:
        accelerationlong = accelerationmaxlong
    if accelerationshort > accelerationmaxshort:
        accelerationshort = accelerationmaxshort

    # Determine initial direction.
    if startvalue == 0.0:
        up_move = high[1] - high[0]
        down_move = low[0] - low[1]
        is_long = 1
        if down_move > 0.0 and down_move > up_move:
            is_long = 0
    elif startvalue > 0.0:
        is_long = 1
    else:
        is_long = 0

    new_high = high[0]
    new_low = low[0]

    if startvalue == 0.0:
        if is_long == 1:
            ep = high[1]
            sar = new_low
        else:
            ep = low[1]
            sar = new_high
    elif startvalue > 0.0:
        ep = high[1]
        sar = startvalue
    else:
        ep = low[1]
        sar = math.fabs(startvalue)

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

                if offsetonreverse != 0.0:
                    sar += sar * offsetonreverse
                out[idx] = -sar

                af_short = accelerationinitshort
                ep = new_low

                sar = sar + af_short * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
            else:
                out[idx] = sar

                if new_high > ep:
                    ep = new_high
                    af_long += accelerationlong
                    if af_long > accelerationmaxlong:
                        af_long = accelerationmaxlong

                sar = sar + af_long * (ep - sar)
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

                if offsetonreverse != 0.0:
                    sar -= sar * offsetonreverse
                out[idx] = sar

                af_long = accelerationinitlong
                ep = new_high

                sar = sar + af_long * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
            else:
                out[idx] = -sar

                if new_low < ep:
                    ep = new_low
                    af_short += accelerationshort
                    if af_short > accelerationmaxshort:
                        af_short = accelerationmaxshort

                sar = sar + af_short * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high

        idx += 1


def SAREXT(
    high,
    low,
    startvalue: float = 0.0,
    offsetonreverse: float = 0.0,
    accelerationinitlong: float = 0.02,
    accelerationlong: float = 0.02,
    accelerationmaxlong: float = 0.2,
    accelerationinitshort: float = 0.02,
    accelerationshort: float = 0.02,
    accelerationmaxshort: float = 0.2,
):
    """
    Parabolic SAR - Extended

    Note: Matches TA-Lib behavior where SAR values are negative when short.
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    if l.shape[0] != h.shape[0]:
        raise ValueError("inputs must have the same length")

    sv = validate_float_param("startvalue", startvalue, Range(min=TA_REAL_MIN, max=TA_REAL_MAX))
    oor = validate_float_param("offsetonreverse", offsetonreverse, Range(min=0.0, max=TA_REAL_MAX))
    ail = validate_float_param("accelerationinitlong", accelerationinitlong, Range(min=0.0, max=TA_REAL_MAX))
    al = validate_float_param("accelerationlong", accelerationlong, Range(min=0.0, max=TA_REAL_MAX))
    aml = validate_float_param("accelerationmaxlong", accelerationmaxlong, Range(min=0.0, max=TA_REAL_MAX))
    ais = validate_float_param("accelerationinitshort", accelerationinitshort, Range(min=0.0, max=TA_REAL_MAX))
    a_s = validate_float_param("accelerationshort", accelerationshort, Range(min=0.0, max=TA_REAL_MAX))
    ams = validate_float_param("accelerationmaxshort", accelerationmaxshort, Range(min=0.0, max=TA_REAL_MAX))

    out = nan_like(h, dtype=np.float64)
    _sarext_kernel(h, l, sv, oor, ail, al, aml, ais, a_s, ams, out)
    return out
