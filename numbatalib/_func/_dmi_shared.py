from __future__ import annotations

import math

import numpy as np
from numba import njit


TA_EPSILON = 1e-14


@njit(cache=True)
def _ta_is_zero(x: float) -> bool:
    return math.fabs(x) < TA_EPSILON


@njit(cache=True)
def _true_range(curr_high: float, curr_low: float, prev_close: float) -> float:
    """TA-Lib TRUE_RANGE macro."""
    tr = curr_high - curr_low
    diff = math.fabs(curr_high - prev_close)
    if diff > tr:
        tr = diff
    diff = math.fabs(curr_low - prev_close)
    if diff > tr:
        tr = diff
    return tr


@njit(cache=True)
def _dm_deltas(
    curr_high: float, curr_low: float, prev_high: float, prev_low: float
) -> tuple[float, float]:
    """Return (diffP, diffM) matching TA-Lib definitions."""
    diff_p = curr_high - prev_high
    diff_m = prev_low - curr_low
    return diff_p, diff_m

