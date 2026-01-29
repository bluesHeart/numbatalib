from __future__ import annotations

import math

import numpy as np
from numba import njit


# TA-Lib RangeType enum
RANGE_REALBODY = 0
RANGE_HIGHLOW = 1
RANGE_SHADOWS = 2

# TA-Lib CandleSettingType enum (see upstream include/ta_defs.h)
BODY_LONG = 0
BODY_VERY_LONG = 1
BODY_SHORT = 2
BODY_DOJI = 3
SHADOW_LONG = 4
SHADOW_VERY_LONG = 5
SHADOW_SHORT = 6
SHADOW_VERY_SHORT = 7
NEAR = 8
FAR = 9
EQUAL = 10


# Default candle settings from `src/ta_common/ta_global.c`.
_CANDLE_RANGE_TYPE = np.array(
    [
        RANGE_REALBODY,  # BODY_LONG
        RANGE_REALBODY,  # BODY_VERY_LONG
        RANGE_REALBODY,  # BODY_SHORT
        RANGE_HIGHLOW,  # BODY_DOJI
        RANGE_REALBODY,  # SHADOW_LONG
        RANGE_REALBODY,  # SHADOW_VERY_LONG
        RANGE_SHADOWS,  # SHADOW_SHORT
        RANGE_HIGHLOW,  # SHADOW_VERY_SHORT
        RANGE_HIGHLOW,  # NEAR
        RANGE_HIGHLOW,  # FAR
        RANGE_HIGHLOW,  # EQUAL
    ],
    dtype=np.int32,
)

_CANDLE_AVG_PERIOD = np.array(
    [
        10,  # BODY_LONG
        10,  # BODY_VERY_LONG
        10,  # BODY_SHORT
        10,  # BODY_DOJI
        0,  # SHADOW_LONG
        0,  # SHADOW_VERY_LONG
        10,  # SHADOW_SHORT
        10,  # SHADOW_VERY_SHORT
        5,  # NEAR
        5,  # FAR
        5,  # EQUAL
    ],
    dtype=np.int32,
)

_CANDLE_FACTOR = np.array(
    [
        1.0,  # BODY_LONG
        3.0,  # BODY_VERY_LONG
        1.0,  # BODY_SHORT
        0.1,  # BODY_DOJI
        1.0,  # SHADOW_LONG
        2.0,  # SHADOW_VERY_LONG
        1.0,  # SHADOW_SHORT
        0.1,  # SHADOW_VERY_SHORT
        0.2,  # NEAR
        0.6,  # FAR
        0.05,  # EQUAL
    ],
    dtype=np.float64,
)


@njit(cache=True)
def real_body(open_: np.ndarray, close: np.ndarray, idx: int) -> float:
    return math.fabs(close[idx] - open_[idx])


@njit(cache=True)
def upper_shadow(open_: np.ndarray, high: np.ndarray, close: np.ndarray, idx: int) -> float:
    oc_max = close[idx] if close[idx] >= open_[idx] else open_[idx]
    return high[idx] - oc_max


@njit(cache=True)
def lower_shadow(open_: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int) -> float:
    oc_min = open_[idx] if close[idx] >= open_[idx] else close[idx]
    return oc_min - low[idx]


@njit(cache=True)
def high_low_range(high: np.ndarray, low: np.ndarray, idx: int) -> float:
    return high[idx] - low[idx]


@njit(cache=True)
def candle_color(open_: np.ndarray, close: np.ndarray, idx: int) -> int:
    return 1 if close[idx] >= open_[idx] else -1


@njit(cache=True)
def candle_range(
    setting: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
) -> float:
    rt = int(_CANDLE_RANGE_TYPE[setting])
    if rt == RANGE_REALBODY:
        return real_body(open_, close, idx)
    if rt == RANGE_HIGHLOW:
        return high_low_range(high, low, idx)
    # RANGE_SHADOWS
    return upper_shadow(open_, high, close, idx) + lower_shadow(open_, low, close, idx)


@njit(cache=True)
def candle_average(
    setting: int,
    period_total: float,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
) -> float:
    avg_period = int(_CANDLE_AVG_PERIOD[setting])
    rng = (
        period_total / avg_period
        if avg_period != 0
        else candle_range(setting, open_, high, low, close, idx)
    )
    # Shadows are split in two (upper/lower).
    if int(_CANDLE_RANGE_TYPE[setting]) == RANGE_SHADOWS:
        rng *= 0.5
    return float(_CANDLE_FACTOR[setting]) * rng


@njit(cache=True)
def real_body_gap_up(open_: np.ndarray, close: np.ndarray, idx2: int, idx1: int) -> bool:
    lo2 = open_[idx2] if open_[idx2] < close[idx2] else close[idx2]
    hi1 = open_[idx1] if open_[idx1] > close[idx1] else close[idx1]
    return lo2 > hi1


@njit(cache=True)
def real_body_gap_down(open_: np.ndarray, close: np.ndarray, idx2: int, idx1: int) -> bool:
    hi2 = open_[idx2] if open_[idx2] > close[idx2] else close[idx2]
    lo1 = open_[idx1] if open_[idx1] < close[idx1] else close[idx1]
    return hi2 < lo1


@njit(cache=True)
def candle_gap_up(high: np.ndarray, low: np.ndarray, idx2: int, idx1: int) -> bool:
    return low[idx2] > high[idx1]


@njit(cache=True)
def candle_gap_down(high: np.ndarray, low: np.ndarray, idx2: int, idx1: int) -> bool:
    return high[idx2] < low[idx1]

