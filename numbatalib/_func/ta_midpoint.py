from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _midpoint_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    nb_initial = timeperiod - 1
    today = nb_initial
    trailing = 0

    while today < n:
        lowest = real[trailing]
        highest = lowest
        i = trailing + 1
        while i <= today:
            tmp = real[i]
            if tmp < lowest:
                lowest = tmp
            elif tmp > highest:
                highest = tmp
            i += 1

        out[today] = (highest + lowest) * 0.5
        trailing += 1
        today += 1


def MIDPOINT(real, timeperiod: int = 14):
    """
    MidPoint over period.
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _midpoint_kernel(real_arr, tp, out)
    return out
