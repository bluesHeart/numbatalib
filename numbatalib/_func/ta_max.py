from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _max_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    nb_initial = timeperiod - 1
    today = nb_initial
    trailing = 0

    highest_idx = -1
    highest = 0.0

    while today < n:
        tmp = real[today]
        if highest_idx < trailing:
            highest_idx = trailing
            highest = real[highest_idx]
            i = highest_idx
            while i < today:
                i += 1
                tmp2 = real[i]
                if tmp2 > highest:
                    highest_idx = i
                    highest = tmp2
        elif tmp >= highest:
            highest_idx = today
            highest = tmp

        out[today] = highest
        trailing += 1
        today += 1


def MAX(real, timeperiod: int = 30):
    """
    Highest value over a specified period.
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _max_kernel(real_arr, tp, out)
    return out

