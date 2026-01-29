from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, validate_int_param


@njit(cache=True)
def _minindex_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    nb_initial = timeperiod - 1
    today = nb_initial
    trailing = 0

    lowest_idx = -1
    lowest = 0.0

    while today < n:
        tmp = real[today]
        if lowest_idx < trailing:
            lowest_idx = trailing
            lowest = real[lowest_idx]
            i = lowest_idx
            while i < today:
                i += 1
                tmp2 = real[i]
                if tmp2 < lowest:
                    lowest_idx = i
                    lowest = tmp2
        elif tmp <= lowest:
            lowest_idx = today
            lowest = tmp

        out[today] = lowest_idx
        trailing += 1
        today += 1


def MININDEX(real, timeperiod: int = 30):
    """
    Index of lowest value over a specified period.
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=2, max=100000))

    out = np.zeros(real_arr.shape[0], dtype=np.int32)
    _minindex_kernel(real_arr, tp, out)
    return out

