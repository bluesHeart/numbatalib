from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


@njit(cache=True)
def _roc_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if n <= timeperiod:
        return
    i = timeperiod
    while i < n:
        old = real[i - timeperiod]
        if old == 0.0:
            out[i] = 0.0
        else:
            out[i] = (real[i] - old) / old * 100.0
        i += 1


def ROC(real, timeperiod: int = 10):
    """
    Rate of change : ((price/prevPrice)-1)*100
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))

    out = nan_like(real_arr, dtype=np.float64)
    _roc_kernel(real_arr, tp, out)
    return out

