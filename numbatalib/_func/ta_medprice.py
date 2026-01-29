from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _medprice_kernel(high: np.ndarray, low: np.ndarray, out: np.ndarray) -> None:
    n = high.shape[0]
    for i in range(n):
        out[i] = (high[i] + low[i]) * 0.5


def MEDPRICE(high, low):
    """
    Median Price
    """
    h = as_1d_float64(high)
    l = as_1d_float64(low)
    if h.shape[0] != l.shape[0]:
        raise ValueError("inputs must have the same length")

    out = np.empty(h.shape[0], dtype=np.float64)
    _medprice_kernel(h, l, out)
    return out

