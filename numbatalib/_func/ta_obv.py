from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _obv_kernel(real: np.ndarray, volume: np.ndarray, out: np.ndarray) -> None:
    n = real.shape[0]
    if n == 0:
        return

    prev_obv = volume[0]
    prev_real = real[0]

    i = 0
    while i < n:
        temp_real = real[i]
        if temp_real > prev_real:
            prev_obv += volume[i]
        elif temp_real < prev_real:
            prev_obv -= volume[i]

        out[i] = prev_obv
        prev_real = temp_real
        i += 1


def OBV(real, volume):
    """
    On Balance Volume
    """
    r = as_1d_float64(real)
    v = as_1d_float64(volume)
    if r.shape[0] != v.shape[0]:
        raise ValueError("inputs must have the same length")

    out = np.empty(r.shape[0], dtype=np.float64)
    _obv_kernel(r, v, out)
    return out

