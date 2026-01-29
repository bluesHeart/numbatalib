from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _acos_kernel(real: np.ndarray, out: np.ndarray) -> None:
    n = real.shape[0]
    for i in range(n):
        out[i] = np.arccos(real[i])


def ACOS(real):
    """
    Vector Trigonometric ACos
    """
    real_arr = as_1d_float64(real)
    out = np.empty(real_arr.shape[0], dtype=np.float64)
    _acos_kernel(real_arr, out)
    return out

