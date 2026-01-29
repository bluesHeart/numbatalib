from __future__ import annotations

import numpy as np
from numba import njit

from numbatalib._core._validation import as_1d_float64


@njit(cache=True)
def _mult_kernel(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    n = a.shape[0]
    for i in range(n):
        out[i] = a[i] * b[i]


def MULT(real0, real1):
    """
    Vector Arithmetic Multiply
    """
    a = as_1d_float64(real0)
    b = as_1d_float64(real1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("inputs must have the same length")

    out = np.empty(a.shape[0], dtype=np.float64)
    _mult_kernel(a, b, out)
    return out

