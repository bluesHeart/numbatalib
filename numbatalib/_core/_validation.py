from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Range:
    min: int | float | None = None
    max: int | float | None = None


def as_1d_float64(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("input must be 1-D")
    return np.ascontiguousarray(arr)


def validate_int_param(name: str, value: Any, allowed: Range) -> int:
    try:
        v = int(value)
    except Exception as e:
        raise ValueError(f"{name} must be an int") from e

    if allowed.min is not None and v < allowed.min:
        raise ValueError(f"{name} out of range")
    if allowed.max is not None and v > allowed.max:
        raise ValueError(f"{name} out of range")
    return v


def validate_float_param(name: str, value: Any, allowed: Range) -> float:
    try:
        v = float(value)
    except Exception as e:
        raise ValueError(f"{name} must be a float") from e

    if allowed.min is not None and v < allowed.min:
        raise ValueError(f"{name} out of range")
    if allowed.max is not None and v > allowed.max:
        raise ValueError(f"{name} out of range")
    return v


def nan_like(x: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=dtype)
    if np.issubdtype(out.dtype, np.floating):
        out.fill(np.nan)
    else:
        out.fill(0)
    return out
