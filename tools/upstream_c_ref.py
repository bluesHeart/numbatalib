from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DLL = REPO_ROOT / "generated" / "upstream_talib_ref.dll"


class TalibCError(RuntimeError):
    pass


def _as_f64_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("inputs must be 1D arrays")
    return np.ascontiguousarray(arr)


def _alloc_full(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


@dataclass(frozen=True)
class _CFuncSig:
    restype: Any
    argtypes: list[Any]


class UpstreamTalibCRef:
    def __init__(self, dll_path: Path = DEFAULT_DLL):
        if os.name != "nt":
            raise OSError("UpstreamTalibCRef is currently Windows-only")
        if not dll_path.exists():
            raise FileNotFoundError(
                f"Missing DLL: {dll_path}. Build it with: python tools/build_upstream_talib_ref.py"
            )

        self._dll = ctypes.CDLL(str(dll_path))
        self._bind()
        ret = int(self._dll.TA_Initialize())
        if ret != 0:
            raise TalibCError(f"TA_Initialize failed with code {ret}")

    def _bind(self) -> None:
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        sigs: dict[str, _CFuncSig] = {
            "TA_Initialize": _CFuncSig(ctypes.c_int, []),
            "TA_Shutdown": _CFuncSig(ctypes.c_int, []),
            "TA_ACCBANDS": _CFuncSig(
                ctypes.c_int,
                [
                    ctypes.c_int,
                    ctypes.c_int,
                    c_double_p,
                    c_double_p,
                    c_double_p,
                    ctypes.c_int,
                    c_int_p,
                    c_int_p,
                    c_double_p,
                    c_double_p,
                    c_double_p,
                ],
            ),
            "TA_AVGDEV": _CFuncSig(
                ctypes.c_int,
                [
                    ctypes.c_int,
                    ctypes.c_int,
                    c_double_p,
                    ctypes.c_int,
                    c_int_p,
                    c_int_p,
                    c_double_p,
                ],
            ),
            "TA_IMI": _CFuncSig(
                ctypes.c_int,
                [
                    ctypes.c_int,
                    ctypes.c_int,
                    c_double_p,
                    c_double_p,
                    ctypes.c_int,
                    c_int_p,
                    c_int_p,
                    c_double_p,
                ],
            ),
        }

        for name, sig in sigs.items():
            fn = getattr(self._dll, name)
            fn.restype = sig.restype
            fn.argtypes = sig.argtypes

    def shutdown(self) -> None:
        ret = int(self._dll.TA_Shutdown())
        if ret != 0:
            raise TalibCError(f"TA_Shutdown failed with code {ret}")

    def AVGDEV(self, real: Any, timeperiod: int = 14) -> np.ndarray:
        x = _as_f64_1d(real)
        n = int(x.shape[0])
        out_full = _alloc_full(n)
        if n == 0:
            return out_full

        out = np.empty(n, dtype=np.float64)
        out_beg = ctypes.c_int()
        out_n = ctypes.c_int()

        ret = int(
            self._dll.TA_AVGDEV(
                0,
                n - 1,
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(timeperiod),
                ctypes.byref(out_beg),
                ctypes.byref(out_n),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if ret != 0:
            raise TalibCError(f"TA_AVGDEV failed with code {ret}")

        beg = int(out_beg.value)
        nb = int(out_n.value)
        if nb > 0:
            out_full[beg : beg + nb] = out[:nb]
        return out_full

    def IMI(self, open_: Any, close: Any, timeperiod: int = 14) -> np.ndarray:
        o = _as_f64_1d(open_)
        c = _as_f64_1d(close)
        if o.shape[0] != c.shape[0]:
            raise ValueError("inputs must have the same length")
        n = int(o.shape[0])
        out_full = _alloc_full(n)
        if n == 0:
            return out_full

        out = np.empty(n, dtype=np.float64)
        out_beg = ctypes.c_int()
        out_n = ctypes.c_int()

        ret = int(
            self._dll.TA_IMI(
                0,
                n - 1,
                o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(timeperiod),
                ctypes.byref(out_beg),
                ctypes.byref(out_n),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if ret != 0:
            raise TalibCError(f"TA_IMI failed with code {ret}")

        beg = int(out_beg.value)
        nb = int(out_n.value)
        if nb > 0:
            out_full[beg : beg + nb] = out[:nb]
        return out_full

    def ACCBANDS(
        self, high: Any, low: Any, close: Any, timeperiod: int = 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = _as_f64_1d(high)
        l = _as_f64_1d(low)
        c = _as_f64_1d(close)
        if h.shape[0] != l.shape[0] or h.shape[0] != c.shape[0]:
            raise ValueError("inputs must have the same length")
        n = int(h.shape[0])

        upper_full = _alloc_full(n)
        middle_full = _alloc_full(n)
        lower_full = _alloc_full(n)
        if n == 0:
            return upper_full, middle_full, lower_full

        upper = np.empty(n, dtype=np.float64)
        middle = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        out_beg = ctypes.c_int()
        out_n = ctypes.c_int()

        ret = int(
            self._dll.TA_ACCBANDS(
                0,
                n - 1,
                h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                l.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(timeperiod),
                ctypes.byref(out_beg),
                ctypes.byref(out_n),
                upper.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                middle.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                lower.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if ret != 0:
            raise TalibCError(f"TA_ACCBANDS failed with code {ret}")

        beg = int(out_beg.value)
        nb = int(out_n.value)
        if nb > 0:
            upper_full[beg : beg + nb] = upper[:nb]
            middle_full[beg : beg + nb] = middle[:nb]
            lower_full[beg : beg + nb] = lower[:nb]
        return upper_full, middle_full, lower_full

