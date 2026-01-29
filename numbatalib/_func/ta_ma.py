from __future__ import annotations

import numpy as np

from numbatalib._core._validation import Range, as_1d_float64, nan_like, validate_int_param


def _validate_matype(matype: int) -> int:
    try:
        mt = int(matype)
    except Exception as e:
        raise ValueError("matype must be an int") from e
    if mt < 0 or mt > 8:
        raise ValueError("matype out of range")
    return mt


def _ma_lookback(timeperiod: int, matype: int) -> int:
    """
    Return TA-Lib lookback for the given MA configuration.

    This is used by indicators like MAVP to determine how many leading
    values are undefined regardless of per-element periods.
    """
    if timeperiod <= 0:
        return 0
    if timeperiod == 1:
        return 0

    lb = timeperiod - 1

    if matype in (0, 1, 2, 5):  # SMA/EMA/WMA/TRIMA
        return lb
    if matype == 3:  # DEMA
        return 2 * lb
    if matype == 4:  # TEMA
        return 3 * lb
    if matype == 6:  # KAMA
        return timeperiod
    if matype == 7:  # MAMA (implemented later)
        # TA-Lib ignores `optInTimePeriod` for MAMA and uses fixed limits (0.5/0.05),
        # which implies a constant lookback of 32 (assuming unstable period = 0).
        return 32
    if matype == 8:  # T3
        return 6 * lb

    return lb


def MA(real, timeperiod: int = 30, matype: int = 0):
    """
    Moving average with selectable type.

    MAType mapping (TA-Lib):
      0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
    """
    real_arr = as_1d_float64(real)
    tp = validate_int_param("timeperiod", timeperiod, Range(min=1, max=100000))
    mt = _validate_matype(matype)

    if mt == 7:
        from numbatalib._func.ta_mama import MAMA

        # TA-Lib behavior: ignore `timeperiod` and use the MAMA output with fixed limits.
        mama, _fama = MAMA(real_arr, fastlimit=0.5, slowlimit=0.05)
        return mama

    if tp == 1:
        return np.asarray(real_arr, dtype=np.float64).copy()

    if mt == 0:
        from numbatalib._func.ta_sma import _sma_kernel

        out = nan_like(real_arr, dtype=np.float64)
        _sma_kernel(real_arr, tp, out)
        return out
    if mt == 1:
        from numbatalib._func.ta_ema import _ema_kernel

        out = nan_like(real_arr, dtype=np.float64)
        _ema_kernel(real_arr, tp, out)
        return out
    if mt == 2:
        from numbatalib._func.ta_wma import _wma_kernel

        out = nan_like(real_arr, dtype=np.float64)
        _wma_kernel(real_arr, tp, out)
        return out
    if mt == 3:
        from numbatalib._func.ta_dema import DEMA

        return DEMA(real_arr, timeperiod=tp)
    if mt == 4:
        from numbatalib._func.ta_tema import TEMA

        return TEMA(real_arr, timeperiod=tp)
    if mt == 5:
        from numbatalib._func.ta_trima import TRIMA

        return TRIMA(real_arr, timeperiod=tp)
    if mt == 6:
        from numbatalib._func.ta_kama import KAMA

        return KAMA(real_arr, timeperiod=tp)
    if mt == 7:
        raise RuntimeError("unreachable")
    if mt == 8:
        from numbatalib._func.ta_t3 import T3

        return T3(real_arr, timeperiod=tp, vfactor=0.7)

    raise RuntimeError("unreachable")
