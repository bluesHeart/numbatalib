from __future__ import annotations

from numbatalib._func.ta_maxindex import MAXINDEX
from numbatalib._func.ta_minindex import MININDEX


def MINMAXINDEX(real, timeperiod: int = 30):
    """
    Indices of lowest and highest values over a specified period.
    """
    return MININDEX(real, timeperiod=timeperiod), MAXINDEX(real, timeperiod=timeperiod)

