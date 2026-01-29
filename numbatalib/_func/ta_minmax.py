from __future__ import annotations

from numbatalib._func.ta_max import MAX
from numbatalib._func.ta_min import MIN


def MINMAX(real, timeperiod: int = 30):
    """
    Lowest and highest values over a specified period.
    """
    return MIN(real, timeperiod=timeperiod), MAX(real, timeperiod=timeperiod)

