from __future__ import annotations

import numbatalib.compat.talib._ta_lib as _ta_lib
from ._ta_lib import __TA_FUNCTION_NAMES__


for _func_name in __TA_FUNCTION_NAMES__:
    globals()[_func_name] = getattr(_ta_lib, f"stream_{_func_name}")

