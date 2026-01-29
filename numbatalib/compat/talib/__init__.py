from __future__ import annotations

import atexit
from functools import wraps
from itertools import chain

# Optional pandas/polars Series support (mirrors talib's behavior).
try:  # pragma: no cover
    from polars import Series as _pl_Series
except Exception:  # pragma: no cover
    _pl_Series = None

try:  # pragma: no cover
    from pandas import Series as _pd_Series
except Exception:  # pragma: no cover
    _pd_Series = None


if _pl_Series is not None or _pd_Series is not None:  # pragma: no cover

    def _wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            if _pl_Series is not None:
                use_pl = any(isinstance(arg, _pl_Series) for arg in args) or any(
                    isinstance(v, _pl_Series) for v in kwds.values()
                )
            else:
                use_pl = False

            if _pd_Series is not None:
                use_pd = any(isinstance(arg, _pd_Series) for arg in args) or any(
                    isinstance(v, _pd_Series) for v in kwds.values()
                )
            else:
                use_pd = False

            if use_pl and use_pd:
                raise Exception("Cannot mix polars and pandas")

            if use_pl:
                _args = [
                    arg.to_numpy().astype(float) if isinstance(arg, _pl_Series) else arg
                    for arg in args
                ]
                _kwds = {
                    k: (v.to_numpy().astype(float) if isinstance(v, _pl_Series) else v)
                    for k, v in kwds.items()
                }
            elif use_pd:
                index = next(
                    arg.index
                    for arg in chain(args, kwds.values())
                    if isinstance(arg, _pd_Series)
                )
                _args = [
                    arg.to_numpy().astype(float) if isinstance(arg, _pd_Series) else arg
                    for arg in args
                ]
                _kwds = {
                    k: (v.to_numpy().astype(float) if isinstance(v, _pd_Series) else v)
                    for k, v in kwds.items()
                }
            else:
                _args = args
                _kwds = kwds

            result = func(*_args, **_kwds)

            first_result = result[0] if isinstance(result, tuple) else result
            is_streaming_fn_result = not hasattr(first_result, "__len__")
            if is_streaming_fn_result:
                return result

            if use_pl:
                if isinstance(result, tuple):
                    return tuple(_pl_Series(arr) for arr in result)
                return _pl_Series(result)

            if use_pd:
                if isinstance(result, tuple):
                    return tuple(_pd_Series(arr, index=index) for arr in result)
                return _pd_Series(result, index=index)

            return result

        return wrapper

else:
    _wrapper = lambda x: x


from ._ta_lib import (  # noqa: E402
    MA_Type,
    __TA_FUNCTION_NAMES__,
    __ta_version__,
    _ta_get_compatibility as get_compatibility,
    _ta_get_unstable_period as get_unstable_period,
    _ta_initialize,
    _ta_set_compatibility as set_compatibility,
    _ta_set_unstable_period as set_unstable_period,
    _ta_shutdown,
)
from ._ta_lib import *  # noqa: F403,E402


_ta_initialize()
atexit.register(_ta_shutdown)


# Wrap functions for pandas/polars support (mirrors talib).
from . import _ta_lib as _func_mod  # noqa: E402

for _func_name in __TA_FUNCTION_NAMES__:
    _wrapped = _wrapper(getattr(_func_mod, _func_name))
    setattr(_func_mod, _func_name, _wrapped)
    globals()[_func_name] = _wrapped

from . import stream as stream  # noqa: E402

for _func_name in __TA_FUNCTION_NAMES__:
    _wrapped = _wrapper(getattr(stream, _func_name))
    setattr(stream, _func_name, _wrapped)
    globals()[f"stream_{_func_name}"] = _wrapped


__version__ = "0.4.32"


def get_functions():
    """
    Returns a list of all the functions supported by TALIB.
    """
    ret = []
    for group in __function_groups__:
        ret.extend(__function_groups__[group])
    return ret


def get_function_groups():
    """
    Returns a dict with keys of function-group names and values of lists
    of function names ie {'group_names': ['function_names']}.
    """
    return __function_groups__.copy()


def _build_function_groups() -> dict[str, list[str]]:
    from numbatalib._registry import _load_meta

    meta = _load_meta()
    groups: dict[str, list[str]] = {}
    for func_name in __TA_FUNCTION_NAMES__:
        group = meta[func_name].group_name
        groups.setdefault(group, []).append(func_name)

    for group, names in groups.items():
        groups[group] = sorted(names)

    order = [
        "Cycle Indicators",
        "Math Operators",
        "Math Transform",
        "Momentum Indicators",
        "Overlap Studies",
        "Pattern Recognition",
        "Price Transform",
        "Statistic Functions",
        "Volatility Indicators",
        "Volume Indicators",
    ]
    ordered: dict[str, list[str]] = {}
    for g in order:
        if g in groups:
            ordered[g] = groups[g]
    for g in sorted(groups.keys()):
        if g not in ordered:
            ordered[g] = groups[g]
    return ordered


__function_groups__ = _build_function_groups()


__all__ = (
    ["get_functions", "get_function_groups"]
    + list(__TA_FUNCTION_NAMES__)
    + [f"stream_{name}" for name in __TA_FUNCTION_NAMES__]
)

