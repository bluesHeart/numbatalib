from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("numbatalib")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

from ._registry import available_functions, get_function, implemented_functions


def __getattr__(name: str):
    fn = get_function(name)
    if fn is None:
        raise AttributeError(name)
    return fn


__all__ = [
    "available_functions",
    "implemented_functions",
    "get_function",
    # Dynamic TA-Lib function names are exposed via __getattr__.
]
