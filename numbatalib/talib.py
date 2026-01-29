from __future__ import annotations

"""
Convenience alias for the TA-Lib compatibility shim.

Prefer this import when you want a shorter path:

    import numbatalib.talib as talib

It is equivalent to:

    import numbatalib.compat.talib as talib
"""

from .compat import talib as _talib
from .compat.talib import *  # noqa: F403


def __getattr__(name: str):  # pragma: no cover
    return getattr(_talib, name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(dir(_talib)))


__all__ = list(getattr(_talib, "__all__", []))

