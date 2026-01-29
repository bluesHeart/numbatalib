from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numbatalib  # noqa: E402
from numbatalib._registry import FunctionMeta, _load_meta  # noqa: E402


TA_REAL_MIN = -3e37
TA_REAL_MAX = 3e37


@dataclass(frozen=True)
class ParityCase:
    func: str
    inputs: list[np.ndarray]
    kwargs: dict[str, Any]


def _optin_to_kw(optin_name: str) -> str:
    if not optin_name.startswith("optIn"):
        return optin_name
    return optin_name[len("optIn") :].lower()


def _bound_to_float(x: str) -> float:
    if x == "TA_REAL_MIN":
        return TA_REAL_MIN
    if x == "TA_REAL_MAX":
        return TA_REAL_MAX
    return float(x)


def _bound_to_int(x: str) -> int:
    return int(x)


def _make_ohlcv(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    # Start from a random walk "close".
    steps = rng.normal(loc=0.0, scale=1.0, size=n)
    close = steps.cumsum().astype(np.float64)
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(loc=0.0, scale=0.5, size=n)).astype(np.float64)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(low=1.0, high=1e6, size=n).astype(np.float64)
    return {
        "inOpen": open_,
        "inHigh": high,
        "inLow": low,
        "inClose": close,
        "inVolume": volume,
    }


def _make_inputs(meta: FunctionMeta, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    ohlcv = _make_ohlcv(n, rng)
    inputs: list[np.ndarray] = []
    for name in meta.inputs:
        if name in ohlcv:
            inputs.append(ohlcv[name])
        elif name.startswith("inReal"):
            inputs.append(rng.normal(size=n).astype(np.float64))
        else:
            # Fallback: generate a generic real-valued series.
            inputs.append(rng.normal(size=n).astype(np.float64))
    return inputs


def _allowed_ranges(meta: FunctionMeta) -> dict[str, tuple[str, str, str]]:
    """
    Return mapping: kw_name -> (c_type, from, to)
    """
    out: dict[str, tuple[str, str, str]] = {}
    for opt in meta.opt_inputs:
        name = _optin_to_kw(opt["name"])
        r_from = opt.get("range_from")
        r_to = opt.get("range_to")
        if r_from is None or r_to is None:
            continue
        out[name] = (opt.get("c_type", ""), r_from, r_to)
    return out


def _talib_defaults(func_name: str) -> dict[str, Any]:
    try:
        from talib import abstract

        f = abstract.Function(func_name)
        return dict(f.parameters)
    except Exception:
        return {}


def _sample_params(
    func_name: str, meta: FunctionMeta, n: int, rng: np.random.Generator
) -> dict[str, Any]:
    allowed = _allowed_ranges(meta)
    defaults = _talib_defaults(func_name)
    params: dict[str, Any] = {}

    for kw, (c_type, r_from, r_to) in allowed.items():
        lo_i = None
        hi_i = None
        lo_f = None
        hi_f = None

        if "double" in c_type:
            lo_f = _bound_to_float(r_from)
            hi_f = _bound_to_float(r_to)
            default = float(defaults.get(kw, 0.0))
            # Sample near default but clamp to allowed.
            span = max(1e-6, abs(default) * 0.25)
            val = float(rng.normal(loc=default, scale=span))
            val = float(np.clip(val, lo_f, hi_f))
            params[kw] = val
        else:
            lo_i = _bound_to_int(r_from)
            hi_i = _bound_to_int(r_to)
            default = int(defaults.get(kw, lo_i))

            # For "period"-like params, avoid absurdly large values by default.
            period_like = "period" in kw
            hi_eff = hi_i
            if period_like and n > 0:
                hi_eff = min(hi_eff, max(lo_i, min(n, 512)))

            # Sample in a small window around the default.
            candidates = list({default - 2, default - 1, default, default + 1, default + 2})
            candidates = [c for c in candidates if lo_i <= c <= hi_eff]
            if candidates:
                params[kw] = int(rng.choice(candidates))
            else:
                params[kw] = int(np.clip(default, lo_i, hi_eff))

    # Generic constraints for common paired parameters.
    if "fastperiod" in params and "slowperiod" in params:
        fast = int(params["fastperiod"])
        slow = int(params["slowperiod"])
        if fast >= slow:
            slow_max = None
            slow_allowed = allowed.get("slowperiod")
            if slow_allowed is not None:
                slow_max = _bound_to_int(slow_allowed[2])

            slow = max(slow, fast + 1)
            if slow_max is not None:
                slow = min(slow, slow_max)
            params["slowperiod"] = int(slow)
    if "fastlimit" in params and "slowlimit" in params:
        fast = float(params["fastlimit"])
        slow = float(params["slowlimit"])
        if slow > fast:
            params["slowlimit"] = fast
    if "acceleration" in params and "maximum" in params:
        acc = float(params["acceleration"])
        mx = float(params["maximum"])
        if acc > mx:
            params["acceleration"] = mx

    return params


def make_parity_case(func_name: str, n: int, seed: int) -> ParityCase:
    meta = _load_meta()[func_name]
    rng = np.random.default_rng(seed)
    inputs = _make_inputs(meta, n, rng)
    kwargs = _sample_params(func_name, meta, n, rng)
    return ParityCase(func=func_name, inputs=inputs, kwargs=kwargs)


def make_inputs(func_name: str, n: int, seed: int) -> list[np.ndarray]:
    meta = _load_meta()[func_name]
    rng = np.random.default_rng(seed)
    return _make_inputs(meta, n, rng)


def _as_tuple(x: Any) -> tuple[Any, ...]:
    if isinstance(x, tuple):
        return x
    return (x,)


def compare_one(case: ParityCase, rtol: float = 1e-10, atol: float = 1e-10) -> None:
    import talib

    talib_fn = getattr(talib, case.func)
    numb_fn = getattr(numbatalib, case.func)

    ref = talib_fn(*case.inputs, **case.kwargs)
    got = numb_fn(*case.inputs, **case.kwargs)

    ref_t = _as_tuple(ref)
    got_t = _as_tuple(got)
    if len(ref_t) != len(got_t):
        raise AssertionError(f"{case.func}: output arity mismatch {len(got_t)} != {len(ref_t)}")

    for idx, (a, b) in enumerate(zip(got_t, ref_t)):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if a_arr.dtype.kind in ("i", "u"):
            if not np.array_equal(a_arr, b_arr):
                raise AssertionError(f"{case.func}[{idx}]: integer output mismatch")
        else:
            if not np.allclose(a_arr, b_arr, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.nanmax(np.abs(a_arr - b_arr))
                raise AssertionError(f"{case.func}[{idx}]: max abs diff {diff}")


def benchmark_one(
    func_name: str,
    n: int,
    seed: int,
    repeat: int = 5,
) -> dict[str, Any]:
    import talib

    case = make_parity_case(func_name, n=n, seed=seed)
    talib_fn = getattr(talib, func_name)
    numb_fn = getattr(numbatalib, func_name)

    # Warmup numba compilation.
    _ = numb_fn(*case.inputs, **case.kwargs)
    _ = talib_fn(*case.inputs, **case.kwargs)

    def _time(fn: Callable[..., Any]) -> float:
        best = math.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = fn(*case.inputs, **case.kwargs)
            t1 = time.perf_counter()
            best = min(best, t1 - t0)
        return best

    t_talib = _time(talib_fn)
    t_numb = _time(numb_fn)
    ratio = t_numb / t_talib if t_talib > 0 else math.inf

    return {
        "function": func_name,
        "n": n,
        "seed": seed,
        "talib_sec_best": t_talib,
        "numbatalib_sec_best": t_numb,
        "ratio_numbatalib_over_talib": ratio,
        "kwargs": dict(case.kwargs),
    }
