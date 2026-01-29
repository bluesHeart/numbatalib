from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
from numba import njit

import numbatalib
from numbatalib._registry import _load_meta

TA_INTEGER_DEFAULT = -2147483648
TA_REAL_DEFAULT = -4e37
TA_EPSILON = 1e-14

__ta_version__ = b"numbatalib (python+numba)"


class MA_Type:
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8


TA_FUNC_FLAGS = {
    16777216: "Output scale same as input",
    67108864: "Output is over volume",
    134217728: "Function has an unstable period",
    268435456: "Output is a candlestick",
}

TA_INPUT_FLAGS = {
    1: "open",
    2: "high",
    4: "low",
    8: "close",
    16: "volume",
    32: "open interest",
    2147483648: "real",
}

TA_OUTPUT_FLAGS = {
    1: "Line",
    2: "Dotted Line",
    4: "Dashed Line",
    8: "Dot",
    16: "Histogram",
    32: "Pattern (Bool)",
    64: "Bull/Bear Pattern (Bearish < 0, Neutral = 0, Bullish > 0)",
    128: "Strength Pattern ([-200..-100] = Bearish, [-100..0] = Getting Bearish, 0 = Neutral, [0..100] = Getting Bullish, [100-200] = Bullish)",
    256: "Positive Trade",
    512: "Negative Trade",
    1024: "Limit",
    2048: "Stop",
    4096: "Open Interest",
}


__TA_FUNCTION_NAMES__ = tuple(numbatalib.implemented_functions())

_META = _load_meta()

_compatibility: int = 0

_UNSTABLE_FUNCS = {
    "ADX",
    "ADXR",
    "ATR",
    "CMO",
    "DX",
    "EMA",
    "HT_DCPERIOD",
    "HT_DCPHASE",
    "HT_PHASOR",
    "HT_SINE",
    "HT_TRENDLINE",
    "HT_TRENDMODE",
    "IMI",
    "KAMA",
    "MAMA",
    "MFI",
    "MINUS_DI",
    "MINUS_DM",
    "NATR",
    "PLUS_DI",
    "PLUS_DM",
    "RSI",
    "STOCHRSI",
    "T3",
}
_unstable: dict[str, int] = {name: 0 for name in _UNSTABLE_FUNCS}


def _ta_initialize() -> None:
    return None


def _ta_shutdown() -> None:
    return None


def _ta_set_compatibility(value: int) -> None:
    global _compatibility
    _compatibility = int(value)


def _ta_get_compatibility() -> int:
    return int(_compatibility)


def _ta_set_unstable_period(func_name: str, period: int) -> None:
    key = func_name.upper()
    if key not in _UNSTABLE_FUNCS:
        raise KeyError(func_name)
    _unstable[key] = int(period)


def _ta_get_unstable_period(func_name: str) -> int:
    key = func_name.upper()
    if key not in _UNSTABLE_FUNCS:
        raise KeyError(func_name)
    return int(_unstable.get(key, 0))


def _raise_bad_param(func_name: str) -> None:
    raise Exception(f"TA_{func_name} function failed with error code 2: Bad Parameter (TA_BAD_PARAM)")


def _as_1d_f64(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise Exception("input array has wrong dimensions")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _normalize_inputs(raw_inputs: list[Any]) -> list[np.ndarray]:
    arrays = [_as_1d_f64(x) for x in raw_inputs]
    if not arrays:
        return arrays
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise Exception("input array lengths are different")
    return arrays


def _optin_to_kw(optin_name: str) -> str:
    if not optin_name.startswith("optIn"):
        return optin_name
    return optin_name[len("optIn") :].lower()


def _default_kwargs(func_name: str) -> dict[str, Any]:
    fn = getattr(numbatalib, func_name)
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            continue
        out[p.name] = p.default
    return out


_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {name: _default_kwargs(name) for name in __TA_FUNCTION_NAMES__}


def _coerce_and_clean_params(func_name: str, params: dict[str, Any]) -> dict[str, Any]:
    meta = _META[func_name]
    out: dict[str, Any] = {}
    for opt in meta.opt_inputs:
        kw = _optin_to_kw(opt["name"])
        if kw not in params:
            continue

        v = params[kw]
        c_type = opt.get("c_type", "")

        if c_type == "double":
            if not isinstance(v, (int, float, np.number, bool)):
                raise TypeError(f"must be real number, not {type(v).__name__}")
            v_f = float(v)
            if v_f == float(TA_REAL_DEFAULT):
                continue
            out[kw] = v_f
            continue

        if not isinstance(v, (int, float, np.number, bool)):
            raise TypeError("an integer is required")
        v_i = int(v)
        if v_i == int(TA_INTEGER_DEFAULT):
            continue
        out[kw] = v_i

    return out


@njit(cache=True)
def _ema_metastock_kernel(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    n = real.shape[0]
    if timeperiod > n:
        return

    k = 2.0 / (timeperiod + 1.0)
    start_idx = timeperiod - 1

    prev = real[0]
    for i in range(1, start_idx + 1):
        prev = ((real[i] - prev) * k) + prev

    out[start_idx] = prev
    for i in range(start_idx + 1, n):
        prev = ((real[i] - prev) * k) + prev
        out[i] = prev


def _ema_metastock(real: np.ndarray, timeperiod: int) -> np.ndarray:
    out = np.full(real.shape[0], np.nan, dtype=np.float64)
    _ema_metastock_kernel(real, int(timeperiod), out)
    return out


def _maybe_add_metastock_first_rsi(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    if timeperiod < 2:
        return
    if real.shape[0] < timeperiod:
        return

    prev_value = float(real[0])
    prev_gain = 0.0
    prev_loss = 0.0
    today = 0
    for _ in range(int(timeperiod)):
        v = float(real[today])
        today += 1
        diff = v - prev_value
        prev_value = v
        if diff < 0.0:
            prev_loss -= diff
        else:
            prev_gain += diff

    avg_loss = prev_loss / timeperiod
    avg_gain = prev_gain / timeperiod
    denom = avg_gain + avg_loss
    out[timeperiod - 1] = 100.0 * (avg_gain / denom) if abs(denom) >= TA_EPSILON else 0.0


def _maybe_add_metastock_first_cmo(real: np.ndarray, timeperiod: int, out: np.ndarray) -> None:
    if timeperiod < 2:
        return
    if real.shape[0] < timeperiod:
        return

    prev_value = float(real[0])
    prev_gain = 0.0
    prev_loss = 0.0
    today = 0
    for _ in range(int(timeperiod)):
        v = float(real[today])
        today += 1
        diff = v - prev_value
        prev_value = v
        if diff < 0.0:
            prev_loss -= diff
        else:
            prev_gain += diff

    avg_loss = prev_loss / timeperiod
    avg_gain = prev_gain / timeperiod
    num = avg_gain - avg_loss
    denom = avg_gain + avg_loss
    out[timeperiod - 1] = 100.0 * (num / denom) if abs(denom) >= TA_EPSILON else 0.0


def _first_non_nan_idx(arr: np.ndarray) -> int | None:
    if arr.dtype.kind != "f":
        return None
    idx = np.flatnonzero(~np.isnan(arr))
    if idx.size == 0:
        return None
    return int(idx[0])


def _apply_unstable(result: Any, unstable: int) -> Any:
    if unstable <= 0:
        return result

    outs = result if isinstance(result, tuple) else (result,)
    base_idx = _first_non_nan_idx(np.asarray(outs[0]))
    if base_idx is None:
        return result

    cutoff = base_idx + int(unstable)
    for out in outs:
        a = np.asarray(out)
        if a.dtype.kind != "f":
            continue
        end = min(a.shape[0], cutoff)
        if end > 0:
            a[:end] = np.nan

    return result


def _call_func(func_name: str, raw_inputs: list[Any], raw_params: dict[str, Any]) -> Any:
    inputs = _normalize_inputs(raw_inputs)

    defaults = _DEFAULT_KWARGS.get(func_name, {})
    params = dict(defaults)
    params.update(raw_params)

    try:
        kwargs = _coerce_and_clean_params(func_name, params)
    except TypeError:
        raise

    if func_name in _UNSTABLE_FUNCS:
        unstable = _unstable.get(func_name, 0)
    else:
        unstable = 0

    try:
        if func_name == "EMA" and _compatibility != 0:
            tp = int(kwargs.get("timeperiod", defaults.get("timeperiod", 30)))
            result = _ema_metastock(inputs[0], tp)
        else:
            fn = getattr(numbatalib, func_name)
            result = fn(*inputs, **kwargs)
    except ValueError:
        _raise_bad_param(func_name)

    if func_name == "RSI" and _compatibility == 1:
        out_arr = np.asarray(result)
        tp = int(kwargs.get("timeperiod", defaults.get("timeperiod", 14)))
        if out_arr.dtype.kind == "f" and out_arr.shape[0] == inputs[0].shape[0]:
            _maybe_add_metastock_first_rsi(inputs[0], tp, out_arr)
        result = out_arr

    if func_name == "CMO" and _compatibility == 1:
        out_arr = np.asarray(result)
        tp = int(kwargs.get("timeperiod", defaults.get("timeperiod", 14)))
        if out_arr.dtype.kind == "f" and out_arr.shape[0] == inputs[0].shape[0]:
            _maybe_add_metastock_first_cmo(inputs[0], tp, out_arr)
        result = out_arr

    result = _apply_unstable(result, unstable)
    return result


def _stream_result(result: Any) -> Any:
    if isinstance(result, tuple):
        return tuple(np.asarray(x)[-1].item() for x in result)
    return np.asarray(result)[-1].item()


def _call_stream(func_name: str, raw_inputs: list[Any], raw_params: dict[str, Any]) -> Any:
    return _stream_result(_call_func(func_name, raw_inputs, raw_params))


def _display_name_for(func_name: str) -> str:
    fn = getattr(numbatalib, func_name)
    doc = (fn.__doc__ or "").strip()
    if not doc:
        return func_name
    return doc.splitlines()[0].strip() or func_name


def _abstract_input_names(func_name: str) -> OrderedDict[str, Any]:
    meta = _META[func_name]
    prices = [n[len("in") :].lower() for n in meta.inputs if n.startswith("in")]
    if prices and all(p in {"open", "high", "low", "close", "volume"} for p in prices):
        if len(prices) == 1:
            return OrderedDict([("price", prices[0])])
        return OrderedDict([("prices", prices)])

    if len(meta.inputs) == 1:
        return OrderedDict([("price", "close")])
    if len(meta.inputs) == 2:
        return OrderedDict([("price0", "high"), ("price1", "low")])
    return OrderedDict([("prices", ["close"] * len(meta.inputs))])


def _abstract_output_names(func_name: str) -> list[str]:
    meta = _META[func_name]
    out_names: list[str] = []
    for out in meta.outputs:
        if out == "outReal":
            out_names.append("real")
            continue
        if out == "outInteger":
            out_names.append("integer")
            continue
        name = out
        for prefix in ("outReal", "outInteger", "out"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        out_names.append(name.lower() if name else "real")
    return out_names


def _get_defaults_and_docs(func_info: dict[str, Any]) -> tuple[dict[str, Any], str]:
    defaults = dict(func_info.get("parameters", {}))
    name = str(func_info.get("name", ""))
    group = str(func_info.get("group", ""))
    display = str(func_info.get("display_name", name))
    inputs = func_info.get("input_names", {})
    outputs = func_info.get("output_names", [])

    params_part = ", ".join(f"{k}={v}" for k, v in defaults.items())
    hdr = f"{name}([input_arrays]{', [' + params_part + ']' if params_part else ''})"
    lines = [
        hdr,
        "",
        f"{display} ({group})".strip(),
        "",
        "Inputs:",
    ]
    for k in inputs.keys():
        lines.append(f"    {k}: (any ndarray)")
    lines.append("Parameters:")
    for k, v in defaults.items():
        lines.append(f"    {k}: {v}")
    lines.append("Outputs:")
    for o in outputs:
        lines.append(f"    {o}")
    return defaults, "\n".join(lines)


class Function:
    def __init__(self, func_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any):
        del args, kwargs

        name = func_name.upper()
        if name not in _META:
            raise Exception(f"{name} not supported by TA-LIB.")

        self.__name = name
        self.__namestr = name
        self._func = func
        self._meta = _META[name]

        self.__input_price_series_names = _abstract_input_names(name)
        self.parameters = OrderedDict(_DEFAULT_KWARGS.get(name, {}))
        self.output_names = _abstract_output_names(name)
        self.info = {
            "name": name,
            "group": self._meta.group_name,
            "display_name": _display_name_for(name),
            "function_flags": [],
            "input_names": self.__input_price_series_names.copy(),
            "parameters": self.parameters.copy(),
            "output_flags": OrderedDict((o, []) for o in self.output_names),
            "output_names": list(self.output_names),
        }

    @property
    def input_names(self):
        return self.__input_price_series_names

    def get_input_names(self):
        return self.__input_price_series_names.copy()

    def set_input_names(self, input_names):
        if not isinstance(input_names, dict):
            raise TypeError("input_names must be a dict")
        self.__input_price_series_names = OrderedDict(input_names)
        self.info["input_names"] = self.__input_price_series_names.copy()

    def get_parameters(self):
        return self.parameters.copy()

    def set_parameters(self, parameters=None, **kwargs):
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise TypeError("parameters must be a dict")
            for k, v in parameters.items():
                self.parameters[k] = v
        for k, v in kwargs.items():
            self.parameters[k] = v
        self.info["parameters"] = self.parameters.copy()

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            if len(args) != 1:
                raise TypeError("Too many price arguments")
            input_arrays = args[0]
            price_args: tuple[Any, ...] = ()
        else:
            input_arrays = None
            price_args = args

        params = dict(self.parameters)
        params.update(kwargs)

        expected = list(self.__input_price_series_names.values())
        flat_expected: list[str] = []
        for v in expected:
            if isinstance(v, list):
                flat_expected.extend(v)
            else:
                flat_expected.append(v)

        if input_arrays is not None:
            for k in flat_expected:
                if k not in input_arrays:
                    raise Exception(f"input_arrays parameter missing required data key: {k}")
            inputs = [input_arrays[k] for k in flat_expected]
        else:
            if len(price_args) < len(flat_expected):
                exp = ", ".join(flat_expected)
                raise TypeError(f"Not enough price arguments: expected {len(flat_expected)} ({exp})")
            if len(price_args) > len(flat_expected):
                exp = ", ".join(flat_expected)
                raise TypeError(f"Too many price arguments: expected {len(flat_expected)} ({exp})")
            inputs = list(price_args)

        return _call_func(self.__name, inputs, params)

    def __repr__(self) -> str:
        _, docs = _get_defaults_and_docs(self.info)
        return docs


def _generate_wrappers() -> None:
    g = globals()
    for func_name in __TA_FUNCTION_NAMES__:
        meta = _META[func_name]

        in_params = [
            n[len("in") :].lower() if n.startswith("in") else n.lower() for n in meta.inputs
        ]
        opt_params: list[tuple[str, str]] = []
        for opt in meta.opt_inputs:
            kw = _optin_to_kw(opt["name"])
            c_type = opt.get("c_type", "")
            if c_type == "double":
                default = "TA_REAL_DEFAULT"
            elif c_type == "TA_MAType":
                default = "0"
            else:
                default = "TA_INTEGER_DEFAULT"
            opt_params.append((kw, default))

        sig_parts = list(in_params) + [f"{k}={d}" for k, d in opt_params]
        sig = ", ".join(sig_parts)

        call_inputs = ", ".join(in_params)
        call_kwargs = ", ".join(f"'{k}': {k}" for k, _ in opt_params)
        call_kwargs = "{" + call_kwargs + "}" if call_kwargs else "{}"

        src = (
            f"def {func_name}({sig}):\n"
            f"    return _call_func('{func_name}', [{call_inputs}], {call_kwargs})\n"
        )
        exec(src, g, g)

        s_src = (
            f"def stream_{func_name}({sig}):\n"
            f"    return _call_stream('{func_name}', [{call_inputs}], {call_kwargs})\n"
        )
        exec(s_src, g, g)


_generate_wrappers()

