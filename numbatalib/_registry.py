from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


@dataclass(frozen=True)
class FunctionMeta:
    name: str
    group_id: int
    group_name: str
    inputs: list[str]
    outputs: list[str]
    opt_inputs: list[dict[str, Any]]
    lookback_args: list[dict[str, Any]]


def _package_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_meta() -> dict[str, FunctionMeta]:
    meta_path = _package_root() / "_generated" / "ta_func_meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            "Missing generated metadata file. "
            "Run `python tools/generate_checklist.py` to (re)generate it."
        )

    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    out: dict[str, FunctionMeta] = {}
    for row in raw:
        out[row["name"]] = FunctionMeta(
            name=row["name"],
            group_id=row["group_id"],
            group_name=row["group_name"],
            inputs=row.get("inputs", []),
            outputs=row.get("outputs", []),
            opt_inputs=row.get("opt_inputs", []),
            lookback_args=row.get("lookback_args", []),
        )
    return out


@lru_cache(maxsize=1)
def _discover_impl_modules() -> dict[str, str]:
    """
    Discover implemented functions by scanning `numbatalib/_func/ta_*.py`.

    Convention:
      - file name: ta_<lowercase function name>.py (underscores preserved)
      - function defined inside module: <UPPERCASE_FUNCTION_NAME>
    """
    func_dir = _package_root() / "_func"
    mapping: dict[str, str] = {}
    for path in func_dir.glob("ta_*.py"):
        if path.name == "__init__.py":
            continue
        func_name = path.stem[len("ta_") :].upper()
        mapping[func_name] = f"numbatalib._func.{path.stem}"
    return mapping


def available_functions() -> list[str]:
    return sorted(_load_meta().keys())


def implemented_functions() -> list[str]:
    all_funcs = set(_load_meta().keys())
    implemented = set(_discover_impl_modules().keys())
    return sorted(all_funcs.intersection(implemented))


def _load_module(dotted: str) -> ModuleType:
    return importlib.import_module(dotted)


def _make_stub(name: str) -> Callable[..., Any]:
    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{name} is not implemented yet in numbatalib. "
            "See `port_checklist.csv` for progress."
        )

    _stub.__name__ = name
    return _stub


def get_function(name: str) -> Callable[..., Any] | None:
    """
    Return a Python-callable indicator function by name (e.g. "SMA").

    If the name matches a known TA-Lib function but is not implemented yet,
    returns a stub that raises NotImplementedError.
    """
    meta = _load_meta()
    if name not in meta:
        return None

    impl_modules = _discover_impl_modules()
    module_name = impl_modules.get(name)
    if module_name is None:
        return _make_stub(name)

    mod = _load_module(module_name)
    fn = getattr(mod, name, None)
    if fn is None:
        raise RuntimeError(f"Module {module_name} does not define {name}")
    return fn

