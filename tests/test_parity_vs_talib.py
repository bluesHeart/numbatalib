from __future__ import annotations

import pytest

import numbatalib
from tools.parity_harness import compare_one, make_parity_case


talib = pytest.importorskip("talib")
_TALIB_FUNCS = set(dir(talib))


@pytest.mark.parametrize(
    "func_name", [f for f in numbatalib.implemented_functions() if f in _TALIB_FUNCS]
)
@pytest.mark.parametrize("n", [128, 1024, 8192])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_parity_against_talib(func_name: str, n: int, seed: int) -> None:
    case = make_parity_case(func_name, n=n, seed=seed)
    compare_one(case)
