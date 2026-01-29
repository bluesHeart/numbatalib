from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import numbatalib
from tools.parity_harness import make_inputs


GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
MANIFEST = GOLDEN_DIR / "manifest.json"


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


@pytest.mark.skipif(not MANIFEST.exists(), reason="golden manifest not generated")
def test_golden_outputs() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    implemented = set(numbatalib.implemented_functions())

    cases = [c for c in manifest.get("cases", []) if c.get("function") in implemented]
    if not cases:
        pytest.skip("no golden cases for implemented functions")

    for case in cases:
        func = case["function"]
        n = int(case["n"])
        seed = int(case["seed"])
        kwargs = dict(case.get("kwargs", {}))
        npz_path = GOLDEN_DIR / case["npz"]
        outputs = list(case["outputs"])

        inputs = make_inputs(func, n=n, seed=seed)
        got = getattr(numbatalib, func)(*inputs, **kwargs)
        got_t = _as_tuple(got)

        with np.load(npz_path) as data:
            ref_t = tuple(data[k] for k in outputs)

        assert len(got_t) == len(ref_t)
        for a, b in zip(got_t, ref_t):
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            if a_arr.dtype.kind in ("i", "u"):
                assert np.array_equal(a_arr, b_arr)
            else:
                assert np.allclose(a_arr, b_arr, rtol=1e-12, atol=1e-12, equal_nan=True)

