# numbatalib

Reimplement TA-Lib (technical analysis indicators) in **pure Python + Numba**, aiming for:

- algorithmic parity with the local installed `talib` (ground-truth reference for tests/benchmarks)
- near-C performance for large vectors (Numba JIT)
- no compiled extension at install time (no `libta-lib` dependency)

## Status

- Implemented TA-Lib Core functions: **161 / 161**
- Parity tests: `generated/pytest_results.txt`
- Parity + benchmark results: `generated/parity_results.csv`, `generated/bench_results.csv`
- Status tracker: `port_checklist.csv`

Note: the local `talib` wheel in this environment does not expose `ACCBANDS`, `AVGDEV`, `IMI`; those are validated against upstream TA-Lib C (see `generated/parity_results_upstream_c.csv`).

## Install

After publishing to PyPI:

```bash
pip install numbatalib
```

## Usage

`numbatalib` exposes TA-Lib function names dynamically:

```python
import numpy as np
import numbatalib as ta

x = np.random.default_rng(0).normal(size=1000).cumsum()

sma = ta.SMA(x, timeperiod=20)
rsi = ta.RSI(x, timeperiod=14)
```

## TA-Lib compatible API (minimal habit cost)

If you want `talib`-like **APIs + error messages**, use the compatibility shim:

```python
import numpy as np
import numbatalib.talib as talib  # or: import numbatalib.compat.talib as talib

x = np.random.default_rng(0).normal(size=1000).cumsum()

talib.SMA(x, timeperiod=20)
talib.stream.SMA(x, timeperiod=20)        # streaming scalar
from numbatalib.talib import abstract
abstract.Function("SMA")(x, timeperiod=20)
```

Notes:
- `set_compatibility/get_compatibility` and `set_unstable_period/get_unstable_period` are supported (matching TA-Lib behavior for EMA/RSI/CMO and unstable-period masking).

## Dev

- Run parity tests vs installed `talib`: `pytest -q`
- Regenerate parity + speed CSVs and update checklist: `python tools/compare_vs_talib.py --bench --write-checklist`
