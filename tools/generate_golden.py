from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numbatalib  # noqa: E402
from tools.parity_harness import make_inputs, make_parity_case  # noqa: E402




def _as_tuple(x: Any) -> tuple[Any, ...]:
    if isinstance(x, tuple):
        return x
    return (x,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden outputs using installed talib.")
    parser.add_argument(
        "--functions",
        default="",
        help="Comma-separated function names (default: implemented functions).",
    )
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds.")
    parser.add_argument("--sizes", default="256,4096", help="Comma-separated sizes.")
    args = parser.parse_args()

    try:
        import talib
    except Exception as e:
        raise SystemExit("talib is required to generate golden data.") from e

    funcs = [f.strip() for f in args.functions.split(",") if f.strip()]
    if not funcs:
        funcs = numbatalib.implemented_functions()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    golden_dir = REPO_ROOT / "tests" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"version": 1, "cases": []}

    for func in funcs:
        if func not in numbatalib.available_functions():
            raise SystemExit(f"Unknown function: {func}")

        if not hasattr(talib, func):
            print(f"[skip] {func}: not available in installed talib")
            continue

        func_npz = golden_dir / f"{func}.npz"
        arrays: dict[str, np.ndarray] = {}
        case_id = 0

        talib_fn = getattr(talib, func)

        for seed in seeds:
            for n in sizes:
                # Generate deterministic inputs, and deterministic kwargs (store them).
                sampled = make_parity_case(func, n=n, seed=seed)
                inputs = make_inputs(func, n=n, seed=seed)
                kwargs = dict(sampled.kwargs)

                ref = talib_fn(*inputs, **kwargs)
                ref_t = _as_tuple(ref)

                output_keys: list[str] = []
                for out_idx, arr in enumerate(ref_t):
                    k = f"case{case_id}_out{out_idx}"
                    arrays[k] = np.asarray(arr)
                    output_keys.append(k)

                manifest["cases"].append(
                    {
                        "function": func,
                        "case_id": case_id,
                        "n": n,
                        "seed": seed,
                        "kwargs": kwargs,
                        "npz": func_npz.name,
                        "outputs": output_keys,
                    }
                )
                case_id += 1

        np.savez_compressed(func_npz, **arrays)
        print(f"Wrote {func_npz} ({len(arrays)} arrays)")

    manifest_path = golden_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(manifest['cases'])} cases)")


if __name__ == "__main__":
    main()
