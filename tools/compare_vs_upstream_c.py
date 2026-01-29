from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import numbatalib  # noqa: E402
from tools.parity_harness import make_parity_case  # noqa: E402
from tools.upstream_c_ref import UpstreamTalibCRef  # noqa: E402


def _as_tuple(x: Any) -> tuple[Any, ...]:
    if isinstance(x, tuple):
        return x
    return (x,)


def _compare_arrays(func: str, got: Any, ref: Any, rtol: float, atol: float) -> None:
    got_t = _as_tuple(got)
    ref_t = _as_tuple(ref)
    if len(got_t) != len(ref_t):
        raise AssertionError(f"{func}: output arity mismatch {len(got_t)} != {len(ref_t)}")

    for idx, (a, b) in enumerate(zip(got_t, ref_t)):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if a_arr.dtype.kind in ("i", "u"):
            if not np.array_equal(a_arr, b_arr):
                raise AssertionError(f"{func}[{idx}]: integer output mismatch")
        else:
            if not np.allclose(a_arr, b_arr, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.nanmax(np.abs(a_arr - b_arr))
                raise AssertionError(f"{func}[{idx}]: max abs diff {diff}")


def _benchmark_one(
    func_name: str,
    ref: UpstreamTalibCRef,
    n: int,
    seed: int,
    repeat: int,
) -> dict[str, Any]:
    case = make_parity_case(func_name, n=n, seed=seed)
    numb_fn = getattr(numbatalib, func_name)
    ref_fn: Callable[..., Any] = getattr(ref, func_name)

    _ = numb_fn(*case.inputs, **case.kwargs)
    _ = ref_fn(*case.inputs, **case.kwargs)

    def _time(fn: Callable[..., Any]) -> float:
        best = math.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = fn(*case.inputs, **case.kwargs)
            t1 = time.perf_counter()
            best = min(best, t1 - t0)
        return best

    t_ref = _time(ref_fn)
    t_numb = _time(numb_fn)
    ratio = t_numb / t_ref if t_ref > 0 else math.inf

    return {
        "function": func_name,
        "n": n,
        "seed": seed,
        "ref_sec_best": t_ref,
        "numbatalib_sec_best": t_numb,
        "ratio_numbatalib_over_ref": ratio,
        "kwargs": dict(case.kwargs),
    }


def _update_checklist(
    checklist_path: Path,
    parity: dict[str, str],
    speed_ratio: dict[str, float],
    note: str,
) -> None:
    rows: list[dict[str, Any]] = []
    with checklist_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            name = row.get("name", "")
            if name in parity:
                row["parity"] = parity[name]
                if note:
                    existing = (row.get("notes") or "").strip()
                    parts = [p.strip() for p in existing.split(";") if p.strip()]
                    if note not in parts:
                        parts.append(note)
                    row["notes"] = "; ".join(parts)
            if name in speed_ratio:
                row["speed_ratio_vs_talib"] = f"{speed_ratio[name]:.6f}"
            rows.append(row)

    with checklist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare numbatalib vs upstream TA-Lib C (ctypes).")
    parser.add_argument(
        "--functions",
        default="ACCBANDS,AVGDEV,IMI",
        help="Comma-separated function names.",
    )
    parser.add_argument("--cases", type=int, default=10, help="Parity cases per function.")
    parser.add_argument("--sizes", default="2,3,7,30000", help="Comma-separated input sizes.")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    parser.add_argument("--rtol", type=float, default=1e-12, help="Relative tolerance (np.allclose).")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance (np.allclose).")
    parser.add_argument("--bench", action="store_true", help="Run speed benchmarks vs upstream C ref.")
    parser.add_argument("--bench-n", type=int, default=30000, help="Benchmark vector size.")
    parser.add_argument("--bench-repeat", type=int, default=5, help="Best-of-N repeats.")
    parser.add_argument(
        "--write-checklist",
        action="store_true",
        help="Update port_checklist.csv parity + speed columns for tested functions.",
    )
    args = parser.parse_args()

    funcs = [f.strip() for f in args.functions.split(",") if f.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    ref = UpstreamTalibCRef()

    parity_rows: list[dict[str, Any]] = []
    parity_status: dict[str, str] = {}
    speed_ratio: dict[str, float] = {}
    bench_rows: dict[str, dict[str, Any]] = {}

    for func in funcs:
        if func not in numbatalib.available_functions():
            raise SystemExit(f"Unknown function: {func}")
        if not hasattr(ref, func):
            raise SystemExit(f"Missing C reference binding for: {func}")

        failures = 0
        last_error = ""
        total = 0
        for case_idx in range(args.cases):
            for n in sizes:
                seed_bytes = f"{func}:{case_idx}:{n}:{args.seed}".encode("utf-8")
                seed = args.seed + (zlib.crc32(seed_bytes) & 0xFFFF_FFFF)
                case = make_parity_case(func, n=n, seed=seed)
                total += 1
                try:
                    got = getattr(numbatalib, func)(*case.inputs, **case.kwargs)
                    ref_out = getattr(ref, func)(*case.inputs, **case.kwargs)
                    _compare_arrays(func, got, ref_out, rtol=args.rtol, atol=args.atol)
                    parity_rows.append(
                        {
                            "function": func,
                            "n": n,
                            "seed": seed,
                            "ok": True,
                            "error": "",
                            "kwargs": repr(case.kwargs),
                        }
                    )
                except Exception as e:  # pragma: no cover
                    failures += 1
                    last_error = str(e)
                    parity_rows.append(
                        {
                            "function": func,
                            "n": n,
                            "seed": seed,
                            "ok": False,
                            "error": last_error,
                            "kwargs": repr(case.kwargs),
                        }
                    )

        parity_status[func] = "PASS" if failures == 0 else "FAIL"
        print(f"[parity] {func}: {total - failures}/{total} passed")
        if failures:
            print(f"  last error: {last_error}")

        if args.bench:
            bench = _benchmark_one(func, ref=ref, n=args.bench_n, seed=args.seed, repeat=args.bench_repeat)
            bench_rows[func] = bench
            speed_ratio[func] = float(bench["ratio_numbatalib_over_ref"])
            print(
                f"[bench] {func}: ref {bench['ref_sec_best']:.6f}s, "
                f"numbatalib {bench['numbatalib_sec_best']:.6f}s, "
                f"ratio {bench['ratio_numbatalib_over_ref']:.3f}x"
            )

    out_dir = REPO_ROOT / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parity = out_dir / "parity_results_upstream_c.csv"
    with out_parity.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["function", "n", "seed", "ok", "error", "kwargs"]
        )
        writer.writeheader()
        writer.writerows(parity_rows)
    print(f"Wrote {out_parity}")

    if args.bench:
        out_bench = out_dir / "bench_results_upstream_c.csv"
        with out_bench.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "function",
                    "n",
                    "seed",
                    "ref_sec_best",
                    "numbatalib_sec_best",
                    "ratio_numbatalib_over_ref",
                    "kwargs",
                ],
            )
            writer.writeheader()
            for func in funcs:
                if func not in bench_rows:
                    continue
                bench = bench_rows[func]
                writer.writerow(
                    {
                        "function": func,
                        "n": bench["n"],
                        "seed": bench["seed"],
                        "ref_sec_best": bench["ref_sec_best"],
                        "numbatalib_sec_best": bench["numbatalib_sec_best"],
                        "ratio_numbatalib_over_ref": bench["ratio_numbatalib_over_ref"],
                        "kwargs": repr(bench["kwargs"]),
                    }
                )
        print(f"Wrote {out_bench}")

    if args.write_checklist:
        checklist = REPO_ROOT / "port_checklist.csv"
        _update_checklist(checklist, parity_status, speed_ratio, note="ref=upstream_c(v0.6.4)")
        print(f"Updated {checklist}")


if __name__ == "__main__":
    main()
