from __future__ import annotations

import argparse
import csv
import sys
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numbatalib  # noqa: E402
from tools.parity_harness import benchmark_one, compare_one, make_parity_case  # noqa: E402



def _update_checklist(
    checklist_path: Path, parity: dict[str, str], speed_ratio: dict[str, float]
) -> None:
    implemented = set(numbatalib.implemented_functions())
    rows: list[dict[str, Any]] = []
    with checklist_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            name = row.get("name", "")
            if name in implemented:
                row["status"] = "IMPLEMENTED"
            if name in parity:
                row["parity"] = parity[name]
            if name in speed_ratio:
                row["speed_ratio_vs_talib"] = f"{speed_ratio[name]:.6f}"
            rows.append(row)

    with checklist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare numbatalib vs installed talib.")
    parser.add_argument(
        "--functions",
        default="",
        help="Comma-separated function names (default: implemented functions).",
    )
    parser.add_argument("--cases", type=int, default=5, help="Parity cases per function.")
    parser.add_argument(
        "--sizes",
        default="3,7,30000",
        help="Comma-separated input sizes for parity checks.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for float outputs (np.allclose).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for float outputs (np.allclose).",
    )
    parser.add_argument("--bench", action="store_true", help="Run speed benchmarks.")
    parser.add_argument("--bench-n", type=int, default=30000, help="Benchmark vector size.")
    parser.add_argument("--bench-repeat", type=int, default=5, help="Best-of-N repeats.")
    parser.add_argument(
        "--write-checklist",
        action="store_true",
        help="Update port_checklist.csv parity + speed columns for tested functions.",
    )
    args = parser.parse_args()

    try:
        import talib  # noqa: F401
    except Exception as e:
        raise SystemExit("talib is required for this script. Install it in this environment.") from e

    funcs = [f.strip() for f in args.functions.split(",") if f.strip()]
    if not funcs:
        funcs = numbatalib.implemented_functions()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    parity_rows: list[dict[str, Any]] = []
    parity_status: dict[str, str] = {}
    speed_ratio: dict[str, float] = {}
    bench_rows: dict[str, dict[str, Any]] = {}

    for func in funcs:
        if func not in numbatalib.available_functions():
            raise SystemExit(f"Unknown function: {func}")

        if not hasattr(__import__("talib"), func):
            parity_status[func] = "NO_TALIB_REF"
            print(f"[skip] {func}: not available in installed talib")
            continue

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
                    compare_one(case, rtol=args.rtol, atol=args.atol)
                    parity_rows.append(
                        {"function": func, "n": n, "seed": seed, "ok": True, "error": ""}
                    )
                except Exception as e:  # pragma: no cover
                    failures += 1
                    last_error = str(e)
                    parity_rows.append(
                        {"function": func, "n": n, "seed": seed, "ok": False, "error": last_error}
                    )

        parity_status[func] = "PASS" if failures == 0 else "FAIL"
        print(f"[parity] {func}: {total - failures}/{total} passed")
        if failures:
            print(f"  last error: {last_error}")

        if args.bench:
            bench = benchmark_one(func, n=args.bench_n, seed=args.seed, repeat=args.bench_repeat)
            bench_rows[func] = bench
            speed_ratio[func] = float(bench["ratio_numbatalib_over_talib"])
            print(
                f"[bench] {func}: talib {bench['talib_sec_best']:.6f}s, "
                f"numbatalib {bench['numbatalib_sec_best']:.6f}s, "
                f"ratio {bench['ratio_numbatalib_over_talib']:.3f}x"
            )

    out_dir = REPO_ROOT / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parity = out_dir / "parity_results.csv"
    with out_parity.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["function", "n", "seed", "ok", "error"])
        writer.writeheader()
        writer.writerows(parity_rows)
    print(f"Wrote {out_parity}")

    if args.bench:
        out_bench = out_dir / "bench_results.csv"
        with out_bench.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "function",
                    "n",
                    "seed",
                    "talib_sec_best",
                    "numbatalib_sec_best",
                    "ratio_numbatalib_over_talib",
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
                        "talib_sec_best": bench["talib_sec_best"],
                        "numbatalib_sec_best": bench["numbatalib_sec_best"],
                        "ratio_numbatalib_over_talib": bench["ratio_numbatalib_over_talib"],
                    }
                )
        print(f"Wrote {out_bench}")

    if args.write_checklist:
        checklist = REPO_ROOT / "port_checklist.csv"
        _update_checklist(checklist, parity_status, speed_ratio)
        print(f"Updated {checklist}")


if __name__ == "__main__":
    main()
