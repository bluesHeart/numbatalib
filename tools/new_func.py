from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
META_PATH = REPO_ROOT / "generated" / "ta_func_meta.json"


def _load_meta() -> dict[str, dict]:
    raw = json.loads(META_PATH.read_text(encoding="utf-8"))
    return {row["name"]: row for row in raw}


def _py_arg_name(in_name: str) -> str:
    mapping = {
        "inOpen": "open",
        "inHigh": "high",
        "inLow": "low",
        "inClose": "close",
        "inVolume": "volume",
        "inReal": "real",
        "inReal0": "real0",
        "inReal1": "real1",
    }
    if in_name in mapping:
        return mapping[in_name]
    return in_name.removeprefix("in")


def _py_kw_name(opt_in: str) -> str:
    if opt_in.startswith("optIn"):
        return opt_in[len("optIn") :].lower()
    return opt_in


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new function implementation stub.")
    parser.add_argument("name", help="TA-Lib function name (e.g., SMA, EMA, MACD).")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Write directly into numbatalib/_func (will be picked up as implemented immediately).",
    )
    args = parser.parse_args()

    name = args.name.strip().upper()
    meta = _load_meta()
    if name not in meta:
        raise SystemExit(f"Unknown function: {name}")

    target_dir = REPO_ROOT / "numbatalib" / ("_func" if args.direct else "_func_drafts")
    module_path = target_dir / f"ta_{name.lower()}.py"
    if module_path.exists():
        raise SystemExit(f"Already exists: {module_path}")

    inputs = meta[name].get("inputs", [])
    opt_inputs = meta[name].get("opt_inputs", [])

    py_inputs = [_py_arg_name(x) for x in inputs]
    py_opts = [_py_kw_name(opt["name"]) + "=None" for opt in opt_inputs]
    args_sig = ", ".join(py_inputs + py_opts)

    upstream_c = meta[name].get("upstream_c_file") or f"src/ta_func/ta_{name}.c"

    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                f"# Upstream reference: upstream/ta-lib/{upstream_c}",
                "",
                f"def {name}({args_sig}):",
                "    raise NotImplementedError",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created {module_path}")
    if not args.direct:
        print(
            "When implemented and passing parity checks, move it to "
            f"{(REPO_ROOT / 'numbatalib' / '_func')}"
        )


if __name__ == "__main__":
    main()
