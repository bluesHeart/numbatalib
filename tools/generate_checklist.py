from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "upstream" / "ta-lib"

GROUP_ID_TO_NAME = {
    0: "Math Operators",
    1: "Math Transform",
    2: "Overlap Studies",
    3: "Volatility Indicators",
    4: "Momentum Indicators",
    5: "Cycle Indicators",
    6: "Volume Indicators",
    7: "Pattern Recognition",
    8: "Statistic Functions",
    9: "Price Transform",
}


@dataclass(frozen=True)
class OptParam:
    name: str
    c_type: str
    range_from: str | None = None
    range_to: str | None = None


@dataclass(frozen=True)
class FuncMeta:
    name: str  # e.g. "SMA"
    group_id: int
    group_name: str
    c_file: str | None
    inputs: list[str]
    outputs: list[str]
    opt_inputs: list[OptParam]
    lookback_args: list[OptParam]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_group_idx(path: Path) -> dict[str, int]:
    """
    Returns mapping of base function name (e.g. "SMA") -> group_id (0..9)
    """
    text = _read_text(path)
    group_map: dict[str, int] = {}

    for match in re.finditer(
        r"const\s+TA_FuncDef\s+\*TA_PerGroupFunc_(\d+)\[\]\s*=\s*\{(.*?)\bNULL\s*\};",
        text,
        flags=re.DOTALL,
    ):
        group_id = int(match.group(1))
        body = match.group(2)
        for func_m in re.finditer(r"&TA_DEF_([A-Z0-9_]+)", body):
            name = func_m.group(1)
            if name in group_map and group_map[name] != group_id:
                raise RuntimeError(f"Function {name} appears in multiple groups")
            group_map[name] = group_id

    if not group_map:
        raise RuntimeError("Failed to parse any groups from ta_group_idx.c")
    return group_map


def _parse_decl_line(line: str) -> tuple[str, str, str | None]:
    """
    Parse a single parameter declaration line from ta_func.h.

    Returns: (c_type, name, comment)
    """
    line = line.strip()
    if not line:
        raise ValueError("empty line")

    comment = None
    if "/*" in line:
        before, after = line.split("/*", 1)
        comment = "/*" + after
        line = before.rstrip()

    line = line.rstrip(",")
    line = line.rstrip(");").rstrip(")")
    line = line.strip()

    # Remove extra spaces around pointers.
    line = line.replace("* ", "*").replace(" *", "*")

    # Tokenize. Last token contains the name (possibly with [] or * prefix).
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"cannot parse decl line: {line!r}")

    raw_name = parts[-1]
    c_type = " ".join(parts[:-1])

    # Normalize name by stripping leading pointer and trailing array brackets.
    name = raw_name.lstrip("*")
    name = name.removesuffix("[]")

    return c_type, name, comment


def _parse_range(comment: str | None) -> tuple[str | None, str | None]:
    if not comment:
        return None, None
    m = re.search(r"From\s+([^\s]+)\s+to\s+([^\s]+)", comment)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_ta_func_h(path: Path) -> dict[str, dict[str, Any]]:
    """
    Extract inputs/outputs/optInputs/lookbackArgs for each base function.

    Returns mapping name -> dict metadata.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    metas: dict[str, dict[str, Any]] = {}

    start_re = re.compile(r"^\s*TA_LIB_API\s+TA_RetCode\s+TA_(?!S_)([A-Z0-9_]+)\s*\(")
    lookback_re = re.compile(r"^\s*TA_LIB_API\s+int\s+TA_([A-Z0-9_]+)_Lookback\s*\(")

    while i < len(lines):
        line = lines[i]
        m = start_re.match(line)
        if m:
            name = m.group(1)
            # Collect prototype lines up to and including the line containing ");"
            proto: list[str] = [line]
            if ");" not in line:
                i += 1
                while i < len(lines):
                    proto.append(lines[i])
                    if ");" in lines[i]:
                        break
                    i += 1

            inputs: list[str] = []
            outputs: list[str] = []
            opt_inputs: list[OptParam] = []

            for pline in proto[1:]:
                pline = pline.strip()
                if not pline or pline in ("(", ")", ");"):
                    continue
                if pline.startswith("int    startIdx") or pline.startswith("int startIdx"):
                    continue
                if pline.startswith("int    endIdx") or pline.startswith("int endIdx"):
                    continue

                try:
                    c_type, pname, comment = _parse_decl_line(pline)
                except ValueError:
                    continue

                if pname in ("outBegIdx", "outNBElement"):
                    continue

                if pname.startswith("in"):
                    inputs.append(pname)
                elif pname.startswith("out"):
                    outputs.append(pname)
                elif pname.startswith("optIn"):
                    r_from, r_to = _parse_range(comment)
                    opt_inputs.append(
                        OptParam(name=pname, c_type=c_type, range_from=r_from, range_to=r_to)
                    )

            metas[name] = {
                "inputs": inputs,
                "outputs": outputs,
                "opt_inputs": [op.__dict__ for op in opt_inputs],
            }

        else:
            m = lookback_re.match(line)
            if m:
                name = m.group(1)
                proto: list[str] = [line]
                if ");" not in line:
                    i += 1
                    while i < len(lines):
                        proto.append(lines[i])
                        if ");" in lines[i]:
                            break
                        i += 1

                lookback_args: list[OptParam] = []
                for pline in proto:
                    pline = pline.strip()
                    if not pline or pline in ("(", ")", ");"):
                        continue

                    if "Lookback" in pline and "(" in pline:
                        # First line often contains the function name + first argument.
                        pline = pline.split("(", 1)[1].strip()
                        if not pline:
                            continue

                    try:
                        c_type, pname, comment = _parse_decl_line(pline)
                    except ValueError:
                        continue

                    if not pname.startswith("optIn"):
                        continue
                    r_from, r_to = _parse_range(comment)
                    lookback_args.append(
                        OptParam(name=pname, c_type=c_type, range_from=r_from, range_to=r_to)
                    )

                metas.setdefault(name, {})
                metas[name]["lookback_args"] = [arg.__dict__ for arg in lookback_args]

            else:
                i += 1
                continue

        i += 1

    if not metas:
        raise RuntimeError("Failed to parse any TA_ prototypes from ta_func.h")
    return metas


def _fmt_params(params: Iterable[OptParam]) -> str:
    out: list[str] = []
    for p in params:
        if p.range_from is not None and p.range_to is not None:
            out.append(f"{p.name}:{p.c_type}[{p.range_from},{p.range_to}]")
        else:
            out.append(f"{p.name}:{p.c_type}")
    return " | ".join(out)


def main() -> None:
    group_idx_path = UPSTREAM_ROOT / "src" / "ta_abstract" / "ta_group_idx.c"
    ta_func_h_path = UPSTREAM_ROOT / "include" / "ta_func.h"
    ta_func_src_dir = UPSTREAM_ROOT / "src" / "ta_func"

    group_map = parse_group_idx(group_idx_path)
    proto_meta = parse_ta_func_h(ta_func_h_path)

    # Preserve progress fields if a prior checklist exists.
    existing_progress: dict[str, dict[str, str]] = {}
    prior_csv = REPO_ROOT / "port_checklist.csv"
    if prior_csv.exists():
        with prior_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "")
                if not name:
                    continue
                existing_progress[name] = {
                    "status": row.get("status", "TODO"),
                    "parity": row.get("parity", "TODO"),
                    "speed_ratio_vs_talib": row.get("speed_ratio_vs_talib", ""),
                    "notes": row.get("notes", ""),
                }

    base_funcs = sorted(group_map.keys())
    if len(base_funcs) != 161:
        raise RuntimeError(f"Expected 161 functions from group idx; got {len(base_funcs)}")

    checklist_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []

    missing_in_header = [name for name in base_funcs if name not in proto_meta]
    if missing_in_header:
        raise RuntimeError(f"Functions missing prototypes in ta_func.h: {missing_in_header}")

    for name in base_funcs:
        group_id = group_map[name]
        group_name = GROUP_ID_TO_NAME.get(group_id, f"Group {group_id}")
        c_file = ta_func_src_dir / f"ta_{name}.c"
        c_file_str = str(c_file.relative_to(UPSTREAM_ROOT)) if c_file.exists() else None

        inputs = proto_meta[name].get("inputs", [])
        outputs = proto_meta[name].get("outputs", [])
        opt_inputs = [OptParam(**d) for d in proto_meta[name].get("opt_inputs", [])]
        lookback_args = [OptParam(**d) for d in proto_meta[name].get("lookback_args", [])]

        fm = FuncMeta(
            name=name,
            group_id=group_id,
            group_name=group_name,
            c_file=c_file_str,
            inputs=inputs,
            outputs=outputs,
            opt_inputs=opt_inputs,
            lookback_args=lookback_args,
        )

        checklist_rows.append(
            {
                "name": fm.name,
                "group_id": fm.group_id,
                "group_name": fm.group_name,
                "upstream_c_file": fm.c_file or "",
                "inputs": ",".join(fm.inputs),
                "outputs": ",".join(fm.outputs),
                "opt_inputs": _fmt_params(fm.opt_inputs),
                "lookback_args": _fmt_params(fm.lookback_args),
                "status": existing_progress.get(fm.name, {}).get("status", "TODO"),
                "parity": existing_progress.get(fm.name, {}).get("parity", "TODO"),
                "speed_ratio_vs_talib": existing_progress.get(fm.name, {}).get(
                    "speed_ratio_vs_talib", ""
                ),
                "notes": existing_progress.get(fm.name, {}).get("notes", ""),
            }
        )

        json_rows.append(
            {
                "name": fm.name,
                "group_id": fm.group_id,
                "group_name": fm.group_name,
                "upstream_c_file": fm.c_file,
                "inputs": fm.inputs,
                "outputs": fm.outputs,
                "opt_inputs": [op.__dict__ for op in fm.opt_inputs],
                "lookback_args": [op.__dict__ for op in fm.lookback_args],
            }
        )

    out_csv = REPO_ROOT / "port_checklist.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(checklist_rows[0].keys()))
        writer.writeheader()
        writer.writerows(checklist_rows)

    out_json_dir = REPO_ROOT / "generated"
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_json_dir / "ta_func_meta.json"
    json_text = json.dumps(json_rows, indent=2, ensure_ascii=False) + "\n"
    out_json.write_text(json_text, encoding="utf-8")

    # Keep a copy inside the Python package for runtime introspection / packaging.
    pkg_json = REPO_ROOT / "numbatalib" / "_generated" / "ta_func_meta.json"
    pkg_json.parent.mkdir(parents=True, exist_ok=True)
    pkg_json.write_text(json_text, encoding="utf-8")

    print(f"Wrote {out_csv} ({len(checklist_rows)} rows)")
    print(f"Wrote {out_json} ({len(json_rows)} functions)")
    print(f"Wrote {pkg_json} ({len(json_rows)} functions)")


if __name__ == "__main__":
    main()
