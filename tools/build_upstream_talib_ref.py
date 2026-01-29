from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "upstream" / "ta-lib"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:  # noqa: S310
        dest.write_bytes(r.read())


def _ensure_zig(zig_root: Path, version: str) -> Path:
    """
    Returns path to zig.exe.
    """
    zig_root.mkdir(parents=True, exist_ok=True)

    arch = "x86_64"
    platform = "windows"
    zip_name = f"zig-{platform}-{arch}-{version}.zip"
    url = f"https://ziglang.org/download/{version}/{zip_name}"
    zip_path = zig_root / zip_name

    extract_dir = zig_root / f"zig-{platform}-{arch}-{version}"
    zig_in_extract = extract_dir / "zig.exe"
    if zig_in_extract.exists():
        return zig_in_extract

    print(f"Downloading {url} -> {zip_path}")
    _download(url, zip_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"Extracting {zip_path} -> {zig_root}")
    shutil.unpack_archive(str(zip_path), str(zig_root))

    if zig_in_extract.exists():
        return zig_in_extract

    candidates = list(zig_root.glob("zig-*/zig.exe"))
    if not candidates:
        raise RuntimeError(f"zig.exe not found after extracting {zip_path}")
    return candidates[0]


def build_ref_dll(
    out_path: Path,
    zig_exe: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_dir = UPSTREAM / "include"
    common_dir = UPSTREAM / "src" / "ta_common"
    func_dir = UPSTREAM / "src" / "ta_func"

    sources = [
        common_dir / "ta_global.c",
        common_dir / "ta_retcode.c",
        common_dir / "ta_version.c",
        func_dir / "ta_SMA.c",
        func_dir / "ta_ACCBANDS.c",
        func_dir / "ta_AVGDEV.c",
        func_dir / "ta_IMI.c",
    ]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required upstream sources:\n" + "\n".join(missing))

    cmd = [
        str(zig_exe),
        "cc",
        "-shared",
        "-O3",
        "-std=c99",
        "-DTA_LIB_SHARED",
        f"-I{include_dir}",
        f"-I{common_dir}",
        f"-I{func_dir}",
        "-o",
        str(out_path),
        *[str(p) for p in sources],
    ]

    print("Building upstream TA-Lib reference DLL:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)  # noqa: S603


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny upstream TA-Lib DLL for C-reference checks.")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "generated" / "upstream_talib_ref.dll"),
        help="Output DLL path.",
    )
    parser.add_argument("--zig-version", default="0.12.0", help="Zig version to download/use.")
    parser.add_argument(
        "--zig-dir",
        default=str(REPO_ROOT / "generated" / "zig"),
        help="Directory to place zig.exe.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if DLL exists.")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Already exists: {out_path}")
        return

    zig_root = Path(args.zig_dir)
    zig_exe = _ensure_zig(zig_root, args.zig_version)
    print(f"Using {zig_exe}")

    build_ref_dll(out_path=out_path, zig_exe=zig_exe)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("This script currently targets Windows only.")
    main()
