"""``vagrant-moses`` command line: ``media extract SRC... -o DIR``."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tqdm import tqdm

from .media import iter_prgs


def _sources(srcs: List[str]) -> Iterator[Tuple[Path, str]]:
    """Each input file with its name relative to the directory argument it came from."""
    for src in map(Path, srcs):
        if src.is_dir():
            for f in sorted(p for p in src.rglob("*") if p.is_file()):
                yield f, f.relative_to(src).as_posix()
        else:
            yield src, src.name


def safe_path(path: str) -> str:
    """``path`` with each component reduced to a portable, non-traversing name."""
    parts = (re.sub(r"[^A-Za-z0-9._+-]", "_", p) for p in path.split("/"))
    return "/".join("_" if not p.strip(".") else p for p in parts)


def index_line(data: bytes, rel: str) -> str:
    """``sha256 size $load path`` as in a ``prg_index.txt``."""
    load = f"${data[0] | data[1] << 8:04X}" if len(data) > 1 else "-"
    return f"{hashlib.sha256(data).hexdigest()} {len(data):6d} {load} {rel}"


def extract(srcs: List[str], out: Path) -> int:
    """Write every leaf file under ``out`` and print its index line; returns errors."""
    errors = []

    def on_error(path, exc):
        errors.append(path)
        tqdm.write(f"ERROR {path}: {exc}", file=sys.stderr)

    written = set()
    files = list(_sources(srcs))
    for src, name in tqdm(files, unit="file", desc="extract"):
        for path, data in iter_prgs(name, src.read_bytes(), on_error):
            rel = base = safe_path(path)
            n = 1
            while rel in written:
                n += 1
                rel = f"{base}~{n}"
            written.add(rel)
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(index_line(data, rel), flush=True)
    return len(errors)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="vagrant-moses")
    sub = parser.add_subparsers(dest="group", required=True)
    media = sub.add_parser("media").add_subparsers(dest="cmd", required=True)
    ext = media.add_parser("extract", help="extract PRGs from nested release media")
    ext.add_argument("src", nargs="+", help="files or directories")
    ext.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 1 if extract(args.src, args.out) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
