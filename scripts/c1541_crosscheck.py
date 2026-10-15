"""Compare native disk extraction with VICE ``c1541 -extract`` over real media.

usage: python scripts/c1541_crosscheck.py SRC...
Walks every disk image reachable from SRC (through zip/gzip nesting) and prints each
file found by only one reader. VICE appends a status byte to a file cut short by a
broken link, so that byte is dropped before comparing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from vagrant_moses.cli import _sources
from vagrant_moses.media import GEOMETRY_BY_SIZE, iter_prgs, read_disk


def c1541_payloads(img: bytes) -> Counter:
    """File payloads VICE extracts from ``img``."""
    with tempfile.TemporaryDirectory() as tmp:
        path, out = Path(tmp, "img"), Path(tmp, "out")
        path.write_bytes(img)
        out.mkdir()
        args = ["c1541", "-attach", str(path), "-extract"]
        env = dict(os.environ, HOME=tmp)
        subprocess.run(args, cwd=out, env=env, capture_output=True, check=False)
        return Counter(p.read_bytes() for p in out.iterdir() if p.is_file())


def main(srcs) -> int:
    """Print per-image differences; returns the number of differing images."""
    disks = [
        (path, data)
        for src, name in tqdm(list(_sources(srcs)), desc="scan", unit="file")
        for path, data in iter_prgs(name, src.read_bytes(), lambda *a: None, False)
        if len(data) in GEOMETRY_BY_SIZE
    ]
    bad = 0
    for path, img in tqdm(disks, desc="c1541", unit="img"):
        files = read_disk(img)
        vice = c1541_payloads(img)
        for f in files:
            if f.error and vice[f.data + b"\xc7"]:
                vice[f.data + b"\xc7"] -= 1
                vice[f.data] += 1
        only_n = [f for f in files if vice[f.data] == 0]
        only_v = vice - Counter(f.data for f in files)
        if only_n or +only_v:
            bad += 1
            tqdm.write(f"{path}:")
            for f in only_n:
                tqdm.write(
                    f"  native-only {f.name!r} {f.type_name} {len(f.data)} {f.error}"
                )
            for blob in +only_v:
                tqdm.write(f"  vice-only {len(blob)} bytes {blob[:8].hex()}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
