"""Run the generated REU programs on VICE's x64sc (headless, in Docker) and the model.

python tools/vice_reu.py --programs 48 --jobs 16 [--write-fixture] [--backend python]
VICE dumps are cached under ``.reu-cache/vice``; ``--write-fixture`` records their
digests in ``tests/data/reu_vice.json`` for the tests to compare against without VICE.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import reugen  # pylint: disable=wrong-import-position
from vagrant_moses import memmap as mm  # pylint: disable=wrong-import-position
from vagrant_moses.roms import rom_less  # pylint: disable=wrong-import-position

IMAGE = "anarkiwi/headlessvice:latest"
CACHE = ROOT / ".reu-cache"
FIXTURE = ROOT / "tests" / "data" / "reu_vice.json"
LIMIT_CYCLES = 30_000_000


_DUMP = 'bank ram\nbsave "/w/ram.bin" 0 0000 ffff\nbank io\nbsave "/w/col.bin" 0 d800 dbff\nquit\n'


def _monitor(program: reugen.Program) -> str:
    return (
        f'l "/w/t.prg" 0\nr pc = {reugen.CODE:04x}\nbreak exec {program.done:04x}\n'
        'command 1 "playback \\"/w/dump.txt\\""\nx\n'
    )


def vice(program: reugen.Program, emulator: str = "x64sc", timeout: int = 120):
    """VICE's :class:`~reugen.Observation` of ``program`` (cached by its key)."""
    cached = CACHE / emulator / f"{program.key}.npz"
    image = np.frombuffer(program.image, np.uint8)
    if cached.exists():
        with np.load(cached) as dump:
            return reugen.Observation(dump["ram"], dump["colour"], dump["reu"] ^ image)
    work = CACHE / "work" / program.key
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "t.prg").write_bytes(program.prg)
    (work / "reu.img").write_bytes(program.image)
    (work / "mon.txt").write_text(_monitor(program))
    (work / "dump.txt").write_text(_DUMP)
    roms = rom_less()
    for name, base in (("kernal", mm.KERNAL_BASE), ("basic", mm.BASIC_BASE)):
        (work / f"{name}.bin").write_bytes(roms[base])
    (work / "chargen.bin").write_bytes(roms[mm.CHARGEN_BASE])
    name = f"vice-reu-{program.key}"
    args = ["docker", "run", "--rm", "--name", name, "-v", f"{work}:/w"]
    args += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/w", IMAGE]
    args += [
        emulator,
        "-default",
        "-silent",
        "-sounddev",
        "dummy",
        "-initbreak",
        "reset",
    ]
    args += ["-kernal", "/w/kernal.bin", "-basic", "/w/basic.bin"]
    args += ["-chargen", "/w/chargen.bin", "-reu", "-reusize", str(program.size_kb)]
    args += ["-reuimage", "/w/reu.img", "-reuimagerw"]
    args += ["-limitcycles", str(LIMIT_CYCLES), "-moncommands", "/w/mon.txt"]
    try:
        subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        raise RuntimeError(f"{program.name}: VICE timed out") from exc
    if not (work / "ram.bin").exists():
        raise RuntimeError(f"{program.name}: VICE never reached done")
    obs = reugen.Observation.of(
        (work / "ram.bin").read_bytes(),
        (work / "col.bin").read_bytes(),
        (work / "reu.img").read_bytes(),
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, ram=obs.ram, colour=obs.colour, reu=obs.reu ^ image)
    shutil.rmtree(work)
    return obs


def describe(program, model, ref) -> str:
    """The first differences between two observations of ``program``."""
    out = []
    rows, log = reugen.results(model, program)
    ref_rows, ref_log = reugen.results(ref, program)
    for i, (got, want) in enumerate(zip(rows, ref_rows)):
        if (got != want).any():
            op = program.ops[i]
            out.append(f"op {i} {op}\n  model {got.tolist()}\n  vice  {want.tolist()}")
            break
    if log.shape != ref_log.shape or (log != ref_log).any():
        out.append(f"irq log model {log.tolist()} vice {ref_log.tolist()}")
    for part in ("ram", "colour", "reu"):
        diff = np.flatnonzero(getattr(model, part) != getattr(ref, part))
        if diff.size:
            out.append(f"{part}: {diff.size} bytes differ from ${diff[0]:X}")
    return "\n".join(out)


def main():
    """Run VICE and the model over the corpus; exit 1 on any difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1764)
    parser.add_argument("--ops", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--emulator", default="x64sc")
    parser.add_argument("--display", choices=("off", "on", "both"), default="both")
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    modes = {"off": (False,), "on": (True,), "both": (False, True)}[args.display]
    programs = [
        p
        for display in modes
        for p in reugen.corpus(args.programs, args.seed, args.ops, display)
    ]
    with ThreadPoolExecutor(args.jobs) as pool:
        refs = list(
            tqdm(
                pool.map(lambda p: vice(p, args.emulator), programs),
                total=len(programs),
                desc=args.emulator,
            )
        )
    bad = 0
    for program, ref in tqdm(list(zip(programs, refs)), desc="model"):
        model = reugen.run(program, args.backend)
        if model.digest() != ref.digest():
            bad += 1
            tqdm.write(
                f"{program.name} ({program.size_kb} KiB):\n"
                + describe(program, model, ref)
            )
    print(f"{len(programs) - bad}/{len(programs)} programs agree with {args.emulator}")
    if args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        corpus = {"count": args.programs, "seed": args.seed, "ops": args.ops}
        corpus["display"] = list(modes)
        entries = {p.name: {"key": p.key, **r.digest()} for p, r in zip(programs, refs)}
        doc = {"emulator": args.emulator, "corpus": corpus, "programs": entries}
        FIXTURE.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
