"""Benchmark ``run_forks``: forks/s and emulated frames/s by thread count.

The base is ``bench_machine``'s ``play`` workload (a CIA IRQ per PAL frame, its handler the
marker); each fork holds a key for ``--frames`` frames and reads a byte (0: the fork alone).
``python tools/bench_forks.py --forks 4000 --frames 2 --threads 1 2 4 8 16 24``
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "tests")]

# pylint: disable=wrong-import-position
from bench_machine import HANDLER, IRQ, ORG, SOURCE, TABLE
from vagrant_moses import Machine, Program, R, run_forks
from vagrant_moses import memmap as mm
from vagrant_moses.machine import NATIVE, boot_cycle

from asm import assemble


def base_machine(backend: str, source: str = SOURCE):
    """The ``play`` workload (or ``source`` as its main program), a tenth of a second in,
    with its events cleared."""
    machine = Machine(backend=backend)
    for org, src in ((ORG, source), (IRQ, HANDLER)):
        code, _ = assemble(src, org)
        machine.ram[org : org + len(code)] = code
    machine.ram[TABLE : TABLE + 256] = bytes(range(256))
    start = machine.reset(ORG, boot_cycle(False))
    machine.run(start + mm.PAL.clock_hz // 10)
    machine.clear_events()
    return machine


def programs(n: int, frames: int):
    """``n`` programs, each holding its own key for ``frames`` frames."""
    frame = mm.PAL.clock_hz // 50
    return [
        Program((frames + 1) * frame, marker=IRQ)
        .key(i % 8, i // 8 % 8)
        .count_to(IRQ, frames)
        .ld(R[0], 0xC000)
        .out(R[0])
        for i in range(n)
    ]


def bench(base, progs, threads: int, batch: int, bar) -> tuple:
    """``(seconds, cycles emulated)`` for every program, ``batch`` forks per call."""
    wall = cycles = 0
    for at in range(0, len(progs), batch):
        chunk = progs[at : at + batch]
        began = time.perf_counter()
        outs = run_forks(base, chunk, threads=threads)
        wall += time.perf_counter() - began
        cycles += sum(o.cycle - base.t for o in outs)
        bar.update(len(chunk))
    return wall, cycles


def main():
    """Print forks/s, frames/s, Mcycles/s and the speedup over the first thread count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forks", type=int, default=4000)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24])
    parser.add_argument(
        "--backend", choices=("python", "native"), default="native" if NATIVE else None
    )
    args = parser.parse_args()
    base = base_machine(args.backend)
    progs = programs(args.forks, args.frames)
    rows = []
    with tqdm(total=args.forks * len(args.threads), unit="fork") as bar:
        for threads in args.threads:
            bar.set_description(f"{threads} threads")
            rows.append((threads, *bench(base, progs, threads, args.batch, bar)))
    print("| threads | forks/s | frames/s | Mcycles/s | speedup |")
    print("|---:|---:|---:|---:|---:|")
    for threads, wall, cycles in rows:
        print(
            f"| {threads} | {args.forks / wall:.0f} | {args.forks * args.frames / wall:.0f} "
            f"| {cycles / wall / 1e6:.1f} | {rows[0][1] / wall:.2f} |"
        )


if __name__ == "__main__":
    main()
