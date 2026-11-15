"""Profile how much of a ``run_forks`` workload recurs as identical segments.

``bench_forks``' ``play`` base, or ``keys``: its main loop scans the keyboard into the copy
pointer so forks diverge by key column. Overhead: best profiled vs unprofiled wall time.
``python tools/segment_profile.py --forks 4000 --frames 2 --threads 8 --workload keys``
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
from bench_forks import base_machine, programs
from bench_machine import SOURCE
from vagrant_moses import run_forks

KEYS = SOURCE.replace(
    "  SEI\n", "  SEI\n  LDA #$FF\n  STA $DC02\n  LDA #$00\n  STA $DC00\n", 1
).replace("main: LDY #$00", "main: LDA $DC01\n  STA $FB\n  LDY #$00")
WORKLOADS = {"play": SOURCE, "keys": KEYS}


def batches(base, progs, args, bar, repeat: int = 1) -> float:
    """Seconds to run ``progs`` in ``--batch`` forks per call, best of ``repeat`` each."""
    wall = 0.0
    for at in range(0, len(progs), args.batch):
        chunk, best = progs[at : at + args.batch], float("inf")
        for _ in range(repeat):
            began = time.perf_counter()
            run_forks(base, chunk, threads=args.threads)
            best = min(best, time.perf_counter() - began)
            bar.update(len(chunk))
        wall += best
    return wall


def arm(base, args):
    """Arm the profiler on ``base`` afresh as ``args`` asks, or disarm it (``None``)."""
    if args is None:
        base.profile_segments(enabled=False)
    else:
        base.profile_segments(
            boundaries=[int(pc, 16) for pc in args.boundaries] or None,
            max_insns=args.max_insns,
            calls=args.calls,
        )


def report(p: dict, plain: float, profiled: float):
    """Print the summary ``p`` and the profiler's overhead."""
    cycles, seg = p["cycles"], p["segment_cycles"]

    def pct(v, of):
        return f"{v:,} ({100 * v / max(of, 1):.1f}%)"

    rows = [
        ("cycles run", f"{cycles:,}"),
        ("in segments", pct(seg, cycles)),
        ("in segments ended early (io/event/irq)", pct(p["early_cycles"], seg)),
        ("segments / distinct keys", f"{p['segments']:,} / {p['distinct']:,}"),
        ("recurrent (key seen twice or more)", pct(p["recurrent_cycles"], cycles)),
        ("saved by a per-machine cache", pct(p["saved"]["machine"], cycles)),
        ("saved by a per-batch cache", pct(p["saved"]["batch"], cycles)),
        ("saved by a global cache", pct(p["saved"]["global"], cycles)),
        ("effect conflicts", f"{p['conflicts']}"),
        ("batches / forks", f"{p['batches']} / {p['forks']:,}"),
        ("unprofiled / profiled s", f"{plain:.3f} / {profiled:.3f}"),
        ("overhead", f"{100 * (profiled / plain - 1):.1f}%"),
    ]
    print("| measure | value |\n|---|---:|")
    print("\n".join(f"| {k} | {v} |" for k, v in rows))
    print("\n| end | segments | cycles |\n|---|---:|---:|")
    for name, (n, c) in p["ends"].items():
        print(f"| {name} | {n:,} | {c:,} |")
    print("\n| instructions | segments | cycles |\n|---|---:|---:|")
    for lo, hi, n, c in p["lengths"]:
        print(f"| {lo}-{hi} | {n:,} | {c:,} |")
    print("\n| entry pc | keys | occurrences | cycles | saved | first (batch, fork) |")
    print("|---|---:|---:|---:|---:|---|")
    for t in p["top"]:
        print(
            f"| ${t['pc']:04X} | {t['keys']:,} | {t['occurrences']:,} | {t['cycles']:,}"
            f" | {t['saved']:,} | {t['first']} |"
        )


def main():
    """Time the workload unprofiled and profiled, then print a fresh profile of it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="keys")
    parser.add_argument("--forks", type=int, default=4000)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-insns", type=int, default=256)
    parser.add_argument("--calls", action="store_true")
    parser.add_argument("--boundaries", nargs="*", default=[], help="hex pcs")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    base = base_machine("native", WORKLOADS[args.workload])
    progs = programs(args.forks, args.frames)
    walls = []
    with tqdm(total=(2 * args.repeat + 1) * args.forks, unit="fork") as bar:
        for mode in (None, args):
            bar.set_description("profiled" if mode else "unprofiled")
            arm(base, mode)
            walls.append(batches(base, progs, args, bar, args.repeat))
        bar.set_description("profile")
        arm(base, args)
        batches(base, progs, args, bar)
    report(base.segment_profile(args.top), *walls)


if __name__ == "__main__":
    main()
