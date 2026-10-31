"""Time N native machines running in N threads against one: the GIL is released in run().

python tools/thread_machines.py --threads 8 --cycles 20000000
"""

from __future__ import annotations

import argparse
import threading
import time

from vagrant_moses.machine import Machine

LOOP = bytes((0xEE, 0x00, 0xC0, 0x4C, 0x00, 0x10))  # INC $C000 / JMP $1000


def _spin(cycles):
    machine = Machine(backend="native")
    machine.ram[0x1000 : 0x1000 + len(LOOP)] = LOOP
    machine.run(machine.reset(0x1000) + cycles)


def wall(threads, cycles):
    """Seconds for ``threads`` threads each running ``cycles`` cycles."""
    pool = [threading.Thread(target=_spin, args=(cycles,)) for _ in range(threads)]
    began = time.perf_counter()
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return time.perf_counter() - began


def main():
    """Print the single and N-thread wall times."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=20_000_000)
    args = parser.parse_args()
    single = wall(1, args.cycles)
    many = wall(args.threads, args.cycles)
    print(
        f"1 thread: {single:.2f}s  {args.threads} threads: {many:.2f}s  ({many / single:.2f}x)"
    )


if __name__ == "__main__":
    main()
