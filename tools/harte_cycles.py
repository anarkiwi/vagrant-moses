"""Compare the machine's bus cycles with Tom Harte's SingleStepTests 6502 cases.

python tools/harte_cycles.py --cases 100 --jobs 16 [--ops 40,60,bd]
Fetches each opcode's cases once into ``.harte-cache``; exits 1 on any discrepancy.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import harte  # pylint: disable=wrong-import-position
from jennings.opcodes import OPCODES  # pylint: disable=wrong-import-position


def _one(args):
    op, count = args
    found = Counter()
    for case in harte.cases(op, count):
        if not harte.unstable(op, case):
            found.update(harte.check(case))
    return op, found


def main():
    """Check ``--cases`` cases of every opcode (or ``--ops``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--ops", type=lambda s: [int(o, 16) for o in s.split(",")])
    args = parser.parse_args()
    ops = args.ops or harte.OPS
    bad = 0
    with Pool(args.jobs) as pool:
        work = pool.imap_unordered(_one, [(op, args.cases) for op in ops])
        for op, found in tqdm(work, total=len(ops), unit="op"):
            if found:
                bad += 1
                mn, mode = OPCODES[op]
                tqdm.write(f"{op:02x} {mn} {mode}: {found.most_common(3)}")
    print(f"{len(ops) - bad}/{len(ops)} opcodes match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
