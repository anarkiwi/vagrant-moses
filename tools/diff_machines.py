"""Run random programs on the Python and native machines and report any divergence.

    python tools/diff_machines.py --programs 500 --cycles 2000000 --jobs 24
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import machine_diff  # pylint: disable=wrong-import-position


def _one(args):
    seed, cycles = args
    return seed, machine_diff.compare(seed, cycles)


def main():
    """Compare ``--programs`` seeds from ``--first``; exit 1 on any divergence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=int, default=500)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=2_000_000)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    seeds = [(s, args.cycles) for s in range(args.first, args.first + args.programs)]
    bad = []
    with Pool(args.jobs) as pool:
        for seed, why in tqdm(pool.imap_unordered(_one, seeds), total=len(seeds)):
            if why:
                bad.append(seed)
                tqdm.write(f"seed {seed}: {why}")
    print(f"{len(seeds) - len(bad)}/{len(seeds)} programs agree")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
