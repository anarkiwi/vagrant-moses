"""Benchmark the Python and native machines on a playroutine-shaped workload.

A CIA IRQ plays a SID routine with indexed writes while the main loop runs an
``(zp),Y`` copy, illegal opcodes and a ``$D012`` spin, with all sprites enabled.

    python tools/bench_machine.py --seconds 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

# pylint: disable=wrong-import-position
from vagrant_moses import memmap as mm
from vagrant_moses.machine import NATIVE, Machine, boot_cycle

from asm import assemble

ORG, IRQ, TABLE = 0x1000, 0x1200, 0x1300
SOURCE = f"""
  SEI
  LDA #$7F
  STA $DC0D
  LDA #${IRQ & 0xFF:02X}
  STA $0314
  LDA #${IRQ >> 8:02X}
  STA $0315
  LDA #${mm.PAL.kernal_latch & 0xFF:02X}
  STA $DC04
  LDA #${mm.PAL.kernal_latch >> 8:02X}
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  LDA #$FF
  STA $D015
  LDA #$40
  STA $D001
  STA $D003
  STA $D005
  STA $D007
  STA $D009
  STA $D00B
  STA $D00D
  STA $D00F
  LDA #$00
  STA $FB
  LDA #${TABLE >> 8:02X}
  STA $FC
  CLI
main: LDY #$00
copy: LDA ($FB),Y
  STA $C000,Y
  SLO $C000,Y
  LAX $C001
  DCP $C002
  ISC $C003
  INY
  BNE copy
spin: LDA $D012
  CMP #$80
  BNE spin
  JMP main
"""
HANDLER = f"""
  LDY #$18
play: LDA ${TABLE:04X},Y
  STA $D400,Y
  DEY
  BPL play
  LDA $DC0D
  PLA
  TAY
  PLA
  TAX
  PLA
  RTI
"""


class Sid:
    """A SID stand-in that counts its writes."""

    def __init__(self):
        self.count = 0

    def read(self, _cycle, _register):
        return 0

    def write(self, _cycle, _register, _value):
        self.count += 1


def bench(backend, seconds):
    """``(Mcycles/s, x realtime, SID writes)`` for ``seconds`` PAL seconds."""
    sid = Sid()
    machine = Machine(sids={mm.SID_BASE: sid}, backend=backend)
    for org, src in ((ORG, SOURCE), (IRQ, HANDLER)):
        code, _ = assemble(src, org)
        machine.ram[org : org + len(code)] = code
    machine.ram[TABLE : TABLE + 256] = bytes(range(256))
    start = machine.reset(ORG, boot_cycle(False))
    clock = mm.PAL.clock_hz
    began = time.perf_counter()
    for second in tqdm(range(1, int(seconds) + 1), desc=backend, unit="s"):
        machine.run(start + second * clock)
    wall = time.perf_counter() - began
    cycles = machine.t - start
    return cycles / wall / 1e6, cycles / wall / clock, sid.count


def main():
    """Print Mcycles/s and the speed relative to a real C64 for each backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--backend", choices=("python", "native"), action="append")
    args = parser.parse_args()
    for backend in args.backend or ["python"] + ["native"] * NATIVE:
        mcps, ratio, writes = bench(backend, args.seconds)
        print(
            f"{backend:>7}: {mcps:8.2f} Mcycles/s  {ratio:8.1f}x realtime  {writes} SID writes"
        )


if __name__ == "__main__":
    main()
