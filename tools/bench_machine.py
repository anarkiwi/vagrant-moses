"""Benchmark the Python and native machines, with and without a SID.

``play``: a CIA IRQ plays a SID routine with indexed writes while the main loop runs an
``(zp),Y`` copy, illegal opcodes and a ``$D012`` spin, with all sprites enabled.
``heavy``: an ``STA $D400,X`` loop over every register. The SID is absent (``none``), a
Python device (``device``) or a NativeSid whose chip is never asked (``logged``).
Only ``Machine.run`` is timed; the NativeSid log is drained between slices.

    python tools/bench_machine.py --seconds 20 --backend native
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
from vagrant_moses.machine import NATIVE, Machine, NativeSid, boot_cycle

from asm import assemble

ORG, IRQ, TABLE, SLICES = 0x1000, 0x1200, 0x1300, 10
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
HEAVY = """
  LDA #$55
main: LDX #$18
loop: STA $D400,X
  DEX
  BPL loop
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


class Counted:
    """A NativeSid whose log is counted and dropped."""

    def __init__(self):
        self.count = 0
        self.dev = NativeSid("6581", lambda _cycle, _register: 0)

    def drain(self):
        self.count += len(self.dev.drain_writes())


def bench(backend, workload, kind, seconds):
    """``(Mcycles/s, x realtime, SID writes)`` for ``seconds`` PAL seconds."""
    sid = {"none": None, "device": Sid(), "logged": Counted()}[kind]
    dev = sid.dev if kind == "logged" else sid
    machine = Machine(sids=None if dev is None else {mm.SID_BASE: dev}, backend=backend)
    programs = (
        ((ORG, SOURCE), (IRQ, HANDLER)) if workload == "play" else ((ORG, HEAVY),)
    )
    for org, src in programs:
        code, _ = assemble(src, org)
        machine.ram[org : org + len(code)] = code
    machine.ram[TABLE : TABLE + 256] = bytes(range(256))
    start = machine.reset(ORG, boot_cycle(False))
    clock = mm.PAL.clock_hz
    slices, wall = int(seconds * SLICES), 0.0
    for n in tqdm(
        range(1, slices + 1), desc=f"{backend}/{workload}/{kind}", unit="s/10"
    ):
        began = time.perf_counter()
        machine.run(start + n * clock // SLICES)
        wall += time.perf_counter() - began
        if kind == "logged":
            sid.drain()
    cycles = machine.t - start
    writes = getattr(sid, "count", 0)
    return cycles / wall / 1e6, cycles / wall / clock, writes


def main():
    """Print Mcycles/s and the speed relative to a real C64 per backend, workload, SID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--backend", choices=("python", "native"), action="append")
    parser.add_argument("--workload", choices=("play", "heavy"), action="append")
    parser.add_argument("--sid", choices=("none", "device", "logged"), action="append")
    args = parser.parse_args()
    print("| backend | workload | SID | Mcycles/s | x realtime | SID writes |")
    print("|---|---|---|---:|---:|---:|")
    for backend in args.backend or ["python"] + ["native"] * NATIVE:
        for workload in args.workload or ("play", "heavy"):
            for kind in args.sid or ("none", "device", "logged"):
                mcps, ratio, writes = bench(backend, workload, kind, args.seconds)
                print(
                    f"| {backend} | {workload} | {kind} | {mcps:.2f} | {ratio:.1f} | {writes} |"
                )


if __name__ == "__main__":
    main()
