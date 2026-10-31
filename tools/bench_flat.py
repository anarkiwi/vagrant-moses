"""Bench ``flat_mpu`` + ``run_to_rts``: Python engine vs native core, in Mcycles/s.

The routine is a playroutine shape: spin on ``$D012`` until a raster line, update
SID registers, return. It is called back to back until the cycle budget is spent.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from asm import assemble  # pylint: disable=wrong-import-position
from vagrant_moses import memmap as mm  # pylint: disable=wrong-import-position
from vagrant_moses.cpu import (  # pylint: disable=wrong-import-position
    flat_mpu,
    run_to_rts,
)

CODE = 0x1000
PLAY = """
play:  LDA #$80
wait:  CMP $D012
       BNE wait
       INC $D400
       LDX #$10
mix:   LDA $2000,X
       ADC $D401
       STA $D402
       DEX
       BNE mix
       RTS
"""


def bench(native: bool, budget: int) -> float:
    """Mcycles/s running :data:`PLAY` repeatedly for ``budget`` cycles."""
    code, _ = assemble(PLAY, CODE)
    ram = bytearray(0x10000)
    ram[mm.CPU_DDR], ram[mm.CPU_PORT] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    ram[CODE : CODE + len(code)] = code
    mpu, mem = flat_mpu(ram, native=native)
    name = "native" if native else "python"
    start = time.perf_counter()
    with tqdm(total=budget, desc=name, unit="cyc", unit_scale=True) as bar:
        while mpu.processorCycles < budget:
            before = mpu.processorCycles
            run_to_rts(mpu, mem, CODE, 0, mm.PAL.cycles_per_frame * 2)
            bar.update(mpu.processorCycles - before)
    return mpu.processorCycles / (time.perf_counter() - start) / 1e6


def main() -> None:
    """Run both backends and print Mcycles/s and the speed-up."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=20_000_000)
    args = ap.parse_args()
    py, nat = bench(False, args.cycles), bench(True, args.cycles)
    print(f"python {py:.2f} Mcycles/s  native {nat:.2f} Mcycles/s  x{nat / py:.0f}")


if __name__ == "__main__":
    main()
