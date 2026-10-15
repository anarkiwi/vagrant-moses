"""Instruction-stepped 6510 helpers for any py65-compatible MPU over a C64 memory.

:func:`flat_mpu` is the cheap alternative to the cycle-exact
:class:`~vagrant_moses.machine.Machine`: a banked :class:`~vagrant_moses.bus.Bus`
whose VIC raster reads follow the cycle counter, so raster spin loops exit.
"""

from __future__ import annotations

from typing import Dict, Optional

from . import memmap as mm
from .bus import Bus


def access_cycle(mpu) -> int:
    """The cycle the executing instruction's data access lands on: its last.

    Exact for accesses that cannot add a page-crossing cycle, such as an indexed
    store or a read within one page.
    """
    memory = getattr(mpu.memory, "_subject", mpu.memory)
    return mpu.processorCycles + mpu.cycletime[memory[mpu.pc]] - 1


def run_to_rts(mpu, mem, pc: int, acc: int, max_cycles: int) -> None:
    """Call the subroutine at ``pc`` with ``A = acc`` until its final RTS or the cap.

    A return address is pushed so the routine's last RTS lifts the stack pointer
    back above its start, ending the step loop.
    """
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        mpu.step()


def flat_mpu(
    ram: bytearray, *, ntsc: bool = False, roms: Optional[Dict[int, bytes]] = None
):
    """A jennings MPU over ``ram`` behind a :class:`Bus`; return ``(mpu, mem)``.

    ``mem`` is an observable memory (``subscribe_to_read``/``_write``) over the bus.
    ``$D011``/``$D012`` read the beam at the cycle counter on a PAL or NTSC frame.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from jennings.devices.mpu6502 import MPU
        from jennings.memory import ObservableMemory
    except ImportError as exc:
        raise ImportError(
            "flat_mpu runs on jennings: pip install vagrant-moses[cpu]"
        ) from exc
    bus = Bus(ram, roms)
    mem = ObservableMemory(subject=bus)
    timing = mm.video(ntsc)
    mpu = MPU(memory=mem)

    def raster(addr):
        if not bus.io:
            return None
        line = mpu.processorCycles // timing.cycles_per_line % timing.lines
        if addr == mm.VIC_RASTER:
            return line & 0xFF
        return ram[mm.VIC_CONTROL_1] & 0x7F | (line >> 8) << 7

    mem.subscribe_to_read((mm.VIC_CONTROL_1, mm.VIC_RASTER), raster)
    return mpu, mem


__all__ = ["access_cycle", "flat_mpu", "run_to_rts"]
