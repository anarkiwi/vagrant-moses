"""Instruction-stepped 6510 helpers for any py65-compatible MPU over a C64 memory.

:func:`flat_mpu` is the cheap alternative to the cycle-exact
:class:`~vagrant_moses.machine.Machine`: a banked :class:`~vagrant_moses.bus.Bus`
whose VIC raster reads follow the cycle counter, so raster spin loops exit. On a
jennings wheel it runs on the native core (``jennings.memory.NATIVE``); otherwise,
or with ``native=False``, on the pure-Python engine.
"""

from __future__ import annotations

from typing import Dict, Optional

from . import memmap as mm
from .bus import Bus, port_read

_PAGE = 0x1000
_BANKED_PAGES = (0xA, 0xB, 0xD, 0xE, 0xF)
_PORT = (mm.CPU_DDR, mm.CPU_PORT)


class NativeBus(Bus):
    """:class:`Bus` banking state over a jennings ``NativeMemory``; also its facade.

    The native memory owns the 64 KiB RAM: ``ram`` is a zero-copy view of it, so
    ``bus.ram[a]``, ``mem[a]`` and the executing CPU always agree. ROM and I/O
    banking are pushed into the native page table whenever ``$00``/``$01`` change.
    Subscriptions and the other ``NativeMemory`` methods are delegated; ``_subject``
    is the bus itself, as ``ObservableMemory._subject`` is on the Python path.
    """

    def __init__(self, native, roms: Optional[Dict[int, bytes]] = None):
        self._native = native
        self._images = {}
        super().__init__(memoryview(native), roms)
        native.subscribe_to_write(_PORT, self._port_write)
        native.subscribe_to_read((mm.CPU_PORT,), self._port_read)

    @property
    def _subject(self):
        return self

    def remap(self) -> None:
        was = self.bits
        super().remap()
        if self.bits == was:
            return
        native, ram = self._native, self.ram
        for n in _BANKED_PAGES:
            image = self.pages[n]
            rom = image is not None and image is not ram
            native.set_page(n, self._page(n, image) if rom else None)
        native.io = self.io

    def _page(self, n: int, image) -> bytes:
        key = (id(image), n)
        if key not in self._images:
            self._images[key] = bytes(image[n * _PAGE : (n + 1) * _PAGE])
        return self._images[key]

    def __getitem__(self, addr):
        return self._native[addr]

    def __setitem__(self, addr, value) -> None:
        self._native[addr] = value

    def __len__(self) -> int:
        return len(self._native)

    def __getattr__(self, name):
        if name == "_native":
            raise AttributeError(name)
        return getattr(self._native, name)

    def _port_read(self, _addr) -> int:
        return port_read(self.port_dir, self.port_data)

    def _port_write(self, addr, value) -> None:
        if addr:
            self.port_data = value
        else:
            self.port_dir = value
        self.remap()


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
    back above its start, ending the step loop. A native MPU runs the same loop
    in its own driver.
    """
    if getattr(mpu, "native", False):
        mpu.run_to_rts(pc, acc, max_cycles)
        return
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        mpu.step()


def flat_mpu(
    ram: bytearray,
    *,
    ntsc: bool = False,
    roms: Optional[Dict[int, bytes]] = None,
    native: Optional[bool] = None,
):
    """A jennings MPU over ``ram`` behind a banked bus; return ``(mpu, mem)``.

    ``mem`` is an observable memory (``subscribe_to_read``/``_write``) over the bus;
    ``mem._subject`` is the :class:`Bus` (``bits``, ``io``, ``ram``).
    ``$D011``/``$D012`` read the beam at the cycle counter on a PAL or NTSC frame.

    ``native`` selects the engine: ``None`` follows ``jennings.memory.NATIVE``,
    ``False`` forces the pure-Python one and ``True`` requires the native core.
    On the Python engine ``ram`` is the live RAM. The native core copies ``ram``
    in once and owns the RAM afterwards: reach it through ``mem[...]`` or
    ``mem.ram`` (a zero-copy view), not through the ``ram`` argument.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from jennings.devices.mpu6502 import MPU
        from jennings.memory import NATIVE, NativeMemory, ObservableMemory
    except ImportError as exc:
        raise ImportError(
            "flat_mpu runs on jennings: pip install vagrant-moses[cpu]"
        ) from exc
    if native and not NATIVE:
        raise ImportError("jennings was installed without its native core")
    if native is None:
        native = NATIVE
    timing = mm.video(ntsc)
    if native:
        core = NativeMemory(ram)
        mem = NativeBus(core, roms)
        core.raster_model(ntsc)
        return MPU(memory=core), mem
    bus = Bus(ram, roms)
    mem = ObservableMemory(subject=bus)
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


__all__ = ["NativeBus", "access_cycle", "flat_mpu", "run_to_rts"]
