"""The 6510's view of C64 memory: RAM, the ``$01``-banked ROMs and the I/O page.

:class:`Bus` is a py65-style memory (``bus[addr]``, ``bus[addr] = value``) any
6502 core can run on. Writes always reach RAM, also under a ROM; the CPU port at
``$00``/``$01`` selects the PLA's banking as the C64 does.
"""

from __future__ import annotations

from typing import Dict, Optional

from . import memmap as mm
from .c64 import poweron_ram
from .roms import rom_less


def port_read(ddr: int, data: int) -> int:
    """The 6510 port ``$01`` as read: output bits as written, inputs as the C64 pulls.

    Input bits 0-4 read high, an input bit 5 and bits 6-7 read low
    (libsidplayfp's ``ZeroRAMBank`` without the data-bit fall-off).
    """
    pins = (0x3F & ~ddr) | (data & ddr)
    value = (data | ~ddr) & (pins | 0x17) & 0xFF
    if not ddr & 0x20:
        value &= ~0x20
    return value & ~(~ddr & 0xC0)


def banking(bits: int):
    """``(basic, kernal, chargen, io)`` mapped for the port's effective low bits."""
    loram, hiram, charen = bits & mm.LORAM, bits & mm.HIRAM, bits & mm.CHAREN
    return (
        bool(loram and hiram),
        bool(hiram),
        bool(not charen and (loram or hiram)),
        bool(charen and (loram or hiram)),
    )


class Bus:
    """C64 memory as the 6510 sees it, banked by the CPU port.

    ``roms`` maps base to bytes and overrides sidplayfp's ROM-less set per ROM.
    The I/O page reads and writes RAM unless :meth:`io_read`/:meth:`io_write`
    are overridden; the port registers start from ``ram[0]``/``ram[1]``.
    """

    def __init__(
        self,
        ram: Optional[bytearray] = None,
        roms: Optional[Dict[int, bytes]] = None,
    ):
        self.ram = bytearray(poweron_ram()) if ram is None else ram
        self.roms = {}
        for base, data in {**rom_less(), **(roms or {})}.items():
            image = bytearray(0x10000)
            image[base : base + len(data)] = data
            self.roms[base] = image
        self.pages = [self.ram] * 16
        self.port_dir, self.port_data = self.ram[mm.CPU_DDR], self.ram[mm.CPU_PORT]
        self.bits, self.io = -1, False
        self.remap()

    def remap(self) -> None:
        """Re-derive the page map from the CPU port registers."""
        bits = (self.port_data | ~self.port_dir) & 7
        if bits == self.bits:
            return
        self.bits = bits
        basic, kernal, chars, self.io = banking(bits)
        pages, ram, roms = self.pages, self.ram, self.roms
        pages[0xA] = pages[0xB] = roms[mm.BASIC_BASE] if basic else ram
        pages[0xD] = None if self.io else roms[mm.CHARGEN_BASE] if chars else ram
        pages[0xE] = pages[0xF] = roms[mm.KERNAL_BASE] if kernal else ram

    def __getitem__(self, addr: int) -> int:
        if addr > 1:
            page = self.pages[addr >> 12]
            return page[addr] if page is not None else self.io_read(addr)
        return port_read(self.port_dir, self.port_data) if addr else self.port_dir

    def __setitem__(self, addr: int, value: int) -> None:
        if self.io and addr >> 12 == 0xD:
            self.io_write(addr, value)
            return
        self.ram[addr] = value
        if addr < 2:
            if addr:
                self.port_data = value
            else:
                self.port_dir = value
            self.remap()

    def io_read(self, addr: int) -> int:
        """A read of the banked-in I/O page ``$D000-$DFFF``."""
        return self.ram[addr]

    def io_write(self, addr: int, value: int) -> None:
        """A write to the banked-in I/O page ``$D000-$DFFF``."""
        self.ram[addr] = value


__all__ = ["Bus", "banking", "port_read"]
