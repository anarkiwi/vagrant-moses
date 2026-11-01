"""The Commodore 17xx RAM Expansion Unit (REC 8726) as VICE 3.10 models it.

:class:`Reu` holds the ``$DF00-$DF1F`` registers (mirrored to ``$DFFF``), the RAM and
the DMA engine, which the machine steps once per halted cycle with the VIC's BA: reads
wait for BA high and a write may take the first BA-low cycle after a write, as x64sc.
"""

from __future__ import annotations

from typing import Callable, Optional

SIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
TO_REU, FROM_REU, SWAP, VERIFY = range(4)

IRQ_PENDING, END_OF_BLOCK, VERIFY_ERROR, CHIPS_256K = 0x80, 0x40, 0x20, 0x10
EXECUTE, AUTOLOAD, FF00_DISABLED = 0x80, 0x20, 0x10
INT_ENABLE, INT_EOB, INT_VERIFY = 0x80, 0x40, 0x20
FIX_C64, FIX_REU = 0x80, 0x40
REGS = 0x0B

_HIGH, _AFTER_WRITE = range(2)


def geometry(size_kb: int):
    """``(wrap, dram_wrap, store_mask, bank_unused, status)`` of an REU of ``size_kb``.

    The REC's 19-bit counter wraps at ``wrap``, the DRAM at ``dram_wrap``; ``store_mask``
    is what an address keeps when stored back and ``bank_unused`` the bank bits stuck at 1.
    """
    if size_kb not in SIZES:
        raise ValueError(f"REU size must be one of {SIZES} KiB, got {size_kb!r}")
    if size_kb == 128:
        return 0x20000, 0x20000, 0x7FFFF, 0xF8, 0
    if size_kb <= 512:
        return 0x80000, 0x80000, 0x7FFFF, 0xF8, CHIPS_256K
    size = size_kb << 10
    return 0x80000, size, size - 1, 0, CHIPS_256K


class Reu:
    """A 17xx REU: registers, ``size_kb`` KiB of RAM and the DMA engine.

    ``bus`` has ``dma_read(addr)``/``dma_write(addr, value)`` (the C64 side as ``$01``
    maps it); ``irq(state)`` drives the IRQ line and ``request()`` halts the CPU.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        size_kb: int = 512,
        bus=None,
        irq: Optional[Callable[[bool], None]] = None,
        request: Optional[Callable[[], None]] = None,
    ):
        self.size_kb = size_kb
        (
            self.wrap,
            self.dram_wrap,
            self.store_mask,
            self.bank_unused,
            self.preset,
        ) = geometry(size_kb)
        self.ram = bytearray(size_kb << 10)
        self.bus = bus
        self.irq = irq or (lambda state: None)
        self.request = request or (lambda: None)
        self.latch = 0xFF
        self.armed = self.busy = self.line = False
        self.kind = self.left = self.host = self.raddr = 0
        self.hstep = self.rstep = self.phase = self.gate = self.delay = 0
        self.last = self.over = self.tail = False
        self.phantom = self.mask = self.value = self.rv = 0
        self.reset()

    def reset(self) -> None:
        """The registers as a reset leaves them (RAM and the bus latch survive)."""
        self.status = self.preset
        self.command = FF00_DISABLED
        self.c64 = self.c64_shadow = 0
        self.base = self.base_shadow = 0
        self.bank = self.bank_shadow = self.bank_unused
        self.length = self.length_shadow = 0xFFFF
        self.int_mask = 0x1F
        self.control = 0x3F
        self._set_line(False)

    def _set_line(self, state: bool) -> None:
        if state != self.line:
            self.line = state
            self.irq(state)

    # --- registers --------------------------------------------------------------

    def peek(self, reg: int) -> int:
        """Register ``reg`` (``0-$1F``) without a read's side effects."""
        reg &= 0x1F
        if reg >= REGS:
            return 0xFF
        return (
            self.status,
            self.command,
            self.c64 & 0xFF,
            self.c64 >> 8,
            self.base & 0xFF,
            self.base >> 8 & 0xFF,
            self.bank | self.bank_unused,
            self.length & 0xFF,
            self.length >> 8,
            self.int_mask,
            self.control,
        )[reg]

    @property
    def regs(self) -> bytes:
        """Registers ``$00-$0A`` as :meth:`peek` reads them."""
        return bytes(self.peek(reg) for reg in range(REGS))

    def read(self, addr: int) -> Optional[int]:
        """A CPU read of ``$DF00-$DFFF``; ``None`` while the DMA drives the bus."""
        if self.busy:
            return None
        reg = addr & 0x1F
        value = self.peek(reg)
        if reg == 0:
            self.status &= ~(VERIFY_ERROR | END_OF_BLOCK | IRQ_PENDING)
            self._set_line(False)
        elif reg == 6:
            value |= 0xF8
        return value

    def write(self, addr: int, value: int) -> None:
        """A CPU write of ``$DF00-$DFFF``."""
        reg = addr & 0x1F
        if self.busy or reg >= REGS:
            return
        if reg == 1:
            self.command = value
            if value & EXECUTE:
                self.armed = not value & FF00_DISABLED
                if not self.armed:
                    self.request()
        elif reg in (2, 3):
            self.c64 = self.c64_shadow = _byte(self.c64_shadow, reg & 1, value)
        elif reg in (4, 5):
            self.base = self.base_shadow = _byte(self.base_shadow, reg & 1, value)
        elif reg == 6:
            self.bank = self.bank_shadow = value & ~self.bank_unused
        elif reg in (7, 8):
            self.length = self.length_shadow = _byte(
                self.length_shadow, reg == 8, value
            )
        elif reg == 9:
            self.int_mask = value | 0x1F
            self._check_irq(self.status)
        else:
            self.control = value | 0x3F

    def _check_irq(self, bits: int) -> None:
        for flag, enable in ((END_OF_BLOCK, INT_EOB), (VERIFY_ERROR, INT_VERIFY)):
            if bits & flag and _all(self.int_mask, enable):
                self.status |= IRQ_PENDING
                self._set_line(True)

    def ff00(self) -> None:
        """A write to ``$FF00``: start the DMA armed by a command with the trigger on."""
        if self.armed:
            self.armed = False
            self.request()

    # --- DMA ----------------------------------------------------------------------

    def _load(self, addr: int) -> int:
        addr &= self.dram_wrap - 1
        return self.ram[addr] if addr < len(self.ram) else self.latch

    def _store(self, addr: int, value: int) -> None:
        addr &= self.dram_wrap - 1
        if addr < len(self.ram):
            self.ram[addr] = value

    def _advance(self) -> None:
        self.host = (self.host + self.hstep) & 0xFFFF
        nxt = (self.raddr & 0x7FFFF) + self.rstep
        self.raddr = (self.raddr & 0xF80000) | (0 if nxt == self.wrap else nxt)
        self.left -= 1

    def _begin(self) -> None:
        self.busy = True
        self.kind = self.command & 3
        self.host, self.raddr = self.c64, self.base | self.bank << 16
        self.left = self.length or 0x10000
        self.hstep = 0 if self.control & FIX_C64 else 1
        self.rstep = 0 if self.control & FIX_REU else 1
        self.phase = self.delay = self.phantom = self.mask = 0
        self.gate, self.last, self.over, self.tail = _HIGH, False, False, False

    def step(self, ba: bool) -> bool:
        """One DMA cycle with the VIC's BA ``ba``; true when the CPU runs this cycle."""
        if not self.busy:
            self._begin()
        if self.gate == _AFTER_WRITE:
            if ba:
                self.delay += 1
                if self.delay > 1:
                    self.delay, self.last, self.gate = 0, True, _HIGH
                    return False
            else:
                self.delay = 0
            self.last = False
        elif ba:
            return False
        self.gate = _HIGH
        return self._slot()

    def _slot(self) -> bool:
        if self.phantom:
            self.phantom -= 1
            return False
        if not self.left or self.over:
            if self.last and not self.tail and self.kind in (FROM_REU, SWAP):
                self.tail = True
                return False
            self._finish()
            return True
        bus, kind = self.bus, self.kind
        if kind == TO_REU:
            self.value = bus.dma_read(self.host)
            self._store(self.raddr, self.value)
            self._advance()
        elif kind == FROM_REU:
            self.latch = value = self._load(self.raddr)
            bus.dma_write(self.host, value)
            self._advance()
            self.gate = _AFTER_WRITE
        elif kind == SWAP and not self.phase:
            self.rv = self._load(self.raddr)
            self.value = bus.dma_read(self.host)
            self.phase = 1
        elif kind == SWAP:
            self._store(self.raddr, self.value)
            bus.dma_write(self.host, self.rv)
            self._advance()
            self.phase, self.gate = 0, _AFTER_WRITE
        else:
            differ = self._load(self.raddr) != bus.dma_read(self.host)
            self._advance()
            if differ:
                self.mask, self.over = VERIFY_ERROR, True
                self.phantom = 1 if self.left else 0
        return False

    def _finish(self) -> None:
        left = self.left
        if self.kind != VERIFY or not left:
            self.mask, left = self.mask | END_OF_BLOCK, 1
        elif left == 1 and self._load(self.raddr) == self.bus.dma_read(self.host):
            self.mask |= END_OF_BLOCK
        raddr = self.raddr & self.store_mask
        self.status |= self.mask
        if self.command & AUTOLOAD:
            self.c64, self.base = self.c64_shadow, self.base_shadow
            self.bank, self.length = self.bank_shadow, self.length_shadow
        else:
            if not self.control & FIX_C64:
                self.c64 = self.host
            if not self.control & FIX_REU:
                self.base, self.bank = raddr & 0xFFFF, raddr >> 16 & 0xFF
            self.length = left & 0xFFFF
        self._check_irq(self.mask)
        if self.kind == TO_REU:
            self.latch = self.value
        elif self.kind == FROM_REU:
            self.latch = self._load(self.raddr)
        self.busy = False
        self.command = (self.command & ~EXECUTE) | FF00_DISABLED


def _byte(word: int, high: int, value: int) -> int:
    return (word & 0x00FF) | value << 8 if high else (word & 0xFF00) | value


def _all(reg: int, bit: int) -> bool:
    return reg & (INT_ENABLE | bit) == INT_ENABLE | bit


__all__ = [
    "Reu",
    "SIZES",
    "TO_REU",
    "FROM_REU",
    "SWAP",
    "VERIFY",
    "geometry",
]
