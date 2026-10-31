"""The host-side face of a SID: its write log and the data-bus value of unreadable registers.

Pure-Python twin of ``vagrant_moses._core.NativeSid``; the chip itself lives behind
``on_read``. Bus value rule and constants: libresidfp ``SID.cpp`` (``BUS_TTL_6581``,
``BUS_TTL_8580``, ``SID::read``, ``SID::write``) and ``SID.h`` (``ageBusValue``). A write,
and a read of a readable register (``$19-$1C``), set the value and its time to live;
reading any other register halves the time left; the value becomes 0 once it has run out.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

BUS_TTL = {"6581": 0x01D00, "8580": 0xA2000}
READABLE = range(0x19, 0x1D)

Write = Tuple[int, int, int]


class PyNativeSid:
    """Logs writes, models the bus value and asks ``on_read(cycle, reg)`` for ``$19-$1C``."""

    def __init__(self, model: str, on_read: Callable[[int, int], int]):
        if model not in BUS_TTL:
            raise ValueError("model must be '6581' or '8580'")
        self.model, self.on_read = model, on_read
        self._ttl_model = BUS_TTL[model]
        self._writes: List[Write] = []
        self._rows: List[Write] = []
        self._last = bytearray(32)
        self._bus = self._ttl = self._clock = 0

    def _age(self, cycle: int) -> None:
        elapsed, self._clock = cycle - self._clock, cycle
        if self._ttl:
            self._ttl -= elapsed
            if self._ttl <= 0:
                self._bus = self._ttl = 0

    def _drive(self, value: int) -> None:
        self._bus, self._ttl = value, self._ttl_model

    def write(self, cycle: int, reg: int, value: int) -> None:
        """Log a write; ``drain_rows`` keeps those that change the register."""
        if not 0 <= value <= 0xFF:
            raise ValueError("byte must be in range(0, 256)")
        reg &= 0x1F
        self._age(cycle)
        self._drive(value)
        self._writes.append((cycle, reg, value))
        if self._last[reg] != value:
            self._last[reg] = value
            self._rows.append((cycle, reg, value))

    def read(self, cycle: int, reg: int) -> int:
        """``on_read`` for ``$19-$1C``, the decaying bus value for the rest."""
        reg &= 0x1F
        if reg not in READABLE:
            self._age(cycle)
            self._ttl //= 2
            return self._bus
        value = self.on_read(cycle, reg)
        if not 0 <= value <= 0xFF:
            raise ValueError("byte must be in range(0, 256)")
        self._age(cycle)
        self._drive(value)
        return value

    def drain_writes(self) -> List[Write]:
        """Every write since the last drain as ``(cycle, reg, value)``."""
        out, self._writes = self._writes, []
        return out

    def drain_rows(self) -> List[Write]:
        """The writes since the last drain that changed their register."""
        out, self._rows = self._rows, []
        return out

    @property
    def last(self) -> bytes:
        """The last value written to each register."""
        return bytes(self._last)
