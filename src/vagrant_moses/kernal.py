"""A C64 program host with the KERNAL's file and keyboard calls serviced in Python.

:class:`KernalHost` runs a 6510 over a :class:`~vagrant_moses.bus.Bus` and traps
the real KERNAL jump-table entries, mapping logical files onto a host directory
as ``host_dir/<device>/<name>``. :func:`sys_addr` finds a BASIC stub's SYS target.
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from . import memmap as mm
from .bus import Bus

SYS_TOKEN = 0x9E
ST = 0x90

JUMP_TABLE: Dict[str, int] = {
    "READST": 0xFFB7,
    "SETLFS": 0xFFBA,
    "SETNAM": 0xFFBD,
    "OPEN": 0xFFC0,
    "CLOSE": 0xFFC3,
    "CHKIN": 0xFFC6,
    "CHKOUT": 0xFFC9,
    "CLRCHN": 0xFFCC,
    "CHRIN": 0xFFCF,
    "CHROUT": 0xFFD2,
    "LOAD": 0xFFD5,
    "SAVE": 0xFFD8,
    "GETIN": 0xFFE4,
    "CLALL": 0xFFE7,
}

FILE_OPEN, FILE_NOT_OPEN, FILE_NOT_FOUND = 2, 3, 4
NOT_INPUT_FILE, NOT_OUTPUT_FILE, MISSING_FILE_NAME = 6, 7, 8
ST_EOF, ST_VERIFY, ST_TIMEOUT_READ = 0x40, 0x10, 0x02

_CARRY, _ZERO, _NEGATIVE = 0x01, 0x02, 0x80
_DRIVE_PREFIX = re.compile(r"^@?\d?:")


def sys_addr(mem, base: int = 0x0801) -> int:
    """The first numeric SYS operand in the tokenised BASIC program at ``base``.

    Only each line's token bytes are scanned; pointer cycles and lines running
    off the end of ``mem`` stop the walk.

    Raises:
      ValueError: no line holds a numeric SYS.
    """
    size = len(mem) if hasattr(mem, "__len__") else 0x10000
    line, seen = base, set()
    while line not in seen and line + 4 <= size:
        seen.add(line)
        nxt = mem[line] | mem[line + 1] << 8
        if not nxt:
            break
        pos = line + 4
        while pos < size and mem[pos]:
            if mem[pos] == SYS_TOKEN:
                pos += 1
                while pos < size and mem[pos] == 0x20:
                    pos += 1
                digits = ""
                while pos < size and 0x30 <= mem[pos] <= 0x39:
                    digits += chr(mem[pos])
                    pos += 1
                if digits:
                    return int(digits)
            else:
                pos += 1
        line = nxt
    raise ValueError(f"no numeric SYS in the BASIC program at ${base:04X}")


class KernalHostError(RuntimeError):
    """A hosted program ran past its step budget."""


class TouchBus(Bus):
    """A :class:`Bus` recording the ``addr >> 12`` page of every write in ``touched``."""

    def __init__(self, ram, roms, touched: set):
        self.touched = touched
        super().__init__(ram, roms)

    def __setitem__(self, addr: int, value: int) -> None:
        self.touched.add(addr >> 12)
        super().__setitem__(addr, value)


class KernalHost:
    """A whole C64 program run with the KERNAL I/O calls serviced against ``host_dir``.

    A call is trapped whenever the KERNAL is banked in and PC reaches its
    jump-table entry, then returns as RTS would with the real register ABI.
    """

    def __init__(self, host_dir, *, roms=None, mpu=None):
        self.host_dir = Path(host_dir)
        ram = bytearray(0x10000)
        ram[mm.CPU_DDR], ram[mm.CPU_PORT] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
        self.touched: set = set()
        self.bus = TouchBus(ram, roms, self.touched)
        self.ram = ram
        if mpu is None:
            try:
                from jennings.devices.mpu6502 import MPU
            except ImportError as exc:
                raise ImportError(
                    "KernalHost needs a 6510 core: pip install vagrant-moses[cpu]"
                ) from exc
            mpu = MPU(memory=self.bus)
        self.mpu = mpu
        self.output = bytearray()
        self.keys: deque = deque()
        self.files: Dict[int, Tuple[object, bool]] = {}
        self.lfs, self.name = (0, 0, 0), ""
        self.chan_in: Optional[int] = None
        self.chan_out: Optional[int] = None
        self.calls: Dict[int, Callable[[], None]] = {
            addr: getattr(self, "_" + name.lower()) for name, addr in JUMP_TABLE.items()
        }

    def tap_key(self, petscii: int) -> "KernalHost":
        """Queue one PETSCII key for GETIN and keyboard CHRIN."""
        self.keys.append(petscii & 0xFF)
        return self

    def screen(self) -> bytes:
        """The default text screen RAM ``$0400-$07FF``."""
        return bytes(self.ram[mm.SCREEN_RAM : mm.SCREEN_RAM + 0x400])

    def load_prg(self, prg: bytes) -> int:
        """Place a PRG at its header address and reset the CPU to start there."""
        load = prg[0] | prg[1] << 8
        for i, byte in enumerate(prg[2:]):
            self.bus[(load + i) & 0xFFFF] = byte
        mpu = self.mpu
        mpu.a = mpu.x = mpu.y = mpu.p = 0
        mpu.sp, mpu.pc = 0xFF, load
        return load

    def step(self) -> None:
        """Execute one instruction, or service a trapped KERNAL call and return."""
        mpu = self.mpu
        call = self.calls.get(mpu.pc) if self.bus.bits & mm.HIRAM else None
        if call is None:
            mpu.step()
            return
        if mpu.pc != JUMP_TABLE["READST"]:
            self.ram[ST] = 0
        call()
        ram, sp = self.ram, mpu.sp
        lo, hi = ram[0x100 | (sp + 1) & 0xFF], ram[0x100 | (sp + 2) & 0xFF]
        mpu.pc, mpu.sp = (hi << 8 | lo) + 1 & 0xFFFF, (sp + 2) & 0xFF
        mpu.processorCycles += 6

    def run(self, until=None, max_steps: int = 2_000_000) -> "KernalHost":
        """Step until ``until(host)`` holds, or by default until PC stops moving.

        Raises:
          KernalHostError: ``max_steps`` ran out first.
        """
        for _ in range(max_steps):
            pc = self.mpu.pc
            if until is not None and until(self):
                return self
            self.step()
            if until is None and self.mpu.pc == pc:
                return self
        raise KernalHostError(
            f"no stop within {max_steps} steps (PC ${self.mpu.pc:04X})"
        )

    def run_to_unpacked(
        self, prg: bytes, entry: int = 0x0801, max_steps: int = 2_000_000
    ) -> Tuple[int, bytes]:
        """Run a packed PRG from ``entry`` until PC enters a page it wrote itself.

        Returns ``(pc, code)``: the code runs from PC to the end of the contiguous
        run of written pages; pages the PRG itself occupies never count.

        Raises:
          KernalHostError: ``max_steps`` ran out first.
        """
        load = self.load_prg(prg)
        own = range(load >> 12, ((load + max(len(prg) - 2, 1) - 1) >> 12) + 1)
        touched = self.touched
        touched.clear()
        self.mpu.pc = entry
        for _ in range(max_steps):
            pc = self.mpu.pc
            if pc >> 12 in touched and pc >> 12 not in own:
                end = pc >> 12
                while end + 1 in touched:
                    end += 1
                return pc, bytes(self.ram[pc : (end + 1) << 12])
            self.step()
        raise KernalHostError(f"nothing unpacked within {max_steps} steps")

    def _flags(self, carry: int, a: Optional[int]) -> None:
        mpu = self.mpu
        p = mpu.p & ~_CARRY | carry
        if a is not None:
            mpu.a = a
            p = p & ~(_ZERO | _NEGATIVE) | (0 if a else _ZERO) | a & _NEGATIVE
        mpu.p = p

    def _ok(self, a: Optional[int] = None) -> None:
        self._flags(0, a)

    def _err(self, code: int) -> None:
        self._flags(_CARRY, code)

    def _path(self) -> Path:
        base = _DRIVE_PREFIX.sub("", self.name.split(",", 1)[0])
        return self.host_dir / str(self.lfs[1]) / base

    def _readst(self) -> None:
        self._ok(self.ram[ST])

    def _setlfs(self) -> None:
        mpu = self.mpu
        self.lfs = (mpu.a, mpu.x, mpu.y)
        self._ok()

    def _setnam(self) -> None:
        mpu, bus = self.mpu, self.bus
        addr = mpu.x | mpu.y << 8
        self.name = bytes(bus[(addr + i) & 0xFFFF] for i in range(mpu.a)).decode(
            "latin-1"
        )
        self._ok()

    def _open(self) -> None:
        lfn, _, sec = self.lfs
        if lfn in self.files:
            self._err(FILE_OPEN)
            return
        if not self.name:
            self._err(MISSING_FILE_NAME)
            return
        mods = self.name.split(",")[1:]
        write = sec == 1 or any(m.strip().upper().startswith("W") for m in mods)
        path = self._path()
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "wb")
        else:
            try:
                handle = open(path, "rb")
            except OSError:
                self._err(FILE_NOT_FOUND)
                return
        self.files[lfn] = (handle, write)
        self._ok()

    def _select(self, write: bool) -> None:
        lfn = self.mpu.x
        if lfn not in self.files:
            self._err(FILE_NOT_OPEN)
        elif self.files[lfn][1] != write:
            self._err(NOT_OUTPUT_FILE if write else NOT_INPUT_FILE)
        else:
            if write:
                self.chan_out = lfn
            else:
                self.chan_in = lfn
            self._ok()

    def _chkin(self) -> None:
        self._select(False)

    def _chkout(self) -> None:
        self._select(True)

    def _clrchn(self) -> None:
        self.chan_in = self.chan_out = None
        self._ok()

    def _close(self) -> None:
        lfn = self.mpu.a
        entry = self.files.pop(lfn, None)
        if entry is not None:
            entry[0].close()
        if self.chan_in == lfn:
            self.chan_in = None
        if self.chan_out == lfn:
            self.chan_out = None
        self._ok()

    def _clall(self) -> None:
        for handle, _ in self.files.values():
            handle.close()
        self.files.clear()
        self._clrchn()

    def _chrin(self) -> None:
        if self.chan_in is None:
            self._ok(self.keys.popleft() if self.keys else 0)
            return
        handle = self.files[self.chan_in][0]
        byte = handle.read(1)
        if not byte:
            self.ram[ST] |= ST_EOF | ST_TIMEOUT_READ
            self._ok(0x0D)
            return
        if not handle.peek(1):
            self.ram[ST] |= ST_EOF
        self._ok(byte[0])

    def _chrout(self) -> None:
        a = self.mpu.a
        if self.chan_out is None:
            self.output.append(a)
        else:
            self.files[self.chan_out][0].write(bytes((a,)))
        self._ok()

    def _getin(self) -> None:
        self._ok(self.keys.popleft() if self.keys else 0)

    def _load(self) -> None:
        mpu, bus = self.mpu, self.bus
        if not self.name:
            self._err(MISSING_FILE_NAME)
            return
        try:
            data = self._path().read_bytes()
        except OSError:
            data = b""
        if len(data) < 2:
            self._err(FILE_NOT_FOUND)
            return
        start = mpu.x | mpu.y << 8 if self.lfs[2] == 0 else data[0] | data[1] << 8
        verify = mpu.a != 0
        for i, byte in enumerate(data[2:]):
            addr = (start + i) & 0xFFFF
            if not verify:
                bus[addr] = byte
            elif bus[addr] != byte:
                self.ram[ST] |= ST_VERIFY
        end = (start + len(data) - 2) & 0xFFFF
        mpu.x, mpu.y = end & 0xFF, end >> 8
        self._ok()

    def _save(self) -> None:
        mpu, bus = self.mpu, self.bus
        if not self.name:
            self._err(MISSING_FILE_NAME)
            return
        start = bus[mpu.a] | bus[(mpu.a + 1) & 0xFF] << 8
        end = mpu.x | mpu.y << 8
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        body = bytes(bus[addr] for addr in range(start, end))
        path.write_bytes(start.to_bytes(2, "little") + body)
        self._ok()


__all__ = [
    "JUMP_TABLE",
    "ST",
    "SYS_TOKEN",
    "KernalHost",
    "KernalHostError",
    "TouchBus",
    "sys_addr",
]
