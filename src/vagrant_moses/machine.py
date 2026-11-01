"""A cycle-exact C64: a 6510 on the :class:`~vagrant_moses.bus.Bus` with the VIC-II and two CIAs.

:class:`Machine` mirrors libsidplayfp's ``MOS6510`` cycle tables, RDY stalls and
interrupt delay over jennings (the ``cpu`` extra). SID chips are pluggable devices;
every interrupt raise, entry and RTI is reported to ``on_event``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from . import memmap as mm
from .bus import Bus
from .c64 import poweron_ram
from .roms import rom_less
from .sid import PyNativeSid
from .cia import Cia
from .reu import Reu
from .eventsched import PHI1, Event, Scheduler
from .vicii import Vic

try:
    from . import _core
except ImportError:  # no wheel for this platform: the Python machine only
    _core = None

NATIVE = _core is not None
NativeSid = _core.NativeSid if NATIVE else PyNativeSid
_MAX = 65536
# Entry kinds of libsidplayfp's per-opcode cycle tables; those below R write.
W, PUSHPCH, LOWPC, PUSHSR = range(4)
R, X2, F, CLI, SEI, POPSR, BR, BRK0, VECLO, VECHI, FN, RTI, HALT, ANE = range(4, 18)
FETCHES = frozenset((F, CLI, SEI, RTI, FN))
BRK_ENTRIES = (BRK0, PUSHPCH, LOWPC, PUSHSR, VECLO, VECHI, FN)
_ANE_MAGIC = 0xEF
_CLI_BASE, _SEI_BASE = 0x58 << 3, 0x78 << 3
_IRQ_I = 0x04
SOURCES = ("nmi", "video_irq", "cia_irq", "reu_irq")
_NMI, _VIDEO, _CIA, _REU = range(4)
_IRQS = (_VIDEO, _CIA, _REU)

_READ_OPS = frozenset(
    "ADC AND CMP EOR LDA LDX LDY ORA SBC BIT CPX CPY LAX LAS ANC ALR ARR ANE LXA "
    "SBX NOP".split()
)
_RMW_OPS = frozenset("ASL LSR ROL ROR INC DEC SLO RLA SRE RRA DCP ISC".split())
_STORE_OPS = frozenset("STA STX STY SAX SHA SHX SHY TAS".split())
_BRANCHES = {
    "BPL": (0x80, 0),
    "BMI": (0x80, 1),
    "BVC": (0x40, 0),
    "BVS": (0x40, 1),
    "BCC": (0x01, 0),
    "BCS": (0x01, 1),
    "BNE": (0x02, 0),
    "BEQ": (0x02, 1),
}
_OP_TAIL = {
    "CLI": (CLI,),
    "SEI": (SEI,),
    "PHA": (W,),
    "PHP": (W,),
    "PLA": (R, R),
    "PLP": (R, POPSR),
    "JSR": (R, W, W, R, F),
    "JMP": (F,),
    "RTS": (R, R, R, R),
    "RTI": (R, POPSR, R, R, RTI),
}
_PREFIX = {
    "impl": (R,),
    "acc": (R,),
    "imm": (R,),
    "rel": (R,),
    "zp": (R,),
    "zpx": (R, R),
    "zpy": (R, R),
    "abs": (R, R),
    "ind": (R, R, R, R),
    "indx": (R, R, R, R),
}


def _jennings():
    try:
        from jennings.devices.mpu6502 import (
            MPU,
        )  # pylint: disable=import-outside-toplevel
        from jennings.opcodes import (
            MEM_MODES,
            OPCODES,
        )  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "the cycle-exact machine runs on jennings: pip install vagrant-moses[cpu]"
        ) from exc
    return MPU, MEM_MODES, OPCODES


def opcode_layout(op: int) -> Tuple[Tuple[int, ...], int, str]:
    """libsidplayfp's cycle entries after ``op``'s fetch, its exec entry and kind.

    The exec entry is where the CPU core runs the instruction (its data access);
    the kind is ``"rmw"``, ``"absx"``/``"absy"``/``"indy"`` (page-cross skip),
    ``"branch"`` or ``""``.
    """
    _, mem_modes, opcodes = _jennings()
    mn, mode = opcodes[op]
    if mn == "JAM":
        return (HALT,), -1, ""
    if mn == "BRK":
        return BRK_ENTRIES, -1, ""
    if mn == "ANE":
        return (ANE, F), -1, ""
    reads = mn in _READ_OPS and mode in mem_modes
    rmw = mn in _RMW_OPS and mode != "acc"
    if mode in ("absx", "absy", "indy"):
        skip = X2 if reads else R
        prefix = (R, skip, R) if mode != "indy" else (R, R, skip, R)
    elif mn == "JSR":
        prefix = (R,)
    else:
        prefix = _PREFIX[mode]
    ents = list(prefix) + ([R] if reads or rmw else [])
    if mn in _BRANCHES:
        ents += [BR, R]
    elif rmw:
        ents += [W, W]
    elif mn in _STORE_OPS:
        ents += [W]
    elif mn in _OP_TAIL:
        ents += _OP_TAIL[mn]
    elif mn != "NOP":
        ents += [F]
    if ents[-1] not in FETCHES:
        ents.append(F)
    exec_at = next(i for i, k in enumerate(ents) if k not in (R, X2)) - 1
    kind = "branch" if mn in _BRANCHES else "rmw" if rmw else mode if X2 in ents else ""
    return tuple(ents), exec_at, kind


def _layouts():
    opcodes = _jennings()[2]
    out = []
    for op in range(256):
        ents, exec_at, kind = opcode_layout(op)
        out.append((ents, exec_at + 1, kind, op << 3, _BRANCHES.get(opcodes[op][0])))
    return out


_KINDS = {"": 0, "rmw": 1, "absx": 2, "absy": 3, "indy": 4, "branch": 5}


def _native_layouts():
    """``Machine.LAYOUTS`` as the native machine takes it, then ``BRK_ENTRIES`` last."""
    if Machine.LAYOUTS is None:
        Machine.LAYOUTS = _layouts()
    return [
        (
            list(ents),
            exec_at,
            _KINDS[kind],
            base,
            branch and (branch[0], bool(branch[1])),
        )
        for ents, exec_at, kind, base, branch in Machine.LAYOUTS
    ]


def _rom_images(roms: Optional[Dict[int, bytes]] = None):
    """BASIC, CHARGEN and KERNAL as 64 KiB images, as :class:`~vagrant_moses.bus.Bus` keeps them;
    ``roms`` (base -> bytes) overrides the ROM-less set per ROM."""
    roms = {**rom_less(), **(roms or {})}
    images = []
    for base in (mm.BASIC_BASE, mm.CHARGEN_BASE, mm.KERNAL_BASE):
        image = bytearray(0x10000)
        image[base : base + len(roms[base])] = roms[base]
        images.append(bytes(image))
    return tuple(images)


def boot_cycle(ntsc: bool) -> int:
    """The cycle sidplayfp's CPU first runs: after its fixed power-on event count.

    sidtrace pins the power-on delay to its maximum, so ``Player::initialise``
    fires ``3 * (0x1FFF + 8000)`` VIC, CIA-timer and TOD events before the reset.
    """
    cpl = mm.video(ntsc).cycles_per_line
    pattern = (0, 1, 11, 14, 15, 54, 55, 56, 57) + ((58,) if ntsc else ())
    tod = 17045 if ntsc else 19704
    target = 3 * (0x1FFF + 8000)
    lines = target // len(pattern) + 2
    times = [0, 1, 1, 1, 1]
    times += [n * cpl + lc for n in range(lines) for lc in pattern if n * cpl + lc]
    times += [k * tod for k in range(lines * cpl // tod + 1) for _ in (0, 1)]
    times.sort()
    return times[target - 1]


class Machine(Bus):
    """The C64 under sidplayfp: a cycle-exact 6510 with the VIC-II and two CIAs.

    ``sids`` maps a 32-byte-aligned I/O base to a device with ``read(cycle, reg)``
    and ``write(cycle, reg, value)``, or to a :class:`NativeSid`, which the native machine
    logs without calling Python except for ``$D419-$D41C``; the ``$D400`` device answers the
    rest of ``$D400-$D7FF``. ``on_event(cycle, name)`` gets each raise, entry and RTI.
    ``reu`` is the size in KiB of a 17xx REU at ``$DF00`` (``None``: none). ``roms`` maps a
    ROM base to its bytes and overrides sidplayfp's ROM-less set per ROM.
    """

    # pylint: disable=too-many-instance-attributes

    LAYOUTS = None

    def __new__(
        cls,
        ntsc: bool = False,
        sids: Optional[Dict[int, object]] = None,
        cia_new: bool = False,
        on_event: Optional[Callable[[int, str], None]] = None,
        backend: Optional[str] = None,
        reu: Optional[int] = None,
        roms: Optional[Dict[int, bytes]] = None,
        **_kwargs,
    ):
        """The native machine for ``backend="native"`` (the default when ``_core``
        imports), else this pure-Python reference; subclasses are always Python."""
        if backend not in (None, "native", "python"):
            raise ValueError(f"backend must be 'native' or 'python', got {backend!r}")
        if backend == "native" and not NATIVE:
            raise ImportError("vagrant_moses._core is not built")
        if cls is not Machine or backend == "python" or not NATIVE:
            return super().__new__(cls)
        _jennings()
        lines, cycles_per_line = mm.video(ntsc).geometry
        return _core.Machine(
            ntsc,
            dict(sids or {}),
            cia_new,
            on_event,
            _native_layouts(),
            bytes(poweron_ram()),
            _rom_images(roms),
            lines,
            cycles_per_line,
            reu,
        )

    def __init__(
        self,
        ntsc: bool = False,
        sids: Optional[Dict[int, object]] = None,
        cia_new: bool = False,
        on_event: Optional[Callable[[int, str], None]] = None,
        backend: Optional[str] = None,
        reu: Optional[int] = None,
        roms: Optional[Dict[int, bytes]] = None,
    ):
        del backend
        mpu_cls = _jennings()[0]
        if Machine.LAYOUTS is None:
            Machine.LAYOUTS = _layouts()
        super().__init__(roms=roms)
        self.ntsc = ntsc
        self.sched = Scheduler()
        self.vic = Vic(self.sched, ntsc, self._vic_irq, self._set_ba)
        self.cia1 = Cia(self.sched, self._cia_irq, cia_new)
        self.cia2 = Cia(self.sched, self._cia_nmi, cia_new)
        self.clear_int = Event(self._remove_irq)
        self.sids = dict(sids or {})
        self.events: List[Tuple[int, str]] = []
        self.on_event = on_event or (
            lambda cycle, name: self.events.append((cycle, name))
        )
        self.colour = bytearray(0x400)
        self.seed = 3686734
        self.asserted = [False] * len(SOURCES)
        self.dma = False
        self.reu = (
            None
            if reu is None
            else Reu(reu, self, lambda on: self._irq(on, _REU), self._request)
        )
        self.irq_cause = _CIA
        self.irq_count = 0
        self.ic, self.pin, self.nmi, self.iflag, self.rdy = (
            _MAX,
            False,
            False,
            False,
            True,
        )
        self.d1x1 = False
        self.pending: List[Tuple[int, int]] = []
        self.last_read = 0
        self.vec_addr = self.target = 0
        self.cross = self.taken = False
        self.ents, self.idx, self.base, self.exec_at, self.rmw = (
            BRK_ENTRIES,
            6,
            0,
            0,
            False,
        )
        self.t, self.cc, self.jammed = 0, 6, False
        self.mpu = mpu_cls(memory=self)
        self.cia1.reset()
        self.cia2.reset()
        self.vic.reset()

    @property
    def cycle(self) -> int:
        """The machine's current cycle."""
        return self.t

    # --- I/O page --------------------------------------------------------------

    def __setitem__(self, addr: int, value: int) -> None:
        if self.io and addr >> 12 == 0xD:
            self.pending.append((addr, value))
            return
        if addr == 0xFF00 and self.reu is not None and self.reu.armed:
            if self.rmw:
                value = self[addr]
            self.reu.ff00()
        super().__setitem__(addr, value)

    def dma_read(self, addr: int) -> int:
        """A DMA read: as the CPU's, but ``$00``/``$01`` are the RAM under the port."""
        if addr < 2:
            return self.ram[addr]
        page = self.pages[addr >> 12]
        return page[addr] if page is not None else self.io_read(addr)

    def dma_write(self, addr: int, value: int) -> None:
        """A DMA write: the I/O page now, else RAM (also under the port and ROMs)."""
        if self.io and addr >> 12 == 0xD:
            self.io_write(addr, value)
        else:
            self.ram[addr] = value

    def _request(self) -> None:
        self.dma = True

    def _sid(self, addr: int):
        dev = self.sids.get(addr & 0xFFE0)
        if dev is None and addr <= mm.SID_LAST:
            dev = self.sids.get(mm.SID_BASE)
        return dev

    def io_read(self, addr: int) -> int:
        page = addr >> 8
        if page < 0xD4:
            value = self.vic.read(addr)
        elif page < 0xD8 or page >= 0xDE:
            dev = self._sid(addr)
            reu = self.reu.read(addr) if page == 0xDF and self.reu else None
            if reu is not None:
                value = reu
            elif dev is None:
                self.seed = (self.seed * 1664525 + 1013904223) & 0xFFFFFFFF
                value = self.seed & 0xFF
            else:
                value = dev.read(self.sched.time(PHI1), addr & 0x1F)
        elif page < 0xDC:
            value = self.colour[addr & 0x3FF]
        else:
            value = (self.cia1 if page == 0xDC else self.cia2).read(addr)
        self.last_read = value
        return value

    def io_write(self, addr: int, value: int) -> None:
        page = addr >> 8
        if page < 0xD4:
            self.vic.write(addr, value)
        elif page == 0xDF and self.reu is not None:
            self.reu.write(addr, value)
        elif page < 0xD8 or page >= 0xDE:
            dev = self._sid(addr)
            if dev is not None:
                dev.write(self.sched.time(PHI1), addr & 0x1F, value)
        elif page < 0xDC:
            self.colour[addr & 0x3FF] = value & 0x0F
        else:
            (self.cia1 if page == 0xDC else self.cia2).write(addr, value)

    # --- interrupt lines -------------------------------------------------------

    def _mark(self, kind: int) -> None:
        if kind != _NMI and not any(self.asserted[k] for k in _IRQS):
            self.irq_cause = kind
        self.asserted[kind] = True
        self.on_event(self.sched.time(PHI1), SOURCES[kind])

    def _irq(self, state: bool, kind: int) -> None:
        if state:
            self._mark(kind)
            if not self.irq_count:
                self.pin = True
                self._calc()
            self.irq_count += 1
        else:
            self.asserted[kind] = False
            other = next((k for k in _IRQS if k != kind and self.asserted[k]), None)
            if other is not None:
                self.irq_cause = other
            self.irq_count -= 1
            if not self.irq_count:
                self.pin = False
                self.sched.schedule(self.clear_int, 2, PHI1)

    def _vic_irq(self, state: bool) -> None:
        self._irq(state, _VIDEO)

    def _cia_irq(self, state: bool) -> None:
        self._irq(state, _CIA)

    def _cia_nmi(self, state: bool) -> None:
        if state:
            self._mark(_NMI)
            self.nmi = True
            self._calc()

    def _set_ba(self, state: bool) -> None:
        self.rdy = state

    def _check(self) -> bool:
        return self.nmi or (self.pin and not self.iflag)

    def _calc(self) -> None:
        if self.ic == _MAX and self._check():
            self.ic = self.cc

    def _remove_irq(self) -> None:
        if not self.nmi and self.ic != _MAX:
            self.ic = _MAX

    # --- CPU -------------------------------------------------------------------

    def _decode(self) -> None:
        """Fetch the opcode at PC (``fetchNextOpcode``) and lay out its cycles."""
        mpu = self.mpu
        pc = mpu.pc
        op = self[pc]
        self.ents, self.exec_at, kind, self.base, branch = Machine.LAYOUTS[op]
        self.idx, self.rmw = 0, kind == "rmw"
        if branch:
            mask, want = branch
            self.taken = bool(mpu.p & mask) == bool(want)
            rel = self[(pc + 1) & 0xFFFF]
            nxt = (pc + 2) & 0xFFFF
            target = (nxt + rel - (0x100 if rel & 0x80 else 0)) & 0xFFFF
            self.cross = (target ^ nxt) & 0xFF00 != 0
        elif kind:
            lo = self[(pc + 1) & 0xFFFF]
            if kind == "indy":
                lo = self[lo]
            self.cross = lo + (mpu.x if kind == "absx" else mpu.y) > 0xFF
        if not self._check():
            self.ic = _MAX
        elif self.ic != _MAX:
            self.ic = -_MAX

    def _fetch(self) -> None:
        """``interruptsAndNextOpcode``: start an interrupt or the next opcode."""
        ic = self.ic
        if ic != _MAX and self.cc > ic + 2:
            self.ents, self.idx, self.base = BRK_ENTRIES, 0, 0
            self.d1x1, self.ic = True, _MAX
        else:
            self._decode()

    def _push(self, value: int) -> None:
        mpu = self.mpu
        self.ram[0x100 | mpu.sp] = value
        mpu.sp = (mpu.sp - 1) & 0xFF

    def _stall(self, cc: int) -> None:
        if cc == _CLI_BASE:
            self.iflag = False
            if self.pin and self.ic == _MAX:
                self.ic = -_MAX
        elif cc == _SEI_BASE:
            self.iflag = True
            if not self.nmi and cc <= self.ic + 2:
                self.ic = _MAX
        if self.ic == cc:
            self.ic = cc - 1

    def _special(self, kind: int, t: int) -> None:
        # pylint: disable=too-many-branches
        mpu = self.mpu
        if kind == F:
            self._fetch()
        elif kind == W:
            if self.pending:
                self.io_write(*self.pending.pop(0))
        elif kind == X2:
            if not self.cross:
                self.idx += 1
                self.cc += 1
        elif kind == BR:
            if not self.taken:
                self._fetch()
            elif not self.cross:
                self.idx += 1
                self.cc += 1
                if self.ic >> 3 == self.cc >> 3:
                    self.ic += 2
        elif kind == CLI:
            self.iflag = False
            self._calc()
            self._fetch()
        elif kind == SEI:
            self.iflag = True
            self._fetch()
            if not self.nmi and self.ic != _MAX:
                self.ic = _MAX
        elif kind == POPSR:
            self.iflag = bool(mpu.p & _IRQ_I)
            self._calc()
        elif kind == RTI:
            self.on_event(t + 1, "rti")
            self._fetch()
        elif kind == BRK0:
            if not self.d1x1:
                mpu.pc = (mpu.pc + 2) & 0xFFFF
        elif kind == PUSHPCH:
            self._push(mpu.pc >> 8)
        elif kind == LOWPC:
            self._push(mpu.pc & 0xFF)
            nmi = self.nmi
            self.vec_addr = mm.CPU_NMI_VECTOR if nmi else mm.CPU_IRQ_VECTOR
            self.nmi = False
            self._calc()
            name = "brk"
            if nmi:
                name = SOURCES[_NMI]
            elif self.d1x1:
                name = SOURCES[self.irq_cause]
            self.on_event(t + 1, name + "_entry")
        elif kind == PUSHSR:
            self._push((mpu.p & ~0x30) | (0x20 if self.d1x1 else 0x30))
        elif kind == VECLO:
            self.target = self[self.vec_addr]
            self.d1x1 = False
        elif kind == VECHI:
            self.target |= self[self.vec_addr + 1] << 8
            self.iflag = True
            mpu.p |= _IRQ_I
        elif kind == FN:
            mpu.pc = self.target
            self._decode()
        elif kind == HALT:
            self.jammed = True
        elif kind == ANE:
            value = (mpu.a | _ANE_MAGIC) & mpu.x & self[(mpu.pc + 1) & 0xFFFF]
            mpu.a = value
            mpu.p = (mpu.p & 0x7D) | (value & 0x80) | (0 if value else 0x02)
            mpu.pc = (mpu.pc + 2) & 0xFFFF

    def run(self, limit: int) -> None:
        """Run the CPU and the chips up to cycle ``limit``."""
        sched, mpu = self.sched, self.mpu
        t = self.t
        while t < limit and not self.jammed:
            t2 = t << 1
            if sched.next <= t2:
                self.cc = self.base + self.idx
                sched.run(t2)
            kind = self.ents[self.idx]
            sched.now = t2 + 1
            if kind >= R and (not self.rdy or self.dma):
                go = False
                if self.dma:
                    self.cc = self.base + self.idx
                    go = self.reu.step(not self.rdy)
                    self.dma = not go
                if not go:
                    self._stall(self.base + self.idx)
                    t += 1
                    continue
            self.idx += 1
            self.cc = self.base + self.idx
            if kind == R:
                if self.idx == self.exec_at:
                    mpu.step()
                    if self.rmw and self.pending:
                        self.pending.insert(0, (self.pending[0][0], self.last_read))
            else:
                self.t = t
                self._special(kind, t)
            t += 1
        self.t = t

    def reset(self, entry: int, cycle: Optional[int] = None) -> int:
        """Bring the chips to ``cycle`` (:func:`boot_cycle`) and start the CPU at ``entry``.

        The CPU port is set to ``$2F``/``$37`` and the registers cleared, as
        sidplayfp's reset leaves them; RAM is whatever the caller wrote to it.
        """
        cycle = boot_cycle(self.ntsc) if cycle is None else cycle
        self.sched.run(cycle << 1)
        self.t = cycle
        self[mm.CPU_DDR], self[mm.CPU_PORT] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
        mpu = self.mpu
        mpu.sp, mpu.p, mpu.a, mpu.x, mpu.y = 0xFF, 0, 0, 0, 0
        self.target = entry
        self.ents, self.idx, self.base, self.cc = BRK_ENTRIES, 6, 0, 6
        return cycle


__all__ = [
    "NATIVE",
    "BRK_ENTRIES",
    "FETCHES",
    "Machine",
    "SOURCES",
    "boot_cycle",
    "opcode_layout",
]
