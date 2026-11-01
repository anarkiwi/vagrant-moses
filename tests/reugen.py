"""Seeded REU test programs: register sequences assembled into PRGs for any C64 with an REU.

A program fills C64 RAM, colour RAM and the VIC colours from the REU image, then runs
ops that each program the REC, trigger it (``$DF01`` or ``$FF00`` under any ``$01``
banking) and record CIA timer reads around the halt, the registers and any IRQ.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from vagrant_moses.bus import banking
from vagrant_moses.machine import Machine
from vagrant_moses.reu import FROM_REU, SIZES, SWAP, TO_REU, VERIFY, geometry

from asm import assemble

LOAD, CODE, RESULTS, OPNO, IRQIDX, IRQLOG, END = (
    0x0800,
    0x0801,
    0x2000,
    0x26FE,
    0x26FF,
    0x2700,
    0x2800,
)
SLOT = 24
MAX_OPS = (OPNO - RESULTS) // SLOT
CONTROLS = (0x00, 0x40, 0x80, 0xC0)
TRIGGERS = ("now", "ff00")
CELLS = tuple(product(range(4), CONTROLS, (False, True), TRIGGERS))
KINDS = ("to_reu", "from_reu", "swap", "verify")
FEATURES = (
    "len64k",
    "c64_wrap",
    "reu_wrap",
    "bank_carry",
    "beyond_size",
    "io_read",
    "io_write",
    "rom_read",
    "port_ram",
    "verify_prep",
    "verify_none",
    "verify_first",
    "verify_second",
    "verify_mid",
    "verify_next_to_last",
    "verify_last",
    "irq_taken",
    "late_mask",
    "decoy_ff00",
    "stale_ff00",
    "rearm",
    "modify_armed",
    "half_regs",
    "inc_ff00",
    "indexed_ff00",
    "sweep",
    "port_prep",
)
_LENGTHS = ((1, 4), (5, 64), (65, 1024), (1025, 12288))
_VECTORS = (0x0314, 0x0315, 0xFFFA, 0xFFFB, 0xFFFC, 0xFFFD, 0xFFFE, 0xFFFF)
_POKE = np.zeros(0x10000, bool)
_POKE[0x0002:0x0100] = _POKE[0x0200:0x0800] = _POKE[END:0xA000] = True
_POKE[0xC000:0xD000] = True
_FILLS = (
    (0x0002, 0x00FE, 0x37),
    (0x0100, 0x0100, 0x37),
    (0x0200, 0x0600, 0x37),
    (END, 0x10000 - END, 0x34),
    (0xD800, 0x0400, 0x37),
    (0xD020, 0x000F, 0x37),
)


def _io_tables():
    page = np.arange(0xD000, 0xE000)
    vic, reg, cia = page < 0xD400, page & 0x3F, (page >= 0xDC00) & (page < 0xDE00)
    read = vic & (reg >= 0x20) | cia & np.isin(page & 0xF, (4, 5, 6, 7, 0xE, 0xF))
    write = vic & (reg >= 0x20) & (reg < 0x2F) | (page >= 0xD400) & (page < 0xDC00)
    return read, write | (page >= 0xDE00)


def _tables():
    """Per ``$01`` config: where a DMA read is reproducible and a DMA write harmless."""
    io_read, io_write = _io_tables()
    readable = np.ones((8, 0x10000), bool)
    writable = np.ones((8, 0x10000), bool)
    rom = np.zeros((8, 0x10000), bool)
    io = np.zeros((8, 0x10000), bool)
    for cfg in range(8):
        basic, kernal, chars, has_io = banking(cfg)
        rom[cfg, 0xA000:0xC000] = basic
        rom[cfg, 0xE000:] = kernal
        rom[cfg, 0xD000:0xE000] = chars
        if has_io:
            io[cfg, 0xD000:0xE000] = True
            readable[cfg, 0xD000:0xE000] = io_read
            writable[cfg, 0xD000:0xE000] = io_write
    writable[:, 0x01E0:0x0200] = writable[:, 0x0800:END] = False
    return readable, writable, rom, io


READABLE, WRITABLE, ROM, IO = _tables()
_ROMS = [
    [
        (lo, hi)
        for lo, hi, on in zip(
            (0xA000, 0xD000, 0xE000),
            (0xC000, 0xE000, 0x10000),
            (banking(cfg)[0], banking(cfg)[2], banking(cfg)[1]),
        )
        if on
    ]
    for cfg in range(8)
]


@dataclass
class Op:
    """One REC programming: what is written, how it is triggered, and its edge cases."""

    # pylint: disable=too-many-instance-attributes

    kind: int
    control: int
    autoload: bool
    trigger: str
    cfg: int = 7
    c64: int = 0
    raddr: int = 0
    length: int = 1
    int_mask: int = 0
    cli: bool = False
    features: set = field(default_factory=set)
    writes: List[Tuple[int, int]] = field(default_factory=list)
    poke: Optional[int] = None
    late_mask: Optional[int] = None
    armed_write: Optional[Tuple[int, int]] = None
    rearm: Optional[int] = None
    ff00_insn: str = "STA"

    @property
    def cell(self):
        """The op's (transfer type, address control, autoload, trigger) cell."""
        return self.kind, self.control, self.autoload, self.trigger

    @property
    def command(self) -> int:
        """The command byte that runs the op."""
        trigger = 0x10 if self.trigger == "now" else 0
        return 0x80 | trigger | (0x20 if self.autoload else 0) | self.kind


@dataclass
class Program:
    """A generated program: its ops, the PRG and the REU's initial image seed."""

    name: str
    seed: str
    size_kb: int
    ops: List[Op]
    prg: bytes
    done: int

    @property
    def image(self) -> bytes:
        """The REU's contents at power-on."""
        return random.Random(self.seed + ":image").randbytes(self.size_kb << 10)

    @property
    def key(self) -> str:
        """A digest of everything the run depends on."""
        return hashlib.sha256(
            self.prg + self.size_kb.to_bytes(4, "little") + self.seed.encode()
        ).hexdigest()[:24]


_GROUPS = {"c64": (2, 3), "base": (4, 5), "bank": (6,), "len": (7, 8)}


def _regs(c64: int, raddr: int, length: int, bank: int) -> Dict[int, int]:
    return {
        2: c64 & 0xFF,
        3: c64 >> 8,
        4: raddr & 0xFF,
        5: raddr >> 8 & 0xFF,
        6: bank,
        7: length & 0xFF,
        8: length >> 8,
    }


class _State:
    """What the generator knows of the REC and the RAM under the CPU port.

    A 16-bit register is known when it equals its shadow combination: after a write to
    either byte (VICE writes reg and shadow together) or an autoload transfer.
    """

    def __init__(self):
        start, n, _ = _FILLS[-1]
        self.shadow = _regs(start, start, n, 0)
        self.known = dict.fromkeys(_GROUPS, False)
        self.port_clean = np.zeros(2, bool)

    def prepped(self, op: Op) -> None:
        """The verify prep's autoload transfer leaves its registers in place."""
        self.shadow = _regs(op.c64, op.raddr, op.length, op.raddr >> 16)
        self.known = dict.fromkeys(_GROUPS, True)

    def after(self, op: Op) -> None:
        """Forget what the op's transfer leaves unpredictable."""
        if not op.autoload:
            self.known["len"] = False
            self.known["c64"] &= bool(op.control & 0x80)
            self.known["base"] &= bool(op.control & 0x40)
            self.known["bank"] &= bool(op.control & 0x40)


def _sequence(start: int, n: int, step: int) -> np.ndarray:
    return (start + np.arange(n) * step) & 0xFFFF


class _Gen:
    """Draws ops for one program, keeping every DMA access reproducible."""

    def __init__(self, rnd: random.Random, size_kb: int):
        self.rnd, self.size_kb = rnd, size_kb
        self.wrap, _, _, self.bank_unused, _ = geometry(size_kb)
        self.state = _State()

    def _raddr(self, op: Op, n: int) -> int:
        rnd, size = self.rnd, self.size_kb << 10
        top = 0x80000 if self.size_kb <= 512 else 1 << 24
        mode = rnd.choice(("rand", "rand", "wrap", "bank", "beyond", "shadow"))
        if mode == "shadow":
            hi = self.state.shadow[5] << 8
            return (
                rnd.randrange(max(1, size >> 16)) << 16 | hi | rnd.randrange(256)
            ) % top
        if op.control & 0x40 or mode == "rand" or n < 2:
            return rnd.randrange(size)
        k = rnd.randrange(1, min(n, 0x10000))
        if mode == "wrap":
            op.features.add("reu_wrap")
            return rnd.randrange(max(1, top // 0x80000)) << 19 | (self.wrap - k)
        if mode == "bank":
            op.features.add("bank_carry")
            return (rnd.randrange(max(1, size >> 16)) << 16 | (0x10000 - k)) % top
        if size < top:
            op.features.add("beyond_size")
            return rnd.randrange(size, top)
        return rnd.randrange(size)

    def _host(self, op: Op, n: int) -> int:
        rnd = self.rnd
        mode = rnd.choice(("rand", "rand", "wrap", "io", "rom"))
        cleans = op.kind in (FROM_REU, SWAP) and op.cfg == 7 and op.trigger == "now"
        if (
            cleans and not op.cli or self.state.port_clean.all()
        ) and rnd.random() < 0.5:
            mode = "wrap"
        if op.kind == VERIFY and not op.control and rnd.random() < 0.5:
            return rnd.randrange(END, 0xA000 - min(n, 0x7000))
        if mode == "wrap" and not op.control & 0x80 and n > 1:
            return 0x10000 - rnd.randrange(1, min(n, 0x10000))
        if rnd.random() < 0.15:
            shadow = self.state.shadow
            pick = (shadow[3] << 8, rnd.randrange(256)), (
                rnd.randrange(256) << 8,
                shadow[2],
            )
            return sum(rnd.choice(pick))
        if mode == "io" and IO[op.cfg].any():
            return rnd.choice(
                (0xD020, 0xD02F, 0xD800, 0xDC04, 0xDD04, 0xD400, 0xDE00)
            ) + (rnd.randrange(8))
        if mode == "rom" and _ROMS[op.cfg]:
            return rnd.randrange(*rnd.choice(_ROMS[op.cfg]))
        return rnd.randrange(0x10000)

    def _ok(self, op: Op, n: int) -> bool:
        seq = _sequence(op.c64, n, 0 if op.control & 0x80 else 1)
        readable = READABLE[op.cfg].copy()
        readable[:2] = self.state.port_clean
        writable = WRITABLE[op.cfg].copy()
        if op.cli:
            writable[list(_VECTORS)] = False
        reads = seq
        if op.kind == TO_REU and op.control & 0x40:
            reads = seq[-1:]
        if op.kind != FROM_REU and not readable[reads].all():
            return False
        if op.kind in (FROM_REU, SWAP) and not writable[seq].all():
            return False
        return True

    def _tag(self, op: Op, n: int) -> None:
        seq = _sequence(op.c64, n, 0 if op.control & 0x80 else 1)
        if n == 0x10000:
            op.features.add("len64k")
        if not op.control & 0x80 and op.c64 + n > 0x10000:
            op.features.add("c64_wrap")
        if IO[op.cfg][seq].any():
            op.features.add("io_read" if op.kind != FROM_REU else "io_write")
            if op.kind == SWAP:
                op.features.add("io_write")
        if op.kind != FROM_REU and ROM[op.cfg][seq].any():
            op.features.add("rom_read")
        if op.kind != FROM_REU and (seq < 2).any():
            op.features.add("port_ram")

    def _place(self, op: Op) -> None:
        rnd = self.rnd
        lengths = _LENGTHS[:2] + ((65, 256),) if "sweep" in op.features else _LENGTHS
        for _ in range(400):
            n = rnd.randrange(*rnd.choice(lengths))
            if rnd.random() < 0.15 and "sweep" not in op.features:
                n = (self.state.shadow[8] << 8 | rnd.randrange(256)) or 0x10000
            elif rnd.random() < 0.08 and "sweep" not in op.features:
                n = 0x10000
            op.c64 = self._host(op, n)
            if self._ok(op, n):
                break
        else:
            n, op.c64 = rnd.randrange(1, 64), rnd.randrange(END, 0x9000)
        op.length = n & 0xFFFF
        op.raddr = self._raddr(op, n)
        self._tag(op, n)
        if op.kind in (FROM_REU, SWAP):
            seq = _sequence(op.c64, n, 0 if op.control & 0x80 else 1)
            hit = np.isin((0, 1), seq)
            self.state.port_clean |= hit

    def _verify(self, op: Op) -> None:
        rnd, n = self.rnd, op.length or 0x10000
        self.state.prepped(op)
        if op.control:
            return
        ks = {"verify_first": 0, "verify_second": 1, "verify_mid": n // 2}
        ks |= {"verify_next_to_last": n - 2, "verify_last": n - 1}
        name = rnd.choice(sorted(ks) + ["verify_none"])
        k = ks.get(name)
        if k is not None and 0 <= k < n:
            addr = (op.c64 + k) & 0xFFFF
            if not _POKE[addr] or ROM[op.cfg][addr]:
                return
            op.poke = addr
        if k is None or op.poke is not None:
            op.features.add(name)

    def _writes(self, op: Op) -> None:
        rnd, state = self.rnd, self.state
        bank = op.raddr >> 16 | (rnd.randrange(256) & self.bank_unused)
        values = _regs(op.c64, op.raddr, op.length, bank)
        values |= {9: op.int_mask, 10: op.control | rnd.randrange(64)}
        chosen = []
        half = rnd.random() < 0.4
        for name, regs in _GROUPS.items():
            need = [r for r in regs if state.shadow[r] != values[r]]
            if not half:
                need = list(regs)
            elif not state.known[name] and not need:
                need = [rnd.choice(regs)]
            if not state.known[name] and len(need) < len(regs):
                op.features.add("half_regs")
            chosen += need
            state.known[name] = True
        chosen += [9, 10]
        rnd.shuffle(chosen)
        op.writes = [(r, values[r]) for r in chosen]
        state.shadow.update({r: values[r] for r in chosen})

    def op(self, cell) -> Op:
        """An op in ``cell`` with its edge cases drawn."""
        rnd = self.rnd
        op = Op(*cell)
        op.cfg = rnd.choice((5, 6, 7, 7)) if op.trigger == "now" else rnd.randrange(8)
        if op.trigger == "now" and self.state.port_clean.all() and rnd.random() < 0.8:
            op.cfg = 7
        op.int_mask = rnd.choice((0, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0))
        op.int_mask |= rnd.randrange(32)
        op.cli = rnd.random() < 0.4
        if op.trigger == "now" and op.autoload and not op.cli and rnd.random() < 0.3:
            op.features.add("sweep")
        if op.kind == VERIFY and (not op.control or rnd.random() < 0.7):
            op.features.add("verify_prep")
        if op.cfg != 7 or op.trigger == "ff00":
            self.state.port_clean[1] = False
        elif op.kind != FROM_REU and not self.state.port_clean.all():
            if rnd.random() < 0.3:
                op.features.add("port_prep")
                self.state.port_clean[:] = True
                self.state.shadow = _regs(0, 0, 2, 0)
                self.state.known = dict.fromkeys(_GROUPS, True)
        self._place(op)
        if "verify_prep" in op.features:
            self._verify(op)
        if rnd.random() < 0.2:
            op.late_mask = rnd.choice((0xC0, 0xA0, 0xE0, 0x40))
            op.features.add("late_mask")
        for name in ("decoy_ff00", "stale_ff00"):
            if rnd.random() < 0.15:
                op.features.add(name)
        if op.trigger == "ff00":
            op.ff00_insn = rnd.choice(("STA", "STA", "INC", "STAX"))
            if op.ff00_insn != "STA":
                op.features.add("inc_ff00" if op.ff00_insn == "INC" else "indexed_ff00")
            if rnd.random() < 0.15:
                op.rearm = rnd.randrange(4)
                op.features.add("rearm")
        self._writes(op)
        if op.trigger == "ff00" and rnd.random() < 0.15:
            op.int_mask = rnd.randrange(256) | 0x1F
            op.armed_write = (9, op.int_mask)
            op.features.add("modify_armed")
        if op.cli and op.int_mask & 0x80 and op.int_mask & 0x60:
            op.features.add("irq_taken")
        if op.cfg != 7 or op.cli or op.trigger == "ff00":
            self.state.port_clean[1] = False
        self.state.after(op)
        return op


def _lda_sta(value: int, addr: int) -> List[str]:
    return [f"LDA #${value:02X}", f"STA ${addr:04X}"]


def _timer(slot: int) -> List[str]:
    return ["LDA $DD04", f"STA ${slot:04X}", "LDA $DD05", f"STA ${slot + 1:04X}"]


def _fill(start: int, n: int, cfg: int, autoload: bool = False) -> List[str]:
    regs = {2: start & 0xFF, 3: start >> 8, 4: start & 0xFF, 5: start >> 8, 6: 0}
    regs |= {7: n & 0xFF, 8: n >> 8 & 0xFF, 9: 0, 10: 0}
    out = [line for reg, v in regs.items() for line in _lda_sta(v, 0xDF00 + reg)]
    if cfg == 0x37:
        return out + _lda_sta(0xB1 if autoload else 0x91, 0xDF01) + ["LDA $DF00"]
    out += _lda_sta(0x81, 0xDF01) + _lda_sta(cfg, 0x01) + ["STA $FF00"]
    return out + _lda_sta(0x37, 0x01) + ["LDA $DF00"]


def _prep(op: Op) -> List[str]:
    regs = {2: op.c64 & 0xFF, 3: op.c64 >> 8, 4: op.raddr & 0xFF}
    regs |= {5: op.raddr >> 8 & 0xFF, 6: op.raddr >> 16, 7: op.length & 0xFF}
    regs |= {8: op.length >> 8, 9: 0, 10: op.control}
    out = [line for reg, v in regs.items() for line in _lda_sta(v, 0xDF00 + reg)]
    if op.cfg == 7:
        out += _lda_sta(0xB0, 0xDF01)
    elif op.cfg in (5, 6):
        out += _lda_sta(0x30 | op.cfg, 0x01) + _lda_sta(0xB0, 0xDF01)
    else:
        out += _lda_sta(0xA0, 0xDF01) + _lda_sta(0x30 | op.cfg, 0x01) + ["STA $FF00"]
    out += (_lda_sta(0x37, 0x01) if op.cfg != 7 else []) + ["LDA $DF00"]
    if op.poke is not None:
        out += [f"LDA ${op.poke:04X}", "EOR #$FF", f"STA ${op.poke:04X}"]
    return out


def _op(i: int, op: Op) -> List[str]:
    res = RESULTS + SLOT * i
    out = _lda_sta(i, OPNO)
    for vec, target in ((0x0314, "irq_body"), (0xFFFE, "irq_raw")):
        out += [f"LDA #<{target}", f"STA ${vec:04X}", f"LDA #>{target}"]
        out += [f"STA ${vec + 1:04X}"]
    if "port_prep" in op.features:
        out += _fill(0, 2, 0x37, autoload=True)
    if "verify_prep" in op.features:
        out += _prep(op)
    if "decoy_ff00" in op.features:
        out.append("STA $FF00")
    for reg, value in op.writes:
        out += _lda_sta(value, 0xDF00 + reg)
    if op.trigger == "now":
        out += _lda_sta(0x30 | op.cfg, 0x01) if op.cfg != 7 else []
        out += ["CLI"] if op.cli else []
        trigger = [f"LDA #${op.command:02X}", "STA $DF01"]
        if "sweep" in op.features:
            trigger = ["LDY #$00", f"sweep{i}: " + trigger[0], trigger[1], "INY"]
            trigger.append(f"BNE sweep{i}")
        out += _timer(res) + trigger
    elif op.rearm is not None:
        out += _lda_sta(op.command & ~3 | op.rearm, 0xDF01)
        out += _lda_sta(op.command & 0x7F, 0xDF01)
    else:
        out += _lda_sta(op.command, 0xDF01)
    if op.trigger == "ff00":
        if op.armed_write is not None:
            out += _lda_sta(op.armed_write[1], 0xDF00 + op.armed_write[0])
        out += ["CLI"] if op.cli else []
        out += _timer(res) + ["LDX #$01"] + _lda_sta(0x30 | op.cfg, 0x01)
        out.append(
            {"STA": "STA $FF00", "INC": "INC $FF00", "STAX": "STA $FEFF,X"}[
                op.ff00_insn
            ]
        )
    out += _lda_sta(0x37, 0x01) if op.cfg != 7 or op.trigger == "ff00" else []
    out += _timer(res + 2)
    if op.late_mask is not None:
        out += _lda_sta(op.late_mask, 0xDF09)
    out.append("SEI")
    for k, reg in enumerate((0, 0, *range(1, 11), 0x0B, 0x1F, 0x21, 0xE6, 0x40)):
        out += [f"LDA ${0xDF00 + reg:04X}", f"STA ${res + 4 + k:04X}"]
    if "stale_ff00" in op.features:
        out += ["STA $FF00", "LDA $DF00", f"STA ${res + 21:04X}"]
    return out


_PROLOGUE = [
    "SEI",
    "CLD",
    "LDX #$FF",
    "TXS",
    *_lda_sta(0x2F, 0x00),
    *_lda_sta(0x37, 0x01),
    "LDA #$7F",
    "STA $DC0D",
    "STA $DD0D",
    "LDA $DC0D",
    "LDA $DD0D",
    "LDA #$00",
    "STA $D011",
    "STA $D015",
    "STA $D01A",
    f"STA ${IRQIDX:04X}",
    "LDA #$FF",
    "STA $DD04",
    "STA $DD05",
    *_lda_sta(0x11, 0xDD0E),
]
_HANDLER = [
    "irq_raw: PHA",
    "TXA",
    "PHA",
    "TYA",
    "PHA",
    "irq_body: LDA $01",
    "PHA",
    *_lda_sta(0x37, 0x01),
    "LDA $DD04",
    f"LDX ${IRQIDX:04X}",
    f"STA ${IRQLOG:04X},X",
    "LDA $DD05",
    f"STA ${IRQLOG + 1:04X},X",
    "LDA $DF00",
    f"STA ${IRQLOG + 2:04X},X",
    f"LDA ${OPNO:04X}",
    f"STA ${IRQLOG + 3:04X},X",
    "TXA",
    "CLC",
    "ADC #$04",
    f"STA ${IRQIDX:04X}",
    "PLA",
    "STA $01",
    "PLA",
    "TAY",
    "PLA",
    "TAX",
    "PLA",
    "RTI",
]


def _display() -> List[str]:
    """Lock to the raster phase, turn the display on at line ``$20`` and restart the timer.

    ``$D012`` is read every 62 cycles until two reads see one line: the earlier read
    was then on the line's first cycle; a 63-cycle wait keeps that phase.
    """
    delay = ["NOP"] * 22 + ["BIT $FB"]
    return [
        "sync0: BIT $D011",
        "BMI sync0",
        "LDA $D012",
        "CMP #$10",
        "BNE sync0",
        "LDA $D012",
        "BIT $FB",
        "JMP sync1",
        "sync1: TAX",
        "STX $FB",
        *delay,
        "LDA $D012",
        "CMP $FB",
        "BNE sync1",
        "sync2: NOP",
        *["NOP"] * 26,
        "LDA $D012",
        "CMP #$20",
        "BNE sync2",
        *_lda_sta(0x1B, 0xD011),
        *_lda_sta(0x11, 0xDD0E),
        "LDX #$00",
    ]


def build(name: str, seed: str, size_kb: int, cells, display: bool = False) -> Program:
    """The program for ``seed`` running one op per cell in ``cells``.

    ``display`` turns the screen on first, so badlines halt both the CPU and the DMA.
    """
    gen = _Gen(random.Random(seed), size_kb)
    ops = [gen.op(cell) for cell in cells[:MAX_OPS]]
    lines = list(_PROLOGUE) + (_display() if display else [])
    for start, n, cfg in _FILLS:
        lines += _fill(start, n, cfg)
    for i, op in enumerate(ops):
        lines += _op(i, op)
    lines += ["done: JMP done"] + _HANDLER
    code, labels = assemble("\n".join(lines), CODE)
    if CODE + len(code) > RESULTS:
        raise ValueError(f"{name}: code overruns the results at ${RESULTS:04X}")
    image = bytearray(END - LOAD)
    image[CODE - LOAD : CODE - LOAD + len(code)] = code
    prg = LOAD.to_bytes(2, "little") + bytes(image)
    return Program(name, seed, size_kb, ops, prg, labels["done"])


def corpus(
    count: int = 48, seed: int = 1764, ops: int = 16, display: bool = False
) -> List[Program]:
    """``count`` programs cycling the REU sizes, together covering every cell."""
    cells = list(CELLS)
    random.Random(seed).shuffle(cells)
    out = []
    tag = "d" if display else ""
    for i in range(count):
        chunk = [cells[(i * ops + k) % len(cells)] for k in range(ops)]
        size = SIZES[i % len(SIZES)]
        name, key = f"reu{tag}{i:04d}", f"{seed}:{i}" + (":display" if display else "")
        out.append(build(name, key, size, chunk, display))
    return out


def coverage(programs) -> Dict[str, Dict]:
    """Op counts per cell and per (feature, transfer type)."""
    cells = {cell: 0 for cell in CELLS}
    features = {(f, k): 0 for f in FEATURES for k in KINDS}
    for program in programs:
        for op in program.ops:
            cells[op.cell] += 1
            for name in op.features:
                features[(name, KINDS[op.kind])] += 1
    return {"cells": cells, "features": features}


@dataclass
class Observation:
    """What a run leaves: C64 RAM, colour RAM nibbles and the REU, port bytes masked."""

    ram: np.ndarray
    colour: np.ndarray
    reu: np.ndarray

    def digest(self) -> Dict[str, str]:
        """SHA-256 prefixes of each part."""
        return {
            part: hashlib.sha256(getattr(self, part).tobytes()).hexdigest()[:24]
            for part in ("ram", "colour", "reu")
        }

    @classmethod
    def of(cls, ram, colour, reu) -> "Observation":
        """Normalise raw dumps: zero ``$00``/``$01`` and the colour high nibbles."""
        ram = np.frombuffer(bytes(ram), np.uint8).copy()
        ram[:2] = 0
        colour = np.frombuffer(bytes(colour), np.uint8) & 0x0F
        return cls(ram, colour, np.frombuffer(bytes(reu), np.uint8).copy())


def run(program: Program, backend: Optional[str] = None, limit: int = 4_000_000):
    """Run ``program`` on the machine; the :class:`Observation` at ``done``."""
    machine = Machine(reu=program.size_kb, backend=backend)
    machine.ram[LOAD:END] = program.prg[2:]
    machine.reu.ram[:] = program.image
    start = machine.reset(CODE)
    while machine.mpu.pc != program.done and machine.cycle < start + limit:
        machine.run(machine.cycle + 5000)
    if machine.mpu.pc != program.done:
        raise RuntimeError(
            f"{program.name} did not reach done (pc ${machine.mpu.pc:04X})"
        )
    return Observation.of(machine.ram[:], machine.colour[:], machine.reu.ram[:])


def results(observation: Observation, program: Program):
    """``(per-op result rows, IRQ log rows)`` decoded from the RAM."""
    ram = observation.ram
    rows = ram[RESULTS : RESULTS + SLOT * len(program.ops)].reshape(-1, SLOT)
    log = ram[IRQLOG : IRQLOG + int(ram[IRQIDX])].reshape(-1, 4)
    return rows, log
