"""Driver programs: a register machine that drives the keyboard and runs a machine to pcs.

A :class:`Program` is built with labels and runs on either backend (``machine.run_program``);
:func:`run_forks` runs many on forks of one machine, natively on threads with the GIL
released. ``docs/native.md`` (Driver programs) has the semantics.
"""

from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional, Tuple, Union

STATUSES = ("halt", "budget", "jam", "ops", "fault")
ALU = ("MOV", "ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR", "ADD8", "SUB8")
CONDS = ("EQ", "NE", "LT", "GE")
LD_CPU, LD_RAM, LD_IO = range(3)
MAX_OPS = 1 << 24
_U32 = 0xFFFFFFFF


class Reg(NamedTuple):
    """Register ``n`` (``R[n]``)."""

    n: int


R = tuple(Reg(n) for n in range(16))
Src = Union[Reg, int]


class Outcome(NamedTuple):
    """How a program ended: ``status`` (one of :data:`STATUSES`), its ``code`` (the
    ``HALT`` code; the op index for ``budget``, ``jam`` and ``ops``; the address for
    ``fault``), the machine's ``cycle``, the marker arrivals counted and the outputs."""

    status: str
    code: int
    cycle: int
    frames: int
    outputs: Tuple[int, ...]


class _Label(NamedTuple):
    name: str


def peekable(addr: int) -> bool:
    """Whether ``addr`` has a side-effect-free I/O view: the VIC-II, colour RAM, the CIAs."""
    return 0xD000 <= addr < 0xD400 or 0xD800 <= addr < 0xDE00


def _reg(r) -> int:
    if not isinstance(r, Reg) or not 0 <= r.n < len(R):
        raise ValueError(f"{r!r} is not a register")
    return r.n


def _src(s) -> Tuple[int, int]:
    if isinstance(s, Reg):
        return 1, _reg(s)
    if not isinstance(s, int) or not 0 <= s <= _U32:
        raise ValueError(f"{s!r} is neither a register nor a u32")
    return 0, s


def _int(v, lo: int, hi: int, what: str) -> int:
    if not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"{what} must be in [{lo}, {hi}], got {v!r}")
    return v


def _word(v) -> int:
    return _int(v, 0, 0xFFFF, "an address")


class Program:
    """A driver program: ops appended by the builder methods (each returns ``self``),
    ``label(name)`` marking the next op as a branch target.

    ``budget`` caps the cycles the program runs from its start; ``marker`` is the pc whose
    arrivals every run op counts (``FRAMES``); ``max_ops`` caps the ops executed.
    """

    def __init__(
        self, budget: int, marker: Optional[int] = None, max_ops: int = MAX_OPS
    ):
        self.budget = _int(budget, 0, (1 << 63) - 1, "budget")
        self.marker = None if marker is None else _word(marker)
        self.max_ops = _int(max_ops, 0, (1 << 64) - 1, "max_ops")
        self._ops: List[tuple] = []
        self._labels = {}
        self._code = self._native = None

    def _emit(self, *op) -> "Program":
        self._ops.append(op)
        self._code = self._native = None
        return self

    def label(self, name: str) -> "Program":
        """The next op is the target ``name``."""
        if name in self._labels:
            raise ValueError(f"label {name!r} is defined twice")
        self._labels[name] = len(self._ops)
        self._code = self._native = None
        return self

    # --- keyboard --------------------------------------------------------------

    def keys(self, rows) -> "Program":
        """``KEYS``: set the whole keyboard matrix (8 row bytes, as ``keymatrix``)."""
        rows = bytes(rows)
        if len(rows) != 8:
            raise ValueError("the keyboard matrix has 8 rows")
        return self._emit("KEYS", *rows)

    def clear(self) -> "Program":
        """``KEYS_CLEAR``: release every key."""
        return self.keys(bytes(8))

    def key(self, row: int, col: int, down: bool = True) -> "Program":
        """``KEY``: hold or release the key at PA line ``row``, PB line ``col``."""
        return self._emit(
            "KEY", _int(row, 0, 7, "row"), _int(col, 0, 7, "col"), int(bool(down))
        )

    def keyr(self, r: Reg, down: bool = True) -> "Program":
        """``KEYR``: hold or release key ``row * 8 + col`` given by ``r``'s low 6 bits."""
        return self._emit("KEYR", _reg(r), int(bool(down)))

    # --- running ---------------------------------------------------------------

    def run_to(self, pcs: Union[int, Iterable[int]], leave: bool = False) -> "Program":
        """``RUN_TO``: run until the CPU is about to execute a pc in ``pcs``; at once when
        the machine is parked at one already, unless ``leave``."""
        pcs = frozenset((pcs,) if isinstance(pcs, int) else pcs)
        return self._emit("RUN", frozenset(map(_word, pcs)), 0, 1, int(not leave))

    def count_to(self, pc: int, n: Src) -> "Program":
        """``COUNT_TO``: run until the CPU has been about to execute ``pc`` ``n`` more times."""
        return self._emit("RUN", frozenset((_word(pc),)), *_src(n), 0)

    def step(self, n: Src = 1) -> "Program":
        """``STEP``: run ``n`` decodes on (``n`` instructions)."""
        return self._emit("RUN", None, *_src(n), 0)

    # --- registers -------------------------------------------------------------

    def _ld(self, mode, r, addr, idx) -> "Program":
        return self._emit(
            "LD", _reg(r), _word(addr), -1 if idx is None else _reg(idx), mode
        )

    def ld(self, r: Reg, addr: int, idx: Optional[Reg] = None) -> "Program":
        """``LD``: ``r = byte at addr (+ idx)`` as the CPU sees it, without side effects."""
        return self._ld(LD_CPU, r, addr, idx)

    def ldr(self, r: Reg, addr: int, idx: Optional[Reg] = None) -> "Program":
        """``LDR``: ``r = RAM[addr (+ idx)]``, whatever the banking."""
        return self._ld(LD_RAM, r, addr, idx)

    def ldio(self, r: Reg, addr: int, idx: Optional[Reg] = None) -> "Program":
        """``LDIO``: ``r = I/O register addr (+ idx)`` as a read now would return it, with
        no side effect, whatever the banking."""
        if idx is None and not peekable(addr):
            raise ValueError(f"${addr:04X} has no side-effect-free I/O view")
        return self._ld(LD_IO, r, addr, idx)

    def _alu(self, name, d, s) -> "Program":
        return self._emit("ALU", ALU.index(name), _reg(d), *_src(s))

    def mov(self, d: Reg, s: Src) -> "Program":
        """``MOV``/``LDI``: ``d = s``."""
        return self._alu("MOV", d, s)

    def add(self, d: Reg, s: Src) -> "Program":
        """``ADD``: ``d = d + s`` mod 2**32."""
        return self._alu("ADD", d, s)

    def sub(self, d: Reg, s: Src) -> "Program":
        """``SUB``: ``d = d - s`` mod 2**32."""
        return self._alu("SUB", d, s)

    def and_(self, d: Reg, s: Src) -> "Program":
        """``AND``: ``d = d & s``."""
        return self._alu("AND", d, s)

    def or_(self, d: Reg, s: Src) -> "Program":
        """``OR``: ``d = d | s``."""
        return self._alu("OR", d, s)

    def xor(self, d: Reg, s: Src) -> "Program":
        """``XOR``: ``d = d ^ s``."""
        return self._alu("XOR", d, s)

    def shl(self, d: Reg, s: Src) -> "Program":
        """``SHL``: ``d = d << (s & 31)`` mod 2**32."""
        return self._alu("SHL", d, s)

    def shr(self, d: Reg, s: Src) -> "Program":
        """``SHR``: ``d = d >> (s & 31)``."""
        return self._alu("SHR", d, s)

    def add8(self, d: Reg, s: Src) -> "Program":
        """``ADD8``: ``d = (d + s) & 0xFF``."""
        return self._alu("ADD8", d, s)

    def sub8(self, d: Reg, s: Src) -> "Program":
        """``SUB8``: ``d = (d - s) & 0xFF`` (with :meth:`bmi8`, the sign on a wrapping
        8-bit compass)."""
        return self._alu("SUB8", d, s)

    def cyc(self, lo: Reg, hi: Optional[Reg] = None) -> "Program":
        """``CYC``: ``lo`` = the cycle's low 32 bits, ``hi`` its high bits."""
        return self._emit("CYC", _reg(lo), _reg(lo if hi is None else hi))

    def frames(self, r: Reg) -> "Program":
        """``FRAMES``: ``r`` = the marker arrivals counted so far."""
        return self._emit("FRAMES", _reg(r))

    # --- flow ------------------------------------------------------------------

    def _br(self, cond, a, b, label) -> "Program":
        return self._emit("BR", CONDS.index(cond), _reg(a), *_src(b), _Label(label))

    def beq(self, a: Reg, b: Src, label: str) -> "Program":
        """``BEQ``: to ``label`` when ``a == b``."""
        return self._br("EQ", a, b, label)

    def bne(self, a: Reg, b: Src, label: str) -> "Program":
        """``BNE``: to ``label`` when ``a != b``."""
        return self._br("NE", a, b, label)

    def blt(self, a: Reg, b: Src, label: str) -> "Program":
        """``BLT``: to ``label`` when ``a < b`` (unsigned)."""
        return self._br("LT", a, b, label)

    def bge(self, a: Reg, b: Src, label: str) -> "Program":
        """``BGE``: to ``label`` when ``a >= b`` (unsigned)."""
        return self._br("GE", a, b, label)

    def bmi8(self, r: Reg, label: str) -> "Program":
        """``BMI8``: to ``label`` when bit 7 of ``r`` is set."""
        return self._emit("BIT7", _reg(r), 1, _Label(label))

    def bpl8(self, r: Reg, label: str) -> "Program":
        """``BPL8``: to ``label`` when bit 7 of ``r`` is clear."""
        return self._emit("BIT7", _reg(r), 0, _Label(label))

    def jmp(self, label: str) -> "Program":
        """``JMP``: to ``label``."""
        return self._emit("JMP", _Label(label))

    def djnz(self, r: Reg, label: str) -> "Program":
        """``DJNZ``: ``r -= 1`` mod 2**32, to ``label`` unless it is now 0."""
        return self._emit("DJNZ", _reg(r), _Label(label))

    def out(self, r: Reg) -> "Program":
        """``OUT``: append ``r`` to the outputs."""
        return self._emit("OUT", _reg(r))

    def outmem(self, addr: int, length: int) -> "Program":
        """``OUTMEM``: append ``length`` bytes from ``addr`` as :meth:`ld` reads them."""
        return self._emit("OUTMEM", _word(addr), _int(length, 0, 0xFFFF, "length"))

    def halt(self, code: Src = 0) -> "Program":
        """``HALT``: end with status ``halt`` and ``code``."""
        return self._emit("HALT", *_src(code))

    # --- assembly --------------------------------------------------------------

    @property
    def code(self) -> Tuple[tuple, ...]:
        """The ops as ``(name, *operands)`` with labels resolved to op indices."""
        if self._code is None:

            def resolve(v):
                if not isinstance(v, _Label):
                    return v
                if v.name not in self._labels:
                    raise ValueError(f"label {v.name!r} is not defined")
                return self._labels[v.name]

            self._code = tuple(tuple(map(resolve, op)) for op in self._ops)
        return self._code

    def native(self):
        """The native machine's validated form of :attr:`code`."""
        if self._native is None:
            from .machine import _core  # pylint: disable=import-outside-toplevel

            self._native = _core.Code(self.code, self.budget, self.marker, self.max_ops)
        return self._native

    def __len__(self) -> int:
        return len(self._ops)


class _Watch:
    """The run op in progress: its stop set (``None``: every pc) and arrivals left."""

    def __init__(self):
        self.target, self.left, self.frames = frozenset(), 0, 0


def _peek_io(machine, addr: int) -> Optional[int]:
    if 0xD800 <= addr < 0xDC00:
        return machine.colour[addr & 0x3FF]
    return machine.clone().io_read(addr) if peekable(addr) else None


def _peek(machine, addr: int, mode: int) -> Optional[int]:
    if mode == LD_RAM:
        return machine.ram[addr]
    if mode == LD_IO:
        return _peek_io(machine, addr)
    if addr < 2:
        return machine[addr]
    page = machine.pages[addr >> 12]
    return page[addr] if page is not None else _peek_io(machine, addr)


def _key(machine, row: int, col: int, down: int) -> None:
    keys = machine.cia1.keys
    keys[row] = keys[row] & ~(1 << col) | down << col


def run(machine, program: Program) -> Outcome:
    """Run ``program`` on the Python ``machine`` (``machine.run_program``)."""
    # pylint: disable=protected-access
    code = program.code
    if machine._running:
        raise RuntimeError("the machine is already running")
    watch, marker = _Watch(), program.marker
    decode = machine._decode

    def hook():
        pc = machine.mpu.pc
        watch.frames += pc == marker
        if watch.left and (watch.target is None or pc in watch.target):
            watch.left -= 1
            if not watch.left:
                machine.stop = machine._parked = True
        decode()

    own = machine.__dict__.get("_decode")
    machine._decode = hook
    try:
        status, value, outputs = _interpret(machine, program, code, watch)
    finally:
        if own is None:
            del machine._decode
        else:
            machine._decode = own
    return Outcome(status, value, machine.t, watch.frames, tuple(outputs))


def _interpret(machine, program: Program, code, watch: _Watch):
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    limit = machine.t + program.budget
    reg, out, pc, ops = [0] * len(R), [], 0, 0

    def val(kind, v):
        return reg[v] if kind else v

    while pc < len(code):
        if ops == program.max_ops:
            return "ops", pc, out
        ops += 1
        op, pc = code[pc], pc + 1
        name = op[0]
        if name == "KEYS":
            machine.cia1.keys[:] = bytes(op[1:])
        elif name == "KEY":
            _key(machine, *op[1:])
        elif name == "KEYR":
            k = reg[op[1]] & 63
            _key(machine, k >> 3, k & 7, op[2])
        elif name == "RUN":
            _, target, kind, v, here = op
            n = val(kind, v)
            parked = machine._parked and (  # pylint: disable=protected-access
                target is None or machine.mpu.pc in target
            )
            if n == 0 or here and parked:
                continue
            watch.target, watch.left = target, n
            machine.run(limit)
            if machine.jammed:
                return "jam", pc - 1, out
            if watch.left:
                return "budget", pc - 1, out
        elif name == "LD":
            _, r, addr, idx, mode = op
            a = addr + (reg[idx] if idx >= 0 else 0) & 0xFFFF
            v = _peek(machine, a, mode)
            if v is None:
                return "fault", a, out
            reg[r] = v
        elif name == "ALU":
            _, alu, d, kind, v = op
            x, y = reg[d], val(kind, v)
            reg[d] = (
                y,
                x + y,
                x - y,
                x & y,
                x | y,
                x ^ y,
                x << (y & 31),
                x >> (y & 31),
                x + y & 0xFF,
                x - y & 0xFF,
            )[alu] & _U32
        elif name == "CYC":
            reg[op[2]] = machine.t >> 32 & _U32
            reg[op[1]] = machine.t & _U32
        elif name == "FRAMES":
            reg[op[1]] = watch.frames
        elif name == "BR":
            _, cond, a, kind, v, to = op
            x, y = reg[a], val(kind, v)
            if (x == y, x != y, x < y, x >= y)[cond]:
                pc = to
        elif name == "BIT7":
            if bool(reg[op[1]] & 0x80) == bool(op[2]):
                pc = op[3]
        elif name == "JMP":
            pc = op[1]
        elif name == "DJNZ":
            reg[op[1]] = reg[op[1]] - 1 & _U32
            if reg[op[1]]:
                pc = op[2]
        elif name == "OUT":
            out.append(reg[op[1]])
        elif name == "OUTMEM":
            for i in range(op[2]):
                a = op[1] + i & 0xFFFF
                v = _peek(machine, a, LD_CPU)
                if v is None:
                    return "fault", a, out
                out.append(v)
        else:
            return "halt", val(*op[1:]), out
    return "halt", 0, out


def _forkable(base) -> None:
    from .machine import NativeSid  # pylint: disable=import-outside-toplevel
    from .sid import PyNativeSid  # pylint: disable=import-outside-toplevel

    if any(not isinstance(d, (NativeSid, PyNativeSid)) for d in base.sids.values()):
        raise ValueError("forks share a Python SID device: use a NativeSid")


def run_forks_python(base, programs, keep: bool = False):
    """The reference :func:`run_forks`: each program on a clone of ``base``, in turn."""
    _forkable(base)
    outs, forks = [], []
    for program in programs:
        fork = base.clone()
        outs.append(run(fork, program))
        if keep:
            forks.append(fork)
    return (outs, forks) if keep else outs


def run_forks(base, programs, threads: Optional[int] = None, keep: bool = False):
    """Each program on its own fork of the idle ``base``: outcomes in program order, with
    ``keep`` ``(outcomes, forks)``. Native forks run on ``threads`` threads (default: one
    per core), results independent of the count; SIDs must be ``NativeSid``s."""
    return base.run_forks(programs, threads=threads, keep=keep)


__all__ = [
    "ALU",
    "CONDS",
    "MAX_OPS",
    "Outcome",
    "Program",
    "R",
    "Reg",
    "STATUSES",
    "peekable",
    "run",
    "run_forks",
]
