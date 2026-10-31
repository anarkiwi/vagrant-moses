"""Run one random program on both machine backends and report the first divergence.

The program is byte soup over every non-JAM opcode with CIA timers, a raster IRQ,
sprite DMA and bank switching set up, a trap and a SID that logs. Both backends are
observed per instruction (cycle, pc, registers, port), plus their events, SID writes
and trap calls. Comparison ends at the first instruction whose fetch has a side effect (see
:func:`_risky`): the Python engine re-reads fetched bytes by its cache state.
"""

from __future__ import annotations

import random
from array import array
from typing import NamedTuple, Optional

from jennings.opcodes import OPCODES

from vagrant_moses import memmap as mm
from vagrant_moses import machine as vm
from vagrant_moses.machine import Machine, boot_cycle
from vagrant_moses.sid import PyNativeSid

from asm import assemble

SOUP, SOUP_LEN, HANDLER, RAM_HANDLER = 0x1000, 0x400, 0x0F00, 0x0F40
_LENGTH = {
    "impl": 1,
    "acc": 1,
    "imm": 2,
    "zp": 2,
    "zpx": 2,
    "zpy": 2,
    "indx": 2,
    "indy": 2,
    "rel": 2,
    "abs": 3,
    "absx": 3,
    "absy": 3,
    "ind": 3,
}
_OPS = [op for op in range(256) if OPCODES[op][0] != "JAM"]
SLED = bytes((0x4C, 0x00, SOUP >> 8))
_FLOW = ("BRK", "RTI", "RTS", "JMP")
_SID_WRITES = [0x8D, 0x8E, 0x8C]
_HI = [0xD0, 0xD4, 0xD8, 0xDC, 0xDC, 0xDD, 0xDD, 0xDE, 0x00, 0x01, 0xFF, 0x0F]
_ZP = [0x00, 0x01, 0x01, 0xFE, 0xFF, 0x02]


class Sid:
    """A SID stand-in logging each write and answering reads from the cycle."""

    def __init__(self):
        self.writes = []

    def read(self, cycle, register):
        return (cycle ^ register * 37) & 0xFF

    def write(self, cycle, register, value):
        self.writes.append((cycle, register, value))


class RefMachine(Machine):
    """The Python machine with irqhost's trap and a per-decode trace."""

    def __init__(self, *args, trap, trace, **kwargs):
        super().__init__(*args, **kwargs)
        self.trap, self.trace = trap, trace
        self.idle, self.armed, self.watch = None, False, frozenset()

    def __setitem__(self, addr, value):
        super().__setitem__(addr, value)
        if addr <= 1 and self.armed:
            self.watch = _ALL

    def _decode(self):
        self.trace(self)
        if not self.armed and self.mpu.pc == self.idle:
            self.armed, self.watch = True, _ALL
        while self.mpu.pc in self.watch and self.trap(self):
            pass
        super()._decode()


class _Anywhere:
    def __contains__(self, _addr):
        return True


_ALL = _Anywhere()


class Run(NamedTuple):
    """Everything observed from one run."""

    trace: array
    events: list
    sid: list
    traps: list
    t: int


def program(seed: int):
    """``(ntsc, cia_new, ram image writes, watch set)`` of a random program."""
    rng = random.Random(seed)
    soup, starts = bytearray(), []
    while len(soup) < SOUP_LEN:
        starts.append(SOUP + len(soup))
        op = rng.choice(_OPS)
        mn, mode = OPCODES[op]
        if mn in _FLOW and rng.random() < 0.97:
            continue
        if rng.random() < 0.06:
            op, mn, mode = rng.choice(_SID_WRITES), "STA", "abs"
            soup += bytes((op, rng.randrange(0x19), 0xD4))
            continue
        soup.append(op)
        n = _LENGTH[mode] - 1
        if mode == "rel":
            soup.append(rng.randrange(256) if rng.random() < 0.3 else rng.randrange(8))
        elif n == 1:
            soup.append(rng.choice(_ZP) if rng.random() < 0.3 else rng.randrange(256))
        elif n == 2:
            if mn in ("JMP", "JSR") and mode == "abs":
                soup += (rng.choice(starts) + rng.randrange(3)).to_bytes(2, "little")
                continue
            hi = rng.choice(_HI)
            lo = (
                rng.randrange(256)
                if rng.random() < 0.7
                else rng.choice(_ZP + [0x0D, 0x0E])
            )
            soup += bytes((lo, hi))
        if len(starts) % 12 == 0:
            soup += b"\x4c" + rng.choice(starts).to_bytes(2, "little")
    soup += b"\x4c" + SOUP.to_bytes(2, "little")
    latch_a, latch_b = rng.randrange(60, 4000), rng.randrange(60, 4000)
    setup = f"""
  SEI
  LDA #${HANDLER & 0xFF:02X}
  STA $0314
  STA $0318
  STA $0316
  LDA #${HANDLER >> 8:02X}
  STA $0315
  STA $0319
  STA $0317
  LDA #$7F
  STA $DC0D
  STA $DD0D
  LDA #${latch_a & 0xFF:02X}
  STA $DC04
  LDA #${latch_a >> 8:02X}
  STA $DC05
  LDA #${latch_b & 0xFF:02X}
  STA $DC06
  LDA #${latch_b >> 8:02X}
  STA $DC07
  LDA #${rng.choice([0x81, 0x83, 0x82]):02X}
  STA $DC0D
  LDA #${rng.choice([0x11, 0x01, 0x09, 0x15]):02X}
  STA $DC0E
  LDA #${rng.choice([0x11, 0x41, 0x01, 0x59]):02X}
  STA $DC0F
  LDA #$81
  STA $DD0D
  LDA #$21
  STA $DD04
  LDA #$13
  STA $DD05
  LDA #$01
  STA $DD0E
  LDA #${rng.randrange(256):02X}
  STA $D012
  LDA #${rng.choice([0x1B, 0x9B, 0x3B, 0x0B]):02X}
  STA $D011
  LDA #${rng.choice([1, 3, 0x0F]):02X}
  STA $D01A
  LDA #${rng.randrange(256):02X}
  STA $D015
  LDA #${rng.randrange(256):02X}
  STA $D017
  LDA #${rng.randrange(256):02X}
  STA $D001
  STA $D003
  STA $D005
  LDA #${rng.randrange(256):02X}
  STA $D007
  STA $D009
  STA $D00B
  STA $D00D
  STA $D00F
  CLI
  JMP ${SOUP:04X}
"""
    handler = """
  LDA $DC0D
  LDA $DD0D
  LDA #$0F
  STA $D019
  INC $D020
  PLA
  TAY
  PLA
  TAX
  PLA
  RTI
"""
    ram_handler = """
  PHA
  LDA $DC0D
  LDA $DD0D
  LDA #$0F
  STA $D019
  INC $D020
  PLA
  RTI
"""
    writes = []
    for org, src in ((0x0800, setup), (HANDLER, handler), (RAM_HANDLER, ram_handler)):
        # labels are numeric here, so assemble() needs none
        code, _ = assemble(src, org)
        writes.append((org, bytes(code)))
    writes.append((SOUP, bytes(soup)))
    writes.append((0xFFFA, bytes((RAM_HANDLER & 0xFF, RAM_HANDLER >> 8)) * 3))
    watch = frozenset(rng.sample(starts, 6))
    return bool(rng.getrandbits(1)), bool(rng.getrandbits(1)), writes, watch


def _snapshot(trace, machine):
    mpu = machine.mpu
    trace.extend(
        (
            machine.t,
            mpu.pc,
            mpu.a,
            mpu.x,
            mpu.y,
            mpu.sp,
            mpu.p,
            machine.port_data,
            mpu.processorCycles,
        )
    )


def run(seed: int, cycles: int, backend: str, logged: bool = False) -> Run:
    """Run program ``seed`` for ``cycles`` cycles on ``backend``.

    ``logged`` puts a NativeSid in front of the chip: the pure-Python one on the Python
    machine, the native one on the native machine.
    """
    ntsc, cia_new, writes, watch = program(seed)
    trace, traps, sid = array("q"), [], Sid()
    dev = sid
    if logged:
        dev = (PyNativeSid if backend == "python" else vm.NativeSid)("6581", sid.read)
    calls = {}

    def trap(machine):
        mpu = machine.mpu
        pc = mpu.pc
        n = calls[pc] = calls.get(pc, 0) + 1
        traps.append((machine.t, pc, mpu.a, n))
        if pc not in watch:
            machine.watch = watch
            return False
        if n % 3 == 0:
            mpu.a = (mpu.a + n) & 0xFF
            mpu.pc = (pc + 1) & 0xFFFF
            return True
        if n % 3 == 1:
            machine.ram[0x1500 + n % 16] = n & 0xFF
            machine[0xD020 + n % 8] = n & 0xFF
            machine[0x01 if n % 5 == 1 else 0x1501] = mpu.a & 0x07 | 0x30
        else:
            machine.watch = watch
        return False

    def port_write(machine):
        machine.watch = machine.WATCH_ALL

    if backend == "python":
        machine = RefMachine(
            ntsc,
            {mm.SID_BASE: dev},
            cia_new,
            backend="python",
            trap=trap,
            trace=lambda m: _snapshot(trace, m),
        )
        machine.idle = SOUP
        machine.WATCH_ALL = _ALL
    else:
        machine = Machine(ntsc, {mm.SID_BASE: dev}, cia_new, backend="native")
        machine.set_trap(trap, frozenset())
        machine.arm_at(SOUP)
        machine.on_port_write(port_write)
        machine.set_decode_hook(lambda m: _snapshot(trace, m))
    rng = random.Random(~seed)
    machine.ram[2:0x200] = bytes(rng.choice(_OPS) for _ in range(0x1FE))
    for lo, hi in ((0x0200, 0xD000), (0xE000, 0x10000)):
        machine.ram[lo:hi] = (SLED * ((hi - lo) // 3 + 1))[: hi - lo]
    for org, data in writes:
        machine.ram[org : org + len(data)] = data
    start = machine.reset(0x0800, boot_cycle(ntsc))
    try:
        machine.run(start + cycles)
    except RuntimeError:  # the Python engine's cache-keyed re-reads met an open bus
        pass
    sid_writes = dev.drain_writes() if logged else sid.writes
    return Run(trace, list(machine.events), sid_writes, traps, machine.t)


def _cut(run_: Run, t_cut: int) -> Run:
    """``run_`` up to cycle ``t_cut`` (the first risky fetch)."""
    rows = len(run_.trace) // 9
    n = next((i for i in range(rows) if run_.trace[9 * i] >= t_cut), rows)
    return Run(
        run_.trace[: 9 * n],
        [e for e in run_.events if e[0] < t_cut],
        [w for w in run_.sid if w[0] < t_cut],
        [c for c in run_.traps if c[0] < t_cut],
        run_.t,
    )


def _risky(pc: int) -> bool:
    """The instruction at ``pc`` has a byte fetched from the I/O page.

    The Python engine reads an instruction's bytes several times, by its cache state:
    the open bus advances, a CIA ICR read clears, and the last value read, which an
    RMW instruction writes back first, is the last byte re-read. The native machine
    reads each byte once; the two agree on code in RAM and ROM.
    """
    return any(0xD000 <= a < 0xE000 for a in range(pc, pc + 3))


def _io_fetch(run_: Run) -> int:
    """The cycle of the first risky fetch, else a huge cycle."""
    pcs = run_.trace[1::9]
    at = next((i for i, pc in enumerate(pcs) if _risky(pc)), None)
    return 1 << 62 if at is None else run_.trace[9 * at]


def first_divergence(a: Run, b: Run) -> Optional[str]:
    """Why the runs differ before either makes a risky fetch, or ``None``."""
    t_cut = min(_io_fetch(a), _io_fetch(b))
    a, b = _cut(a, t_cut), _cut(b, t_cut)
    for name, xs, ys in (
        ("events", a.events, b.events),
        ("sid", a.sid, b.sid),
        ("traps", a.traps, b.traps),
    ):
        if xs != ys:
            at = next((i for i, (x, y) in enumerate(zip(xs, ys)) if x != y), None)
            return f"{name} differ at {at}: {len(xs)} vs {len(ys)}"
    if a.trace != b.trace:
        rows = min(len(a.trace), len(b.trace)) // 9
        for i in range(rows):
            x, y = a.trace[9 * i : 9 * i + 9], b.trace[9 * i : 9 * i + 9]
            if x != y:
                return f"trace differs at instruction {i}: {list(x)} vs {list(y)}"
        return f"trace lengths {len(a.trace) // 9} vs {len(b.trace) // 9}"
    return None


def compare(seed: int, cycles: int, logged: bool = False) -> Optional[str]:
    """Run ``seed`` on both backends; ``None`` if they agree."""
    return first_divergence(
        run(seed, cycles, "python", logged), run(seed, cycles, "native", logged)
    )
