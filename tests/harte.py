"""Tom Harte's SingleStepTests 6502 cases against the Python machine's bus cycles.

Cases (MIT, ``SingleStepTests/65x02`` at a pinned commit) are cached trimmed under
``.harte-cache``, never committed. :func:`check` runs one on a machine whose every access
is I/O; ``docs/c64.md`` (Bus cycles) states what a case must match.
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

from jennings.opcodes import OPCODES

from vagrant_moses.machine import Machine

SHA = "2f6980a2d95757486c7bee24355c360e40e2a224"
URL = "https://raw.githubusercontent.com/SingleStepTests/65x02/{sha}/6502/v1/{op:02x}.json"
CACHE = Path(__file__).resolve().parent.parent / ".harte-cache"
OPS = [op for op in range(256) if OPCODES[op][0] != "JAM"]
START = 1000
UNSTABLE = frozenset((0x93, 0x9B, 0x9C, 0x9E, 0x9F))


def _trim(case: dict) -> dict:
    ini = case["initial"]
    return {
        "regs": [ini[k] for k in ("pc", "s", "a", "x", "y", "p")],
        "ram": ini["ram"],
        "final_pc": case["final"]["pc"],
        "cycles": [[addr, kind[0]] for addr, _, kind in case["cycles"]],
    }


def cases(op: int, count: int, cache: Path = CACHE) -> List[dict]:
    """The first ``count`` cases of ``op``, fetched once and cached trimmed."""
    path = cache / SHA[:12] / f"{op:02x}-{count}.json.gz"
    if not path.exists():
        with urllib.request.urlopen(URL.format(sha=SHA, op=op), timeout=120) as resp:
            data = json.load(resp)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as out:
            json.dump([_trim(c) for c in data[:count]], out)
        tmp.replace(path)
    with gzip.open(path, "rt") as src:
        return json.load(src)


class Probe(Machine):
    """The Python machine on a flat 64 KiB space that answers every access as I/O."""

    def __init__(self):
        super().__init__(backend="python", on_io=lambda *_: None)
        self.flat = bytearray(0x10000)
        self.pages = [None] * 16
        self.log: List[Tuple[int, int, str, bool]] = []
        self.decodes: List[Tuple[int, int]] = []
        self.stepping = False

    def _cycle(self) -> int:
        return self.sched.now >> 1

    def __getitem__(self, addr: int) -> int:
        if self._obs:
            self.log.append((self._cycle(), addr, "r", self.stepping))
        self.last_read = self.flat[addr]
        return self.last_read

    def __setitem__(self, addr: int, value: int) -> None:
        self.pending.append((addr, value))

    def io_write(self, addr: int, value: int) -> None:
        self.log.append((self._cycle(), addr, "w", False))
        self.flat[addr] = value

    def _push(self, value: int) -> None:
        self.io_write(0x100 | self.mpu.sp, value)
        self.mpu.sp = (self.mpu.sp - 1) & 0xFF

    def _exec(self) -> None:
        self.stepping = True
        super()._exec()
        self.stepping = False

    def _decode(self) -> None:
        self.decodes.append((self._cycle(), self.mpu.pc))
        super()._decode()


def unstable(op: int, case: dict) -> bool:
    """A page-crossing SHA/SHX/SHY/TAS case, which jennings writes unANDed."""
    if op not in UNSTABLE:
        return False
    pc, _, _, x, y, _ = case["regs"]
    ram = dict(case["ram"])
    mode = OPCODES[op][1]
    operand = ram[(pc + 1) & 0xFFFF]
    lo = ram[operand] if mode == "indy" else operand
    return lo + (x if mode == "absx" else y) > 0xFF


def check(case: dict) -> List[str]:
    """The ways the machine's bus cycles for ``case`` differ from it (empty: none)."""
    pc, sp, a, x, y, p = case["regs"]
    probe = Probe()
    for addr, value in case["ram"]:
        probe.flat[addr] = value
    probe.reset(pc, START)
    probe.pending.clear()
    mpu = probe.mpu
    mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p = sp, a, x, y, p
    want = [tuple(c) for c in case["cycles"]]
    n = len(want)
    probe.run(START + n + 1)
    errors = []
    if probe.decodes[:2] != [(START, pc), (START + n, case["final_pc"])]:
        errors.append(
            f"decodes {probe.decodes}: want {n} cycles to {case['final_pc']:04x}"
        )
    log = [(c - START, addr, rw, lumped) for c, addr, rw, lumped in probe.log]
    log = [entry for entry in log if entry[0] < n]
    timed: Dict[int, Tuple[int, str]] = {}
    lumped_reads = set()
    for i, addr, rw, lumped in log:
        if lumped and rw == "r":
            lumped_reads.add(addr)
            if (addr, "r") not in want and (addr - pc) & 0xFFFF >= 3:
                errors.append(f"execution reads {addr:04x}, which no cycle does")
            continue
        if i in timed:
            errors.append(f"cycle {i + 1}: two accesses")
        timed[i] = (addr, rw)
        if want[i] != (addr, rw):
            errors.append(
                f"cycle {i + 1}: {rw} {addr:04x}, want {want[i][1]} {want[i][0]:04x}"
            )
    for i, (addr, rw) in enumerate(want[1:], 1):
        if i in timed or rw == "r" and (addr < 0x200 or addr in lumped_reads):
            continue
        errors.append(f"cycle {i + 1}: {rw} {addr:04x} not made")
    return errors


__all__ = ["CACHE", "OPS", "SHA", "Probe", "cases", "check", "unstable"]
