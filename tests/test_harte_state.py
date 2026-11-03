"""Registers, every flag and RAM after an instruction against Tom Harte's SingleStepTests 6502.

Both machines run with the CPU port all RAM. ADC, SBC, ARR, RRA, ISC and their aliases run
every case, decimal-mode ones included.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jennings")

import harte  # pylint: disable=wrong-import-position
from vagrant_moses import machine as vm  # pylint: disable=wrong-import-position

START = 1000
FLAGS = 0xCF
# ANE and LXA take a chip-dependent constant.
MAGIC = frozenset((0x8B, 0xAB))
BACKENDS = ["python"] + (["native"] if vm.NATIVE else [])


def _cached(op):
    if not (harte.CACHE / harte.SHA[:12] / f"{op:02x}.json.gz").exists():
        pytest.skip("no cached cases: run tools/harte_cycles.py")
    return harte.cases(op, harte.want(op))


def _on_ports(case):
    """A case that touches the CPU port's registers, which are not RAM on a C64."""
    addrs = [a for a, _ in case["ram"]] + [a for a, _ in case["cycles"]]
    return any(a < 2 for a in addrs)


def run(backend, case):
    """The ``(pc, s, a, x, y, p)`` and RAM the machine ends with, from ``case``'s start,
    one cycle into the next opcode's fetch (``BRK`` takes its vector then)."""
    pc, sp, a, x, y, p = case["regs"]
    machine = vm.Machine(False, backend=backend)
    machine.reset(pc, START)
    machine[0], machine[1] = 0x2F, 0x30
    for addr, value in case["ram"]:
        machine.ram[addr] = value
    mpu = machine.mpu
    mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p = sp, a, x, y, p
    machine.run(START + len(case["cycles"]) + 1)
    regs = [mpu.pc, mpu.sp, mpu.a, mpu.x, mpu.y, mpu.p & FLAGS]
    return regs, {addr: machine.ram[addr] for addr, _ in case["final_ram"]}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("op", harte.OPS, ids=lambda op: f"{op:02x}")
def test_state_matches_single_step_tests(backend, op):
    """Every case's registers, flags and RAM, bar the chip-dependent opcodes."""
    if op in MAGIC:
        pytest.skip("magic constant is chip-dependent")
    for case in _cached(op):
        if _on_ports(case):
            continue
        regs, ram = run(backend, case)
        want = case["final"][:5] + [case["final"][5] & FLAGS]
        assert regs == want, case
        assert ram == dict(case["final_ram"]), case


def test_decimal_cases_are_present_and_exercise_the_flag():
    """The decimal opcodes are checked in full, and about half their cases set D."""
    cases = _cached(0x69)
    assert len(cases) == 10_000
    assert 4000 < sum(c["regs"][5] & 8 != 0 for c in cases) < 6000
