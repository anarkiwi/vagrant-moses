"""The machine's bus cycles against Tom Harte's SingleStepTests 6502 cases."""

from __future__ import annotations

import pytest

pytest.importorskip("jennings")

import harte  # pylint: disable=wrong-import-position
from vagrant_moses import machine as vm  # pylint: disable=wrong-import-position

CASES = 100


def _cached(op):
    if not (harte.CACHE / harte.SHA[:12] / f"{op:02x}-{CASES}.json.gz").exists():
        pytest.skip("no cached cases: run tools/harte_cycles.py")
    return harte.cases(op, CASES)


@pytest.mark.parametrize("op", harte.OPS, ids=lambda op: f"{op:02x}")
def test_bus_cycles_match_single_step_tests(op):
    """Every case's cycles, bar the page-crossing writes of jennings' SH* opcodes."""
    for case in _cached(op):
        if not harte.unstable(op, case):
            assert harte.check(case) == [], case


def test_a_misplaced_cycle_is_reported():
    """``check`` flags a throw-away read the machine makes on another cycle or address."""
    case = {
        "regs": [0x2000, 0xFD, 0, 0, 0, 0],
        "ram": [[0x2000, 0xAA], [0x2001, 0xEA]],
        "final_pc": 0x2001,
        "cycles": [[0x2000, "r"], [0x2001, "r"]],
    }
    assert not harte.check(case)
    case["cycles"][1] = [0x3001, "r"]
    assert harte.check(case)
    case["cycles"].append([0x2001, "r"])
    assert harte.check(case)


def test_an_interrupt_reads_pc_twice_then_pushes_and_takes_the_vector():
    """64doc's IRQ sequence: two reads of PC, three pushes, the vector's two bytes."""
    probe = harte.Probe()
    probe.flat[0x2000] = 0xEA
    probe.reset(0x2000, harte.START)
    probe.pending.clear()
    probe.run(harte.START + 1)
    probe.mpu.sp, probe.mpu.p = 0xFD, 0
    probe.nmi, probe.ic = True, -vm._MAX  # pylint: disable=protected-access
    probe.run(harte.START + 12)
    entry = harte.START + 2
    log = [(c - entry, a, rw) for c, a, rw, _ in probe.log if c >= entry]
    assert log[:6] == [
        (1, 0x2001, "r"),
        (2, 0x01FD, "w"),
        (3, 0x01FC, "w"),
        (4, 0x01FB, "w"),
        (5, 0xFFFA, "r"),
        (6, 0xFFFB, "r"),
    ]
