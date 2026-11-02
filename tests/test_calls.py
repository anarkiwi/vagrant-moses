"""The native call queue against the same calls driven from a Python trap."""

from __future__ import annotations

import struct

import pytest

from vagrant_moses import memmap as mm
from vagrant_moses import smc
from vagrant_moses.machine import Machine, NativeSid

from asm import assemble

pytest.importorskip("jennings")
pytestmark = pytest.mark.skipif(not smc.CALLS, reason="no native call queue")

ORG = 0x1000
WINDOW = (0xD000, 0xE000)
SRC = """
init:  LDA #$35
       STA $01
       LDA #$00
       STA $80
       RTS
play:  LDX $80
       LDA $D41B
       STA $D400,X
       STA $0150,X
       PHA
       PLA
       INC $80
       JSR sub
       RTS
sub:   LDA $A000
       STA $81
       RTS
spin:  JMP spin
idle:  JMP idle
"""
TABLE = bytearray(512)
for _op, _e in {0xAD: 4 | 1 << 4, 0x8D: 4 | 1 << 5 | 1 << 6, 0x9D: 5 | 1 << 5 | 1 << 6,
                0x48: 1 << 9, 0xE6: 1 | 3 << 4, 0x20: 4 | 1 << 5 | 2 << 9,
                0xA6: 1 | 1 << 4, 0x85: 1 | 1 << 5 | 1 << 6}.items():  # fmt: skip
    TABLE[2 * _op : 2 * _op + 2] = _e.to_bytes(2, "little")


def _machine(facts=True):
    m = Machine(
        backend="native", sids={mm.SID_BASE: NativeSid("6581", lambda c, r: 0x40 | r)}
    )
    code, labels = assemble(SRC, ORG)
    m.ram[ORG : ORG + len(code)] = code
    rec = smc.Recorder(m, bytes(m.ram))
    tr = smc.Tracer(rec, bytes(TABLE), labels["idle"], WINDOW, facts)
    return m, rec, tr, labels


def _python_run(queue, ports, call_cycles, facts=True):
    """The queue driven from a Python trap, as ``vagrant_moses.smc.Calls`` does natively."""
    m, rec, tr, labels = _machine(facts)
    queue, out, cur, entries = list(queue), [], [None], (set(), set())

    def snap(halt):
        if cur[0] is not None:
            log, idx, vals = tr.snap()
            r = m.mpu
            out.append(((cur[0], halt, r.a, r.x, r.y, r.sp, r.p), log, idx, vals))
            tr.phase(tr.NONE)
            cur[0] = None

    def trap(mc):
        snap(0)
        if not queue:
            mc.stop = True
            return False
        phase, pc, a = queue.pop(0)
        cur[0], sp0 = phase, (mc.mpu.sp - 2) & 0xFF
        tr.phase(phase, sp0 if phase == tr.CALL else 0)
        port = ports.get("init" if phase == tr.INIT else "call")
        if port is not None and port != mc.port_data:
            rec.host_write(1, port)
        if phase == tr.CALL:
            rec.begin() if rec.call < 0 else rec.next_call()  # pylint: disable=W0106
        sp, ret = mc.mpu.sp, labels["idle"] - 1
        rec.host_write(0x100 | sp, ret >> 8)
        rec.host_write(0x100 | (sp - 1) & 0xFF, ret & 0xFF)
        mc.mpu.sp, mc.mpu.a, mc.mpu.pc = sp0, a, pc
        rec.deadline = mc.cycle + call_cycles if call_cycles else None
        if phase == tr.CALL:
            entries[0].add((pc, sp0))
            entries[1].add((pc, mc.port_dir, mc.port_data))
        return True

    m.set_trap(trap, {labels["idle"]})
    m.run(m.reset(labels["idle"]) + 100_000)
    rec.finish()
    snap(2 if m.jammed else 1)
    return out, tuple(map(sorted, entries)), tr.facts(), rec.events(), len(queue)


def _native_run(queue, ports, call_cycles, facts=True):
    m, rec, tr, labels = _machine(facts)
    calls = smc.Calls(rec, queue, labels["idle"], ports, call_cycles)
    m.run(m.reset(labels["idle"]) + 100_000)
    rec.finish()
    calls.close()
    r = calls.records()
    meta = [tuple(r.meta[k : k + 7]) for k in range(0, len(r.meta), 7)]
    ends = [struct.unpack_from("<Q", e, k)[0] for e in (r.log_end, r.delta_end)
            for k in range(0, len(e), 8)]  # fmt: skip
    log_end, delta_end = ends[: len(meta)], ends[len(meta) :]
    out, lo, do = [], 0, 0
    for k, mt in enumerate(meta):
        log = [(r.log[i] | r.log[i + 1] << 8, r.log[i + 2], bool(r.log[i + 3]))
               for i in range(4 * lo, 4 * log_end[k], 4)]  # fmt: skip
        out.append(
            (mt, log, r.idx[2 * do : 2 * delta_end[k]], r.vals[do : delta_end[k]])
        )
        lo, do = log_end[k], delta_end[k]
    assert r.origin[ORG] == 0xA9
    return out, calls.entries(), tr.facts(), rec.events(), calls.left


def _labels():
    return assemble(SRC, ORG)[1]


@pytest.mark.parametrize("facts", [True, False])
@pytest.mark.parametrize(
    "calls,ports,call_cycles",
    [
        ([("init", 0), ("play", 1), ("play", 2), ("play", 0)], {}, 0),
        ([("init", 0), ("play", 0), ("play", 0)], {"init": 0x37, "call": 0x36}, 5000),
        ([("play", 0), ("spin", 0), ("play", 0)], {"call": 0x37}, 500),
    ],
)
def test_the_queue_runs_as_a_python_trap_would(calls, ports, call_cycles, facts):
    labels = _labels()
    queue = [(1 if n == "init" else 2, labels[n], a) for n, a in calls]
    py = _python_run(queue, ports, call_cycles, facts)
    nat = _native_run(queue, ports, call_cycles, facts)
    assert py == nat
    assert bool(nat[2].starts) == facts
    if "spin" in dict(calls):
        assert nat[0][-1][0][1] == 1 and nat[4] == 1


def test_a_packed_queue_and_bad_queues():
    m, rec, _, labels = _machine()
    pc = labels["play"]
    smc.Calls(rec, bytes([2, pc & 0xFF, pc >> 8, 0]), labels["idle"])
    assert m.smc_calls_left == 1
    with pytest.raises(ValueError):
        smc.Calls(rec, b"\x02\x00\x10", labels["idle"])
    with pytest.raises(ValueError):
        smc.Calls(rec, bytes(4), labels["idle"])
    with pytest.raises(RuntimeError):
        Machine(backend="native").smc_calls_left  # pylint: disable=W0106


def test_a_queue_without_a_trace_keeps_no_records():
    m = Machine(backend="native")
    code, labels = assemble(SRC, ORG)
    m.ram[ORG : ORG + len(code)] = code
    rec = smc.Recorder(m, bytes(m.ram))
    calls = smc.Calls(rec, [(2, labels["play"], 0)] * 3, labels["idle"])
    m.run(m.reset(labels["idle"]) + 100_000)
    calls.close()
    assert calls.left == 0 and rec.call == 2 and not any(calls.records()[1:])
    assert calls.entries()[0] == [(labels["play"], 0xFD)]
