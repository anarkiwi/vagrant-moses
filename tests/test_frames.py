"""Native interrupt frames against an independent model of the same program's decode trace."""

from __future__ import annotations

import numpy as np
import pytest

from vagrant_moses import smc
from vagrant_moses.machine import Machine

from asm import assemble

pytest.importorskip("jennings")
pytestmark = pytest.mark.skipif(not smc.FRAMES, reason="no native frames")

ORG = 0x1000
WINDOW = (0xD000, 0xE000)
EVENT = np.dtype(smc.EVENT_FIELDS)
ENTRY = np.dtype(smc.ENTRY_FIELDS)
CIA_IRQ, NMI = 2, 0
SETUP = """
main:  SEI
       LDA #$00
       STA $80
       STA $81
       STA $82
       LDA #$35
       STA $01
       LDA #$7F
       STA $DC0D
       STA $DD0D
       LDA #$81
       STA $DC0D
       STA $DD0D
       LDA #$0C
       STA $DC04
       STA $DD04
       LDA #$00
       STA $DC05
       STA $DD05
"""


def nest(first, second, nmi):
    """An IRQ whose handler starts an NMI, whose handler starts an IRQ, whose handler starts
    an NMI: each runs `first`, `second` or `nmi` NOPs after starting the next."""
    pad = lambda n: "       NOP\n" * n  # noqa: E731
    return (
        SETUP
        + """
       LDA #$19
       STA $DC0E
       CLI
       LDA #$11
       LDX #$22
       LDY #$33
spin:  JMP spin
irq:   LDA $DC0D
       INC $81
       LDA $81
       CMP #$01
       BNE irq2
       LDA #$19
       STA $DD0E
"""
        + pad(first)
        + """       RTI
irq2:  LDA #$19
       STA $DD0E
"""
        + pad(second)
        + """       RTI
nmi:   LDA $DD0D
       INC $82
       LDA $82
       CMP #$01
       BNE out
       LDA #$19
       STA $DC0E
       CLI
"""
        + pad(nmi)
        + """out:   RTI
idle:  JMP idle
"""
    )


NEST = nest(8, 4, 8)
DEEP = nest(24, 24, 24)
# A continuous timer: a frame every 200 cycles; the main code works in RAM when `work`.
PERIODIC = SETUP + """
       LDA #$C8
       STA $DC04
       LDA #$00
       STA $DC05
       LDA #$11
       STA $DC0E
       CLI
spin:  JMP spin
irq:   LDA $DC0D
       INC $80
       LDA $80
       STA $D400
       RTI
idle:  JMP idle
"""
WORK = PERIODIC.replace("spin:  JMP spin", "spin:  INC $0200\n       JMP spin")
LOOP = SETUP + """
       LDA #$C8
       STA $DC04
       LDA #$00
       STA $DC05
       LDA #$11
       STA $DC0E
       CLI
spin:  JMP spin
irq:   LDA $DC0D
       LDX #$03
l:     DEX
       BNE l
       RTI
idle:  JMP idle
"""
BRK_AT = PERIODIC.replace(
    "       LDA #$C8\n", "       BRK\n       NOP\n       LDA #$C8\n", 1
).replace(
    "irq:   LDA $DC0D",
    "irq:   LDA $DC0D\n       LDA $80\n       CMP #$02\n       BNE go\n       BRK\n       NOP\ngo:    NOP",
)
JAM = PERIODIC.replace("STA $D400", ".byte $02")
# A play routine that sits in a CLI window long enough for the continuous timer's IRQs.
CALLS = (
    SETUP.replace("main:  SEI", "init:  SEI")
    + """
       LDA #$28
       STA $DC04
       LDA #$11
       STA $DC0E
       RTS
play:  CLI
"""
    + "       NOP\n" * 50
    + """       SEI
       RTS
irq:   LDA $DC0D
       INC $80
       RTI
idle:  JMP idle
"""
)
BRANCHES = {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0}


def _machine(src, frames=True, budget=None):
    m = Machine(backend="native")
    code, labels = assemble(src, ORG)
    m.ram[ORG : ORG + len(code)] = code
    for vec, name in ((0xFFFE, "irq"), (0xFFFA, "nmi")):
        if name in labels:
            m.ram[vec], m.ram[vec + 1] = labels[name] & 0xFF, labels[name] >> 8
    rec = smc.Recorder(m, bytes(m.ram))
    smc.Tracer(rec, bytes(512), labels["idle"], WINDOW)
    fr = smc.Frames(rec, labels["idle"], budget) if frames else None
    return m, rec, fr, labels


def _trace(m, labels, cycles):
    """The decode trace of the plain machine: no interrupt is redirected."""
    out = []

    def hook(mc):
        sp = mc.mpu.sp
        out.append(
            dict(pc=mc.mpu.pc, t=mc.cycle, a=mc.mpu.a, x=mc.mpu.x, y=mc.mpu.y, sp=sp,
                 p=mc.ram[0x100 | (sp + 1) & 0xFF], op=mc.ram[mc.mpu.pc])  # fmt: skip
        )

    m.set_decode_hook(hook)
    m.run(m.reset(labels["main"]) + cycles)
    m.set_decode_hook(None)
    return out


def _io(d, ram):
    """The I/O accesses of the instruction of decode `d`: one for an absolute load or store."""
    pc = d["pc"]
    return int(d["op"] in (0xAD, 0x8D) and ram[pc + 2] >= 0xD0 and ram[pc + 2] < 0xE0)


def model(trace, ram, handlers, sentinel=None):
    """Frames (or, with `sentinel`, calls) from a decode trace by the contexts the handler
    entries and RTIs open and close: `[(row, events)]`."""
    units, stack, pending, prev, cur = [], [], [], None, None
    for i, d in enumerate(trace):
        if sentinel is not None and d["pc"] == sentinel and len(stack) <= 1:
            units.append(cur) if cur else None
            stack, cur = [dict(n=1, brk=False)], dict(pos=0, events=[], insns=1)
            prev = d
            continue
        entry = d["pc"] in handlers and not (prev and prev["op"] == 0)
        if not stack and sentinel is None:
            if not entry:
                for u in pending:
                    u["row"]["pc"] = d["pc"]
                pending = []
        if entry:
            kind = handlers[d["pc"]]
            regs = (d["a"], d["x"], d["y"], (d["sp"] + 3) & 0xFF, d["p"])
            live = [c for c in stack if not c["brk"]]
            if live:
                cur["events"].append(
                    dict(icount=live[-1]["n"], pos=cur["pos"], at=d["pc"], regs=regs,
                         depth=len(live) - 1, kind=kind)  # fmt: skip
                )
            else:
                cur = dict(
                    row=dict(at=d["pc"], pc=None, regs=regs, kind=kind, t0=d["t"]),
                    pos=0, events=[], insns=0,
                )  # fmt: skip
                units.append(cur)
                pending.append(cur)
            stack.append(dict(n=0, brk=False))
        elif prev and prev["op"] == 0 and stack:
            stack.append(dict(n=0, brk=True))
        live = [c for c in stack if not c["brk"]]
        if live:
            live[-1]["n"] += 1
            cur["insns"] += 1
            cur["pos"] += _io(d, ram)
        if d["op"] == 0x40 and stack:
            stack.pop()
            if not stack and sentinel is None:
                cur["row"]["cycles"] = trace[i + 1]["t"] - cur["row"]["t0"]
        prev = d
    return units


def _run(src, cycles, window=0, budget=None):
    """`(rows, events, records, fr)` of the native frames over `cycles`, taken per `window`."""
    m, rec, fr, labels = _machine(src, budget=budget)
    end = m.reset(labels["main"]) + cycles
    chunks = []
    while m.cycle < end and not m.stop and not m.jammed:
        m.run(min(end, m.cycle + window) if window else end)
        chunks.append(fr.take())
    rec.finish()
    fr.close()
    chunks.append(fr.take())
    rows, events, nrec = [], [], 0
    for c in chunks:
        rows.append(np.frombuffer(c.entries, ENTRY))
        ev = np.frombuffer(c.events, EVENT).copy()
        ev["record"] += nrec
        events.append(ev)
        nrec += len(c.entries) // ENTRY.itemsize
    return np.concatenate(rows), np.concatenate(events), chunks, fr, labels


def _expect(src, cycles, handlers):
    m, _, _, labels = _machine(src, frames=False)
    trace = _trace(m, labels, cycles)
    return model(trace, m.ram, {labels[n]: k for n, k in handlers.items()}), trace


def _same(rows, events, units, whole=True):
    """The native rows and events against the model's units."""
    if not whole:
        units = units[: len(rows)]
    assert len(rows) == len(units)
    for k, (r, u) in enumerate(zip(rows, units)):
        want = u["row"]
        got = (r["at"], r["pc"], *(r[f] for f in ("a", "x", "y", "sp", "p")), r["kind"])
        assert got == (want["at"], want["pc"], *want["regs"], want["kind"]), k
        assert (r["insns"], r["cycles"]) == (u["insns"], want["cycles"]), k
        ev = events[events["record"] == k]
        assert len(ev) == len(u["events"]), k
        for e, w in zip(ev, u["events"]):
            assert (e["icount"], e["pos"], e["at"], e["depth"], e["kind"]) == (
                w["icount"], w["pos"], w["at"], w["depth"], w["kind"]
            )  # fmt: skip
            assert tuple(e[f] for f in ("a", "x", "y", "sp", "p")) == w["regs"]


HANDLERS = {"irq": CIA_IRQ, "nmi": NMI}


def test_nested_interrupts_are_taken_at_the_instruction_boundaries_of_the_plain_run():
    units, _ = _expect(NEST, 2500, HANDLERS)
    rows, events, _, fr, _ = _run(NEST, 2500)
    assert len(units) == 3 and sum(len(u["events"]) for u in units) == 1
    # the NMI is taken at the boundary after the IRQ's RTI, and the second at the NMI's
    assert [r["pc"] for r in rows] == [units[0]["row"]["pc"]] * 3
    assert units[0]["events"] == [] and units[1]["events"][0]["icount"] == 16
    _same(rows, events, units, whole=False)
    assert (
        fr.outcome == 0 and fr.foreign == 0 and fr.closed == len(rows) and not fr.open
    )


def test_depth_two_interrupts_are_events_of_the_outermost_frame():
    units, _ = _expect(DEEP, 2500, HANDLERS)
    rows, events, _, _, _ = _run(DEEP, 2500)
    assert sorted(events["depth"]) == [0, 1, 2] and list(events["kind"]) == [0, 2, 0]
    _same(rows, events, units, whole=False)


def test_periodic_frames_taken_in_windows_match_the_plain_run():
    units, _ = _expect(PERIODIC, 20_000, {"irq": CIA_IRQ})
    whole = _run(PERIODIC, 20_000)
    windowed = _run(PERIODIC, 20_000, window=999)
    assert len(units) > 20 and len(whole[0]) >= len(units) - 1
    _same(whole[0], whole[1], units, whole=False)
    assert np.array_equal(windowed[0], whole[0])
    assert not windowed[3].dirty and windowed[3].foreign == 0
    assert windowed[3].image[ORG : ORG + 4] == bytes(whole[3].image[ORG : ORG + 4])
    assert len(windowed[2]) > 3 and not any(
        c.records.origin == b"" for c in windowed[2][:-1] if c.entries
    )


def test_records_replay_to_the_machines_ram():
    rows, _, chunks, fr, _ = _run(PERIODIC, 20_000, window=3000)
    ram = None
    for c in chunks:
        r = c.records
        if not c.entries:
            continue
        ram = bytearray(r.origin)
        lo = 0
        for end in np.frombuffer(r.delta_end, "<u8"):
            idx = np.frombuffer(r.idx, "<u2")[lo:end]
            vals = np.frombuffer(r.vals, np.uint8)[lo:end]
            ram_arr = np.frombuffer(ram, np.uint8).copy()
            ram_arr[idx] = vals
            ram = bytearray(ram_arr.tobytes())
            lo = int(end)
    assert ram[0x80] == len(rows)
    assert fr.image is not None


def test_ram_changes_between_frames_outside_page_one_are_dirty():
    assert _run(WORK, 5000)[3].dirty
    assert not _run(PERIODIC, 5000)[3].dirty


def test_brk_is_no_frame_and_is_foreign_only_outside_one():
    outside = BRK_AT.replace("CMP #$02", "CMP #$FF")
    units, _ = _expect(outside, 6000, {"irq": CIA_IRQ})
    rows, events, _, fr, _ = _run(outside, 6000)
    assert fr.foreign == 1 << 4 and len(events) == 0
    _same(rows, events, units, whole=False)
    units, _ = _expect(BRK_AT, 6000, {"irq": CIA_IRQ})
    rows, events, _, fr, _ = _run(BRK_AT, 6000)
    _same(rows, events, units, whole=False)
    assert fr.foreign == 1 << 4 and len(events) == 0


@pytest.mark.parametrize(
    "src,budget,outcome",
    [
        (LOOP, smc.Budget(), "none"),
        (LOOP, smc.Budget(unit_insns=5), "unit_insns"),
        (LOOP, smc.Budget(unit_cycles=9), "unit_cycles"),
        (LOOP, smc.Budget(song_insns=20), "song_insns"),
        (LOOP, smc.Budget(song_cycles=90), "song_cycles"),
        (JAM, smc.Budget(), "jam"),
    ],
)
def test_a_limit_stops_the_run_and_cuts_the_open_frame(src, budget, outcome):
    m, _, _, labels = _machine(src, frames=False)
    trace = _trace(m, labels, 6000)
    rows, _, chunks, fr, _ = _run(src, 6000, window=700, budget=budget)
    assert smc.STOPS[fr.outcome] == outcome
    cut = outcome != "none"
    halts = np.concatenate(
        [
            np.frombuffer(c.records.meta, np.uint8).reshape(-1, 7)[:, 1]
            for c in chunks
            if c.entries
        ]
    )
    assert (len(halts) > 0 and halts[-1] != 0) == cut
    if outcome == "none":
        return
    starts = [d["t"] for d in trace if d["pc"] == labels["irq"]]
    last = starts[len(rows) - 1]

    def cut_at(limit):
        return next(d["t"] for d in trace if d["t"] > last + limit) - last

    spent = int(rows["cycles"][:-1].sum())
    if outcome == "unit_insns":
        assert list(rows["insns"]) == [5]
    elif outcome == "unit_cycles":
        assert list(rows["cycles"]) == [cut_at(9)]
    elif outcome == "song_insns":
        assert list(rows["insns"]) == [9, 9, 2]
    elif outcome == "song_cycles":
        assert rows["cycles"][-1] == cut_at(90 - spent) and spent < 90
    else:
        assert halts[-1] == 2


def test_frames_without_a_trace_or_a_second_arm():
    m = Machine(backend="native")
    rec = smc.Recorder(m, bytes(m.ram))
    with pytest.raises(RuntimeError):
        smc.Frames(rec, 0x1000)
    with pytest.raises(RuntimeError):
        m.smc_frames_take()


def _call_run(native, cycles=40_000):
    m, rec, _, labels = _machine(CALLS, frames=False)
    queue = [(1, labels["init"], 0)] + [(2, labels["play"], 0)] * 6
    if native:
        calls = smc.Calls(rec, queue, labels["idle"])
        m.run(m.reset(labels["idle"]) + cycles)
        rec.finish()
        calls.close()
        return calls.records(), m, labels
    trace, left = [], list(queue)

    def hook(mc):
        trace.append(dict(pc=mc.mpu.pc, t=mc.cycle, a=mc.mpu.a, x=mc.mpu.x, y=mc.mpu.y,
                          sp=mc.mpu.sp, p=mc.ram[0x100 | (mc.mpu.sp + 1) & 0xFF],
                          op=mc.ram[mc.mpu.pc]))  # fmt: skip

    def trap(mc):
        if not left:
            mc.stop = True
            return False
        _, pc, _ = left.pop(0)
        sp = mc.mpu.sp
        mc.ram[0x100 | sp], mc.ram[0x100 | (sp - 1) & 0xFF] = (
            (labels["idle"] - 1) >> 8, (labels["idle"] - 1) & 0xFF)  # fmt: skip
        mc.mpu.sp, mc.mpu.pc = (sp - 2) & 0xFF, pc
        return True

    m.set_decode_hook(hook)
    m.set_trap(trap, {labels["idle"]})
    m.run(m.reset(labels["idle"]) + cycles)
    return trace, m, labels


def test_interrupts_inside_a_play_call_are_events_of_its_record():
    rec, _, labels = _call_run(True)
    trace, m, _ = _call_run(False)
    units = model(trace, m.ram, {labels["irq"]: CIA_IRQ}, sentinel=labels["idle"])
    ev = np.frombuffer(rec.events, EVENT)
    assert (
        len(rec.meta) // 7 == 7
        and len(ev) > 6
        and len(ev) == sum(len(u["events"]) for u in units)
    )
    for k, u in enumerate(units[:7]):
        got = ev[ev["record"] == k]
        assert [
            (e["icount"], e["pos"], e["at"], e["depth"], e["kind"]) for e in got
        ] == [
            (w["icount"], w["pos"], w["at"], w["depth"], w["kind"]) for w in u["events"]
        ]
    assert set(ev["record"]) <= set(range(1, 7)) and (ev["depth"] == 0).all()
    assert _call_run(True)[1].ram[0x80] == m.ram[0x80] > 0


def _init_insns():
    """The instructions of the init call, from the plain run's decode trace."""
    trace, _, labels = _call_run(False)
    idle = [i for i, d in enumerate(trace) if d["pc"] == labels["idle"]]
    return idle[1] - idle[0]


@pytest.mark.parametrize(
    "budget,outcome,left",
    [
        (smc.Budget(), "none", 0),
        (smc.Budget(unit_insns="init"), "unit_insns", 5),
        (smc.Budget(song_insns="init"), "song_insns", 5),
        (smc.Budget(unit_cycles=100), "unit_cycles", None),
        (smc.Budget(song_cycles=500), "song_cycles", None),
    ],
)
def test_a_call_queue_stops_at_its_budget(budget, outcome, left):
    n = _init_insns()
    budget = smc.Budget(*(n + 5 if v == "init" else v for v in budget))
    m, rec, _, labels = _machine(CALLS, frames=False)
    queue = [(1, labels["init"], 0)] + [(2, labels["play"], 0)] * 6
    calls = smc.Calls(rec, queue, labels["idle"], budget=budget)
    m.run(m.reset(labels["idle"]) + 40_000)
    rec.finish()
    calls.close()
    halts = calls.records().meta[1::7]
    assert smc.STOPS[calls.outcome] == outcome
    assert bool(halts[-1]) == (outcome != "none") and set(halts[:-1]) <= {0}
    assert left is None or calls.left == left


def test_the_call_cycle_limit_is_the_unit_cycle_budget():
    m, rec, _, labels = _machine(CALLS, frames=False)
    queue = [(1, labels["init"], 0)] + [(2, labels["play"], 0)] * 6
    calls = smc.Calls(rec, queue, labels["idle"], call_cycles=100)
    m.run(m.reset(labels["idle"]) + 40_000)
    assert smc.STOPS[calls.outcome] == "unit_cycles"
