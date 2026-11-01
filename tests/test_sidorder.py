"""vagrant_moses.sidorder on synthetic write streams: the envelope behaviour the solver relies
on, the orders and their cover, and the hard restart."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from vagrant_moses import sidorder as S

FRAME = 19656
SAW = S.SAW


def advanced(writes, until=None):
    """A chip after ``writes`` ``(cycle, reg, value)``, run to ``until`` (default 1000 past)."""
    sid = S.chip()
    last = max(c for c, _, _ in writes)
    sid.trace(writes, until or last + 1000)
    return sid


def hard_restart_chip(prev_ad, frames=2):
    """A chip whose voice 1 played a note with ``prev_ad`` and then has had the gate off with
    SR 0 for ``frames`` frames."""
    sid = S.chip()
    sid.trace([(1, 5, prev_ad), (1, 6, 0), (2, 4, SAW)], 2 + frames * FRAME)
    return sid


def onsets(sid, writes):
    """Per rate-counter phase, the cycles after the first write until the envelope leaves 0."""
    c = sid.cycle
    ph = sid.envelope_phases(0, [(c + a, r, v) for a, r, v in writes], c + 60000)
    out = []
    for k in range(ph.count):
        t = ph.trajectory(k)
        rising = np.flatnonzero(t.envelope > 0)
        out.append(int(t.cycle[rising[0]] - c - writes[0][0]))
    return out


def test_test_bit_never_reaches_the_envelope():
    plain = [
        (10, 5, 0x34),
        (10, 6, 0x58),
        (20, 4, SAW | 1),
        (9000, 4, SAW),
        (20000, 4, SAW | 1),
    ]
    tested = [(c, r, v | 8 if c in (20, 9000) and r == 4 else v) for c, r, v in plain]
    tested += [
        (c + 100, 4, v & ~8 | 8 * (c == 20)) for c, r, v in plain if r == 4 and c == 20
    ]
    tested.sort()
    traces = []
    for writes in (plain, tested):
        sid = advanced([(1, 5, 0x34)], 2)
        traces.append(sid.envelope_trace(0, writes, 40000))
    assert len(traces[0]) > 20 and S.same(*traces)


def test_ad_is_latent_with_the_gate_off_and_active_with_it_on():
    base = advanced([(1, 5, 0x0F), (1, 6, 0xF0), (2, 4, SAW | 1)], 3000)
    c = base.cycle
    quiet = base.envelope_trace(0, [(c, 4, SAW)], c + 45000)
    off_ad = base.envelope_trace(0, [(c, 4, SAW), (c + 50, 5, 0x00)], c + 45000)
    assert S.same(quiet, off_ad) and len(quiet) > 2
    decaying = advanced([(1, 5, 0x08), (1, 6, 0x00), (2, 4, SAW | 1)], 3000)
    held = decaying.envelope_trace(0, [], c + 8000)
    on_ad = decaying.envelope_trace(0, [(c + 50, 5, 0x00)], c + 8000)
    assert len(held) > 2 and not S.same(held, on_ad)


def test_sr_is_active_with_the_gate_off():
    base = advanced([(1, 5, 0x00), (1, 6, 0xF0), (2, 4, SAW | 1)], 3000)
    c = base.cycle
    slow = base.envelope_trace(0, [(c, 4, SAW), (c + 1, 6, 0x00)], c + 20000)
    fast = base.envelope_trace(0, [(c, 4, SAW), (c + 1, 6, 0x08)], c + 20000)
    assert not S.same(slow, fast) and slow.envelope[-1] < fast.envelope[-1]


def test_order_matters_through_the_rate_counter():
    gate_ad = [(1, 4, SAW | 1), (10, 5, 0x0A)]
    ad_gate = [(1, 5, 0x0A), (10, 4, SAW | 1)]
    assert onsets(hard_restart_chip(0x0F), gate_ad) == [11, 10, 9, 8, 7, 6, 5, 32771, 2]
    assert onsets(hard_restart_chip(0xF0), gate_ad) == [
        32778,
        32777,
        32776,
        32775,
        32774,
        32773,
        32772,
        4,
        2,
    ]
    for prev in (0x0F, 0xF0):
        assert onsets(hard_restart_chip(prev), ad_gate) == [
            20,
            19,
            18,
            17,
            16,
            15,
            14,
            32780,
            11,
        ]


def test_settle_cycles_is_a_wrap_and_a_release():
    assert S.settle_cycles(S.PAL_CLOCK) == 39566


def stalled(decay, frames):
    """``(stalled phases, phases)`` of a note after ``frames`` frames of hard restart from a
    note decaying at ``decay``: the attack has not started a rate-counter wrap on."""
    sid = S.chip()
    sid.trace([(1, 5, decay), (1, 6, 0x88), (2, 4, SAW | 1)], 3 * FRAME)
    c = sid.cycle
    n = c + frames * FRAME
    writes = [(c, 5, 0), (c, 6, 0), (c, 4, SAW)]
    writes += [
        (n, 5, 0x0A),
        (n + 4, 6, 0x88),
        (n + 8, 4, SAW | 9),
        (n + 12, 4, SAW | 0x41),
    ]
    ph = sid.envelope_phases(0, writes, n + 4 * FRAME)
    count = 0
    for k in range(ph.count):
        t = ph.trajectory(k)
        rising = np.flatnonzero((t.envelope > 0) & (t.cycle > n))
        count += not len(rising) or t.cycle[rising[0]] - n >= S.WRAP
    return int(count), ph.count


def test_a_long_hard_restart_forgets_the_previous_decay():
    shares = {k: [stalled(d, k) for d in (0, 4, 0xA, 0xF)] for k in (1, 3, 4)}
    assert shares[1] == [(5, 9), (5, 149), (5, 1954), (6310, 31251)]
    assert shares[3] == shares[4] == [(5, 9), (83, 149), (1086, 1954), (17363, 31251)]
    long = [a / b for a, b in shares[3]]
    short = [a / b for a, b in shares[1]]
    assert max(long) - min(long) < 2e-3 and long[0] == pytest.approx(5 / 9)
    assert max(short) - min(short) > 0.5


def frame_stream(prev_dur, frame, frames=4, ad=0x04):
    """``(init, frames)`` of a note held ``prev_dur`` cycles whose next frame writes ``frame``
    ``(register, value)`` 4 cycles apart."""
    start = 2 + prev_dur
    init = [(1, 5, ad), (1, 6, 0x88), (2, 4, 1)]
    writes = [(start + 4 * i, r, v) for i, (r, v) in enumerate(frame)]
    return S.split_frames(init + writes, [start + j * FRAME for j in range(frames)])


PULSE = [(4, 0x40), (4, 0x41)]
BUILDS = {
    "sr first": [(6, 0x20), (5, 0x06)] + PULSE,
    "ad first": [(5, 0x06), (6, 0x20)] + PULSE,
    "gate first": PULSE + [(5, 0x06), (6, 0x20)],
}


@pytest.mark.parametrize(
    "build,serving", [("sr first", 0), ("ad first", 1), ("gate first", 2)]
)
def test_each_canonical_order_keeps_its_build_and_the_others_do_not(build, serving):
    init, frames = frame_stream(3003, BUILDS[build])
    found = S.order_masks(S.Timeline(init, frames), 3)
    assert len(found) == 1
    ((mask, (count, _)),) = found.items()
    keeps = S.bits(mask)
    assert count == 1
    assert [bool(keeps[S.ORDERS.index(o)]) for o in S.CANONICAL] == [
        i == serving for i in range(3)
    ]


def test_covers_finds_the_exact_minimum():
    masks = {S.mask_of(1, 2): 5, S.mask_of(2, 3): 1, S.mask_of(4): 2, 0: 9}
    assert S.covers(masks, 3) == (2, [(2, 4)])
    assert S.covers({S.mask_of(1): 1, S.mask_of(2): 1, S.mask_of(3): 1}, 2) == (
        None,
        [],
    )
    chain = {
        S.mask_of(0, 1): 1,
        S.mask_of(1, 2): 1,
        S.mask_of(2, 3): 1,
        S.mask_of(3, 4): 1,
    }
    size, found = S.covers(chain, 4)
    assert (
        size == 2
        and (1, 3) in found
        and all(
            all(any(i in c for i in np.flatnonzero(S.bits(m))) for m in chain)
            for c in found
        )
    )
    assert np.flatnonzero(S.bits(S.mask_of(0, 7, 59))).tolist() == [0, 7, 59]


def hard_restart_stream(note, held):
    """A note, a hard restart writing AD 0x35 and SR 0xB0 with the gate off, then ``note``
    ``(offset, register, value)`` ``held`` frames later."""
    start = 3005
    writes = [(1, 5, 0x04), (1, 6, 0x88), (2, 4, 1)]
    writes += [(start, 4, 0x20), (start + 4, 6, 0xB0), (start + 8, 5, 0x35)]
    writes += [(start + held * FRAME + o, r, v) for o, r, v in note]
    init, frames = S.split_frames(writes, [start + j * FRAME for j in range(held + 4)])
    return S.hard_restarts(S.Timeline(init, frames), 39566, 3)


ENVELOPE_FIRST = [(0, 5, 0x0A), (4, 6, 0x88), (8, 4, 0x41)]
GATE_FIRST = [(0, 4, 0x41), (2000, 5, 0x0A), (2004, 6, 0x88)]


def test_hard_restart_values_matter_only_when_the_gate_comes_first():
    assert dict(hard_restart_stream(ENVELOPE_FIRST, 2)) == {(False, 1, True): 1}
    assert dict(hard_restart_stream(GATE_FIRST, 2)) == {(False, 1, False): 1}
    assert dict(hard_restart_stream(ENVELOPE_FIRST, 4)) == {(True, 3, True): 1}
    assert dict(hard_restart_stream(GATE_FIRST, 4)) == {(True, 3, False): 1}


def test_events_are_edges_with_the_envelope_in_effect():
    note = [(5, 0x00), (6, 0xE5), (4, 0x09)]
    assert S.events((0x40, 0x0F, 0x00), note) == ((6, 0xE5), (5, 0x00), (4, 0x09))
    pulse = [(6, 0x48), (5, 0x08), (4, 0x40), (4, 0x41)]
    assert S.events((0x41, 0x08, 0x48), pulse) == ((4, 0x00), (4, 0x01))
    assert S.events((0x41, 0, 0), [(4, 0x21), (4, 0x41), (5, 0x0A)]) == ((5, 0x0A),)
    off = (0x40, 0x00, 0x00)
    assert S.events(off, [(5, 0x0A), (6, 0x88), (5, 0x0B), (4, 0x41)]) == (
        (6, 0x88),
        (5, 0x0B),
        (4, 0x01),
    )
    assert S.events(off, [(4, 0x41), (5, 0x0A)]) == ((4, 0x01), (5, 0x0A))


def test_orders_and_kept_writes():
    assert len(S.ORDERS) == len(set(S.ORDERS)) == 6 + 12 + 12 + 30
    assert all(o in S.ORDERS for o in S.CANONICAL)
    note = [(5, 0x00), (6, 0xE5), (4, 0x09)]
    assert S.kept(note, ("ctrl:2", "ctrl", "sr", "ad")) == [
        (4, 0x09),
        (6, 0xE5),
        (5, 0x00),
    ]
    assert S.kept([(6, 0x93), (6, 0x20)], ("sr:2", "sr")) == [(6, 0x93), (6, 0x20)]
    assert S.text(
        ((5, 0x0A), (6, 0x88), (4, 0x09), (4, 0x00))
    ) == "a0A s88 g+t+ g-t-" and (S.text(()) == "-")


def test_split_frames_and_ordered():
    raw = [(5, 4, 1), (9, 5, 2), (30, 6, 3), (31, 30, 9), (55, 4, 0)]
    init, frames = S.split_frames(raw, [9, 50])
    assert init.tolist() == [[5, 4, 1]]
    assert [(s, w[:, 0].tolist()) for s, w in frames] == [(9, [9, 30]), (50, [55])]
    w = np.array([(10, 6, 5), (20, 6, 7), (30, 5, 1), (40, 4, 0x41)], np.int64)
    out = S.ordered(w, ("ctrl", "ad", "sr"))
    assert out.tolist() == [[20, 4, 0x41], [30, 5, 1], [40, 6, 7]]


def test_pattern_changes_count_orders_that_alter_a_pattern():
    init, frames = frame_stream(3003, BUILDS["gate first"])
    changed, pairs = S.pattern_changes(init, frames)
    assert changed[S.ORDERS.index(S.CANONICAL[2])] == 0
    assert changed[S.ORDERS.index(S.CANONICAL[0])] == 1
    assert sum(n for *_, n in [(*k, n) for k, n in pairs.items()]) == changed.sum()


def test_order_cases_compare_the_envelopes_an_order_changes():
    init, frames = frame_stream(3003, BUILDS["gate first"])
    tl = S.Timeline(init, frames)
    cases = list(S.order_cases(tl, S.CANONICAL[0], 3))
    assert [c["voice"] for c, _ in cases] == [0]
    case, (start, native, new, until) = cases[0]
    assert case["ms"] > 0 and case["dbfs"] < 0 and case["native"] != case["ordered"]
    assert (
        until == tl.starts[3] and len(native) == len(new) == 4 and start == tl.starts[0]
    )
    same_order = list(S.order_cases(tl, S.CANONICAL[2], 3))
    assert same_order == []


def test_compare_and_levels():
    trace = np.rec.fromarrays(
        [[4, 8], [10, 20], [0, 0]], names=("cycle", "envelope", "state")
    )
    assert S.levels(trace, 4, 12).tolist() == [10] * 4 + [20] * 4
    out = S.compare(np.array([1, 2, 3]), np.array([1, 4, 3]), 1000.0)
    assert out["ms"] == pytest.approx(1.0) and out["db"] == pytest.approx(
        6.02, abs=0.01
    )
    assert S.compare(np.array([1]), np.array([1]))["dbfs"] == -np.inf


def test_neutral_clears_values_from_the_hard_restart_on():
    rows = np.array(
        [(0, 5, 0x2A), (10, 5, 0x0F), (10, 6, 0xF3), (20, 4, 0x20)], np.int64
    )
    assert S.neutral(rows, 10)[:, 2].tolist() == [0x2A, 0x00, 0x03, 0x20]


INSTRUMENTS = [
    (0x09, 0xA9),
    (0x4A, 0xF5),
    (0x00, 0xF0),
    (0x28, 0x88),
    (0x8C, 0x6B),
    (0x1F, 0xD3),
]


def busy_stream(frames, seed=7):
    """Three voices retriggered every few frames with varying envelopes and hard restarts."""
    rng = np.random.default_rng(seed)
    writes, starts = [(1, 24, 15)], [3000 + j * FRAME for j in range(frames)]
    for j, s in enumerate(starts):
        for v in range(3):
            if (j + v) % 4 == 0:
                writes += [
                    (s + 40 * v, 7 * v + 4, 0x40),
                    (s + 40 * v + 4, 7 * v + 6, 0x00),
                ]
            elif (j + v) % 4 == 2:
                ad, sr = INSTRUMENTS[int(rng.integers(len(INSTRUMENTS)))]
                writes += [
                    (s + 40 * v + 8 * i, 7 * v + r, x)
                    for i, (r, x) in enumerate([(5, ad), (6, sr | 0x08), (4, 0x41)])
                ]
    return S.split_frames(writes, starts)


def test_power_on_state_is_a_fresh_chips():
    sid = S.chip()
    assert {sid.envelope_state(v) for v in range(3)} == {S.POWER_ON}
    sid.write(5, 0x44)
    assert sid.envelope_state(0) != S.POWER_ON


def test_the_solver_builds_no_chip_while_it_judges_cases(monkeypatch):
    import pylibresidfp  # pylint: disable=import-outside-toplevel

    built = []
    real = pylibresidfp.SID.__init__
    monkeypatch.setattr(
        pylibresidfp.SID,
        "__init__",
        lambda self, *a, **k: built.append(1) or real(self, *a, **k),
    )
    init, frames = busy_stream(120)
    tl = S.Timeline(init, frames)
    masks = S.order_masks(tl, 10)
    notes = S.hard_restarts(tl, S.settle_cycles(), 10)
    cases = list(S.order_cases(tl, S.CANONICAL[0], 10))
    assert sum(n for n, _ in masks.values()) > 30 and sum(notes.values()) > 15 and cases
    assert not built


HOST_CALLS = 0.023


def machine_slowness():
    """How many times slower than the development host this machine runs the solver's own unit
    of work: 2000 single-voice envelope runs through :func:`sidorder.run` (best of five).
    """
    writes = np.array([(5, 5, 0x44), (10, 6, 0x88), (15, 4, 0x41)], np.int64)
    best = []
    for _ in range(5):
        begin = time.process_time()
        for _ in range(2000):
            S.run(S.POWER_ON, writes, FRAME, 0)
        best.append(time.process_time() - begin)
    return min(best) / HOST_CALLS


def real_time_multiple(frames=3000):
    """How many times faster than real time the solver judges a busy stream of ``frames`` PAL
    frames (the best of three runs, in CPU time), as it would on the development host.
    """
    init, stream = busy_stream(frames)
    elapsed = []
    for _ in range(3):
        begin = time.process_time()
        tl = S.Timeline(init, stream)
        masks = S.order_masks(tl, 10)
        notes = S.hard_restarts(tl, 39566, 10)
        elapsed.append(time.process_time() - begin)
    assert sum(n for n, _ in masks.values()) > 1000 and sum(notes.values()) > 400
    return frames * FRAME / S.PAL_CLOCK / min(elapsed) * machine_slowness()


def test_the_solver_runs_far_faster_than_real_time_on_a_busy_stream():
    env = {k: v for k, v in os.environ.items() if "COV" not in k}
    out = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(out.stdout) > 50


if __name__ == "__main__":
    print(real_time_multiple())
