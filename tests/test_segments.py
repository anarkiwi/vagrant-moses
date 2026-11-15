"""The native segment profiler: segment ends, keys, fork aggregation and determinism."""

from __future__ import annotations

import pytest

from vagrant_moses import machine as vm
from vagrant_moses.machine import Machine
from vagrant_moses.program import Program

pytest.importorskip("jennings")

# pylint: disable=wrong-import-position
import machine_diff as md
from asm import assemble

pytestmark = pytest.mark.skipif(not vm.NATIVE, reason="vagrant_moses._core not built")
ORG = 0x1000
IRQ_SOURCE = """
  SEI
  LDA #$7F
  STA $DC0D
  LDA #<handler
  STA $0314
  LDA #>handler
  STA $0315
  LDA #$00
  STA $DC04
  LDA #$10
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  CLI
loop: NOP
  JMP loop
handler: LDA $DC0D
  PLA
  TAY
  PLA
  TAX
  PLA
  RTI
"""
KEYS = """
  SEI
  LDA #$FF
  STA $DC02
  LDA #$00
  STA $DC00
loop: LDA $DC01
  STA $FB
  LDX #$10
work: LDA $FB
  EOR $0400,X
  STA $0400,X
  DEX
  BNE work
  JMP loop
"""


def _load(src, **kwargs):
    m = Machine(backend="native", **kwargs)
    code, labels = assemble(src, ORG)
    m.ram[ORG : ORG + len(code)] = code
    m.reset(ORG)
    return m, labels


def _profile(src, cycles=50_000, machine_kw=None, **kwargs):
    m, labels = _load(src, **(machine_kw or {}))
    m.profile_segments(**kwargs)
    m.run(m.t + cycles)
    return m.segment_profile(), labels


def _keys_at(profile, pc):
    return sum(t["keys"] for t in profile["top"] if t["pc"] == pc)


def test_the_python_machine_has_no_profiler():
    m = Machine(backend="python")
    with pytest.raises(NotImplementedError):
        m.profile_segments(max_insns=8)
    with pytest.raises(NotImplementedError):
        m.segment_profile()


def test_arming_disarming_and_clones():
    m, _ = _load("loop: JMP loop")
    assert m.segment_profile() is None
    with pytest.raises(ValueError):
        m.profile_segments(max_insns=0)
    with pytest.raises(ValueError):
        m.profile_segments(boundaries=[0x10000])
    m.profile_segments()
    m.run(m.t + 1000)
    assert m.segment_profile()["segments"] > 0
    clone = m.clone().segment_profile()
    assert clone is not None and clone["segments"] == clone["cycles"] == 0
    m.profile_segments(enabled=False)
    assert m.segment_profile() is None


def test_the_cap_bounds_a_quiet_loop_into_one_key():
    p, _ = _profile("loop: NOP\n  NOP\n  JMP loop", 70_000, max_insns=30)
    cap, cycles = p["ends"]["cap"]
    assert cap > 900 and cycles == 70 * cap
    assert p["segments"] == cap + p["ends"]["run"][0]
    assert p["ends"]["run"][0] == 1 and p["early_cycles"] == 0
    assert p["distinct"] == 2 and p["conflicts"] == 0
    assert p["saved"]["machine"] == p["saved"]["global"] == 70 * (cap - 1)
    assert p["lengths"] == [(16, 31, cap + 1, 70 * cap + p["ends"]["run"][1])]
    assert p["cycles"] == 70_000


@pytest.mark.parametrize(
    "insn, reu, cut",
    [
        ("LDA $D020", None, True),
        ("STA $D020", None, True),
        ("LDA $01", None, True),
        ("STA $01", None, True),
        ("STA $FF00", 128, True),
        ("STA $FF00", None, False),
        ("STA $2000", None, False),
    ],
)
def test_an_io_access_drops_its_instruction(insn, reu, cut):
    p, _ = _profile(
        f"loop: NOP\n  NOP\n  {insn}\n  JMP loop",
        machine_kw={"reu": reu},
        boundaries=[ORG],
    )
    n, cycles = p["ends"]["io"]
    if not cut:
        assert n == 0 and p["ends"]["pc"][0] > 0
        return
    assert n > 1000 and cycles == 4 * n == p["early_cycles"]
    assert p["ends"]["pc"][1] == 3 * p["ends"]["pc"][0]
    assert p["segment_cycles"] < p["cycles"] * 8 // 10


def test_stalls_and_interrupts_end_segments():
    badline = "  LDA #$1B\n  STA $D011\nloop: NOP\n  JMP loop"
    p, _ = _profile(badline, 312 * 63 * 2)
    assert p["ends"]["event"][0] > 20
    assert p["segment_cycles"] < p["cycles"] - 20 * 40
    p, labels = _profile(IRQ_SOURCE, 0x1000 * 8)
    assert p["ends"]["irq"][0] >= 6
    assert {0xFF48, labels["handler"] + 3} <= {t["pc"] for t in p["top"]}
    assert p["conflicts"] == 0


def test_pc_and_call_boundaries_align_segments():
    m, _ = _load("loop: NOP\n  NOP\n  NOP\n  JMP loop")
    m.profile_segments(boundaries=[ORG])
    m.run_program(Program(1_000_000).count_to(ORG, 1000))
    p = m.segment_profile()
    n, cycles = p["ends"]["pc"]
    assert (n, cycles, p["ends"]["run"]) == (999, 9 * 999, (0, 0))
    assert p["top"] == [
        {
            "pc": ORG,
            "keys": 1,
            "occurrences": 999,
            "cycles": 8991,
            "saved": 8982,
            "first": (0, 0),
        }
    ]
    src = "loop: JSR sub\n  JMP loop\nsub: NOP\n  RTS"
    p, labels = _profile(src, calls=True)
    assert {t["pc"] for t in p["top"][:2]} == {labels["sub"], ORG + 3}
    assert p["ends"]["pc"][0] > 1000 and p["ends"]["cap"] == (0, 0)


def test_a_hook_that_may_write_ends_a_segment_but_a_program_does_not():
    m, _ = _load("loop: NOP\n  NOP\n  JMP loop")
    m.set_trap(lambda _m: False, {ORG})
    m.profile_segments()
    m.run(m.t + 10_000)
    p = m.segment_profile()
    assert p["ends"]["host"][0] > 1000 and p["ends"]["cap"] == (0, 0)
    m.set_trap(None, set())
    m.set_decode_hook(lambda _m: None)
    m.profile_segments()
    m.run(m.t + 1000)
    assert [b[:2] for b in m.segment_profile()["lengths"]] == [(1, 1)]
    m.set_decode_hook(None)
    m.profile_segments(max_insns=64)
    m.run_program(Program(20_000).step(5_000))
    p = m.segment_profile()
    assert p["ends"]["cap"][0] > 50 and p["ends"]["host"] == (0, 0)


def test_a_key_follows_every_read_byte_and_only_those():
    src = "loop: LDA #$05\n  LDA $2000\n  STA $2001\n  JMP loop"
    m, _ = _load(src)
    m.profile_segments(boundaries=[ORG])
    count = Program(1_000_000).count_to(ORG, 50)
    seen = []
    for addr, value in ((None, 0), (0x2000, 1), (0x2001, 0x55), (ORG + 1, 0x05)):
        if addr is not None:
            m.ram[addr] = value
        m.run_program(count)
        seen.append(_keys_at(m.segment_profile(), ORG))
    m.ram[ORG + 1] = 0x06
    m.run_program(count)
    seen.append(_keys_at(m.segment_profile(), ORG))
    assert seen == [2, 3, 3, 3, 4]
    assert m.segment_profile()["conflicts"] == 0


def _keys_base():
    m, labels = _load(KEYS)
    m.run(m.t + 5_000)
    return m, labels


def _progs(labels, cols, frames=6):
    return [
        Program(1_000_000).key(0, c, True).count_to(labels["loop"], frames)
        for c in cols
    ]


def test_forks_fold_into_machine_batch_and_global_scopes():
    base, labels = _keys_base()
    base.profile_segments(calls=False)
    one = _progs(labels, [0])
    base.run_forks(one)
    single = base.segment_profile()
    base.profile_segments()
    progs = _progs(labels, [0, 1, 0, 1, 2, 3, 2, 3])
    base.run_forks(progs)
    p1 = base.segment_profile()
    assert (p1["forks"], p1["batches"]) == (8, 1)
    assert p1["distinct"] > single["distinct"]
    assert p1["saved"]["batch"] > p1["saved"]["machine"] > 0
    assert p1["saved"]["global"] == p1["saved"]["batch"]
    assert all(t["first"][0] == 1 and t["first"][1] < 4 for t in p1["top"])
    base.run_forks(progs)
    p2 = base.segment_profile()
    assert (p2["forks"], p2["batches"], p2["distinct"]) == (16, 2, p1["distinct"])
    assert p2["saved"]["machine"] == 2 * p1["saved"]["machine"]
    assert p2["saved"]["batch"] == 2 * p1["saved"]["batch"]
    added = p2["segment_cycles"] - p1["segment_cycles"]
    assert p2["saved"]["global"] == p1["saved"]["global"] + added
    assert p2["conflicts"] == 0
    _, forks = base.run_forks(progs[:2], keep=True)
    assert all(f.segment_profile()["segments"] == 0 for f in forks)
    assert base.segment_profile()["forks"] == 18


def test_a_batch_profile_does_not_depend_on_the_thread_count():
    base, labels = _keys_base()
    progs = _progs(labels, [i % 5 for i in range(12)], frames=3)
    got = []
    for threads in (1, 3, 8):
        base.profile_segments(calls=True, max_insns=40)
        base.run_forks(progs, threads=threads)
        base.run_forks(progs[:5], threads=threads)
        got.append(base.segment_profile(top=50))
    assert got[0] == got[1] == got[2]


@pytest.mark.parametrize("seed", range(6))
def test_profiling_observes_random_programs_without_changing_them(seed):
    runs = []
    for armed in (False, True):
        m = md._machine(
            seed, "native", md.Sid(), None
        )  # pylint: disable=protected-access
        start = md.load(m, seed)
        if armed:
            m.profile_segments(calls=bool(seed & 1), max_insns=1 + seed * 50)
        m.run(start + 30_000)
        runs.append(md.state(m))
    assert runs[0] == runs[1]
    assert m.segment_profile()["conflicts"] == 0
