"""Driver programs and batched forks on both backends."""

from __future__ import annotations

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.machine import Machine
from vagrant_moses.program import MAX_OPS, Program, R, Reg, peekable, run_forks
from vagrant_moses.sid import PyNativeSid

pytest.importorskip("jennings")

# pylint: disable=wrong-import-position
import drivers as d
import machine_diff as md
from asm import assemble

BACKENDS = ["python"] + ["native"] * vm.NATIVE
native = pytest.mark.skipif(not vm.NATIVE, reason="vagrant_moses._core not built")
TIMER = """
  SEI
  LDA #$7F
  STA $DC0D
  LDA #$40
  STA $DC04
  LDA #$00
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
spin: INC $C000
  JMP spin
"""


def _load(backend, src, **kwargs):
    m = Machine(backend=backend, **kwargs)
    code, labels = assemble(src, d.ORG)
    m.ram[d.ORG : d.ORG + len(code)] = code
    m.reset(d.ORG)
    return m, labels


@pytest.mark.parametrize("backend", BACKENDS)
def test_register_ops(backend):
    m, _ = _load(backend, "loop: JMP loop")
    p = Program(1000).mov(R[0], 0xFFFFFFF0).add(R[0], 0x20).out(R[0])
    p.mov(R[1], 3).sub(R[1], 5).out(R[1]).mov(R[2], 0xF0).add8(R[2], 0x20).out(R[2])
    p.mov(R[3], 2).sub8(R[3], 3).out(R[3]).mov(R[4], 0b1100).and_(R[4], 0b1010)
    p.out(R[4]).or_(R[4], 1).out(R[4]).xor(R[4], R[4]).out(R[4])
    p.mov(R[5], 1).shl(R[5], 33).out(R[5]).shr(R[5], 1).out(R[5])
    p.mov(R[6], 0x81).bmi8(R[6], "neg").halt(9)
    p.label("neg").bpl8(R[6], "bad").mov(R[7], 4)
    p.label("loop").add(R[8], 2).djnz(R[7], "loop").out(R[8])
    p.mov(R[9], 7).beq(R[9], 7, "eq").halt(9)
    p.label("eq").bne(R[9], R[8], "ne").halt(9)
    p.label("ne").blt(R[9], 8, "lt").halt(9)
    p.label("lt").bge(R[9], R[9], "ge").halt(9)
    p.label("ge").blt(R[9], 7, "bad").bge(R[9], 8, "bad").jmp("end")
    p.label("bad").halt(9)
    p.label("end").mov(R[10], 42).halt(R[10])
    o = m.run_program(p)
    assert o.status == "halt" and o.code == 42 and o.frames == 0
    assert o.outputs == (0x10, 0xFFFFFFFE, 0x10, 0xFF, 0b1000, 0b1001, 0, 2, 1, 8)
    assert len(p) == len(p.code)


@pytest.mark.parametrize("backend", BACKENDS)
def test_keyboard_ops_set_the_matrix(backend):
    m, _ = _load(backend, "loop: JMP loop")
    rows = bytes(range(1, 9))
    o = m.run_program(Program(10).keys(rows))
    assert o.status == "halt" and m.keymatrix == rows
    p = Program(10).clear().key(7, 7).key(0, 0).key(0, 0, down=False)
    m.run_program(p.mov(R[1], 3 * 8 + 5).keyr(R[1]).keyr(R[1], down=False).keyr(R[1]))
    assert m.keymatrix == bytes((0, 0, 0, 1 << 5, 0, 0, 0, 1 << 7))


@pytest.mark.parametrize("backend", BACKENDS)
def test_loads_see_the_cpu_view_ram_and_io_without_side_effects(backend):
    m, labels = _load(backend, TIMER)
    m.ram[0xE000] = 0x5A
    m.colour[5] = 7
    p = Program(100_000).run_to(labels["spin"]).count_to(labels["spin"], 300)
    p.ld(R[0], 0xE000).ldr(R[1], 0xE000).ld(R[2], 0x0001).ld(R[3], 0x0000)
    p.ldio(R[4], 0xDC0D).ldio(R[5], 0xDC0D).ld(R[6], 0xDC0D).ldio(R[7], 0xD805)
    p.mov(R[8], 0x12).ldio(R[9], 0xD000, idx=R[8]).ld(R[10], 0xD012).out(R[0])
    for r in range(1, 11):
        p.out(R[r])
    o = m.run_program(p)
    assert o.status == "halt"
    kernal = m.pages[0xE][0xE000]
    icr = o.outputs[4]
    assert o.outputs[:4] == (kernal, 0x5A, m[1], m[0])
    assert icr & 0x81 == 0x81 and o.outputs[4:9] == (icr, icr, icr, 7, 0x12)
    assert o.outputs[9] == o.outputs[10] == m.vic.raster_y & 0xFF
    assert m.io_read(0xDC0D) == icr and m.io_read(0xDC0D) & 1 == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_cycle_counter_and_outmem(backend):
    m, _ = _load(backend, "loop: JMP loop")
    m.ram[0x2000:0x2003] = b"\x01\x02\x03"
    p = Program(10_000).step(5).cyc(R[0], R[1]).out(R[0]).out(R[1]).cyc(R[2]).out(R[2])
    o = m.run_program(p.outmem(0x2000, 3))
    assert o.outputs == (m.t & 0xFFFFFFFF, m.t >> 32, m.t & 0xFFFFFFFF, 1, 2, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_to_stops_before_the_pc_and_holds_while_parked(backend):
    m, labels = _load(backend, "a: NOP\nb: INC $C000\n  JMP a")
    m.ram[0xC000] = 0
    m.run_program(Program(1000).run_to(labels["b"]))
    t = m.t
    assert m.mpu.pc == labels["b"] and m.ram[0xC000] == 0
    o = m.run_program(Program(1000).run_to({labels["b"], labels["a"]}))
    assert o.cycle == t and m.ram[0xC000] == 0
    m.run_program(Program(1000).run_to(labels["b"], leave=True))
    assert m.mpu.pc == labels["b"] and m.ram[0xC000] == 1 and m.t > t
    m.run_program(Program(1000).step(2))
    assert m.mpu.pc == labels["a"] and m.ram[0xC000] == 2
    m.run(m.t + 1)
    o = m.run_program(Program(1000).run_to(labels["a"]))
    assert m.mpu.pc == labels["a"] and o.cycle == m.t > t and m.ram[0xC000] == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_count_to_counts_arrivals_and_the_marker_counts_every_run(backend):
    m, labels = _load(backend, "a: INC $C000\n  JMP a")
    m.ram[0xC000] = 0
    p = Program(10_000, marker=labels["a"]).run_to(labels["a"]).count_to(labels["a"], 5)
    p.mov(R[1], 3).count_to(labels["a"], R[1]).count_to(labels["a"], 0).frames(R[0])
    o = m.run_program(p.out(R[0]).step(R[1]).step(0).frames(R[0]).out(R[0]))
    assert m.ram[0xC000] == 10 and o.outputs == (9, 10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_budget_jam_ops_and_fault(backend):
    m, labels = _load(backend, "a: INC $C000\n  JMP a\nj: .byte $02")
    start = m.t
    o = m.run_program(Program(500).count_to(labels["a"], 10_000))
    assert (o.status, o.code) == ("budget", 0) and o.cycle == start + 500
    o = m.run_program(Program(0).step())
    assert (o.status, o.cycle) == ("budget", start + 500)
    o = m.run_program(Program(100, max_ops=3).label("x").jmp("x"))
    assert (o.status, o.code) == ("ops", 0)
    o = m.run_program(Program(100).mov(R[0], 1).ld(R[1], 0xD400))
    assert (o.status, o.code) == ("fault", 0xD400)
    o = m.run_program(Program(100).outmem(0xD3FE, 4))
    assert (o.status, o.code, len(o.outputs)) == ("fault", 0xD400, 2)
    o = m.run_program(Program(100).mov(R[0], 0x10).ldio(R[1], 0xD3F0, idx=R[0]))
    assert (o.status, o.code) == ("fault", 0xD400)
    j, _ = _load(backend, "  JMP j\n  NOP\nj: .byte $02")
    o = j.run_program(Program(10_000).step(5))
    assert o.status == "jam" and j.jammed
    assert j.run_program(Program(10).run_to(0)).status == "jam"
    assert m.run_program(Program(10)).status == "halt"


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_three_driver_loops(backend):
    m, labels = d.game("angle", backend, angle=250, limit=0x80)
    o = m.run_program(d.steer(labels, 5))
    assert (o.status, o.code, m.ram[d.ANGLE], m.keymatrix) == ("halt", 0, 5, bytes(8))
    assert o.outputs == (12,)
    m, labels = d.game("angle", backend, angle=0x70, limit=0x80)
    o = m.run_program(d.steer(labels, 0x90))
    assert (o.status, o.code, m.ram[d.ANGLE]) == ("halt", 1, 0x80)
    m, labels = d.game("angle", backend, angle=0x40, limit=0x80)
    assert m.run_program(d.steer(labels, 0x30)).code == 0 and m.ram[d.ANGLE] == 0x30
    m, labels = d.game("scan", backend, count=0, frame=0)
    o = m.run_program(d.press(labels, 8))
    assert (o.status, o.code, o.outputs, m.ram[d.COUNT]) == ("halt", 0, (5,), 1)
    assert m.keymatrix == bytes(8)
    m, labels = d.game("scan", backend, count=0, frame=0)
    assert m.run_program(d.press(labels, 3)).code == 1 and m.ram[d.COUNT] == 0
    m, labels = d.game("cursor", backend, x=10, y=20)
    o = m.run_program(d.walk(labels, 13, 7))
    assert (o.code, m.ram[d.X], m.ram[d.Y], m.keymatrix) == (0, 13, 7, bytes(8))
    m, labels = d.game("cursor", backend, x=10, y=20)
    assert m.run_program(d.walk(labels, 13, 70, steps=8)).code == 1


@native
def test_backends_agree():
    for name, ram, build in (
        ("angle", {"angle": 3, "limit": 0x80}, lambda l: d.steer(l, 0xF0)),
        ("scan", {"count": 0, "frame": 1}, lambda l: d.press(l, 9)),
        ("cursor", {"x": 0, "y": 0}, lambda l: d.walk(l, 4, 9)),
    ):
        runs = []
        for backend in BACKENDS:
            m, labels = d.game(name, backend, **ram)
            runs.append((m.run_program(build(labels)), md.state(m)))
        assert runs[0] == runs[1]


@native
def test_a_program_runs_as_traps_and_keys_driven_from_python():
    m, labels = d.game("angle", "native", angle=250, limit=0x80)
    o = m.run_program(d.steer(labels, 5))
    ref, _ = d.game("angle", "native", angle=250, limit=0x80)
    ref.set_trap(lambda mc: setattr(mc, "stop", True), {labels["pass"]})
    limit = ref.t + 2_000_000
    while True:
        ref.run(limit)
        angle = ref.ram[d.ANGLE]
        ref.release_all()
        if angle == 5:
            break
        ref.set_key(*(d.INC_KEY if (5 - angle) & 0x80 == 0 else d.DEC_KEY), True)
    assert md.state(ref) == md.state(m) and o.cycle == ref.t


@native
def test_hooks_are_suspended_and_restored():
    m, labels = d.game("angle", "native", angle=0, limit=0x80)
    calls = []
    m.set_trap(lambda mc: calls.append(mc.mpu.pc), {labels["pass"]})
    m.arm_at(labels["noinc"])
    m.run_program(d.steer(labels, 3))
    assert not calls and not m.armed and m.watch == frozenset()
    m.run(m.t + 2000)
    assert m.armed and calls


def test_python_decode_override_is_restored():
    m, labels = d.game("angle", "python", angle=0, limit=0x80)
    seen = []
    plain = m._decode  # pylint: disable=protected-access

    def mine():
        seen.append(m.mpu.pc)
        plain()

    m._decode = mine  # pylint: disable=protected-access
    m.run_program(d.steer(labels, 3))
    assert m._decode is mine and seen  # pylint: disable=protected-access


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_forks_equal_sequential_clones_at_any_thread_count(backend):
    base, labels = d.game("angle", backend, angle=0x40, limit=0x80)
    base.run(base.t + 3000)
    targets = range(0, 256, 7 if backend == "native" else 51)
    progs = [d.steer(labels, t, budget=200_000) for t in targets]
    want = [base.clone().run_program(p) for p in progs]
    assert {o.code for o in want} == {0, 1}
    t0 = base.t
    for threads in (1, 3, 8, None):
        assert run_forks(base, progs, threads=threads) == want
    outs, forks = run_forks(base, progs, threads=4, keep=True)
    assert outs == want and base.t == t0 and base.ram[d.ANGLE] == 0x40
    for p, fork in zip(progs, forks):
        clone = base.clone()
        clone.run_program(p)
        assert md.state(fork) == md.state(clone)
        fork.run(fork.t + 5000)
        clone.run(clone.t + 5000)
        assert md.state(fork) == md.state(clone)
    assert run_forks(base, []) == [] and run_forks(base, [], keep=True) == ([], [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_forks_are_independent(backend):
    base, labels = d.game("cursor", backend, x=5, y=5)
    _, forks = run_forks(
        base, [d.walk(labels, x, 9) for x in (0, 7, 12)], threads=3, keep=True
    )
    assert [f.ram[d.X] for f in forks] == [0, 7, 12] and base.ram[d.X] == 5
    forks[0].ram[0x3000] = 0xAB
    assert base.ram[0x3000] != 0xAB and forks[1].ram[0x3000] != 0xAB
    twin = forks[1].clone()
    kernal = base.pages[0xE]
    old = kernal[0xE000]
    forks[2].pages[0xE][0xE000] = old ^ 0xFF
    assert kernal[0xE000] == old and twin.pages[0xE][0xE000] == old
    kernal[0xE001] ^= 0xFF
    later = base.clone()
    kernal[0xE002] ^= 0xFF
    assert later.pages[0xE][0xE001] == kernal[0xE001]
    assert later.pages[0xE][0xE002] != kernal[0xE002]
    assert forks[1].pages[0xE][0xE001] != kernal[0xE001]


@pytest.mark.parametrize("backend", BACKENDS)
def test_forks_with_a_native_sid(backend):
    reads = []
    sid = (PyNativeSid if backend == "python" else vm.NativeSid)(
        "6581", lambda cycle, reg: reads.append(reg) or 0x42
    )
    base, _ = _load(
        backend, "a: LDA $D41B\n  STA $C000\n  JMP a", sids={mm.SID_BASE: sid}
    )
    outs = run_forks(base, [Program(1000).step(3).ld(R[0], 0xC000).out(R[0])] * 3)
    assert [o.outputs for o in outs] == [(0x42,)] * 3 and reads == [0x1B] * 3
    plain, _ = _load(backend, "a: JMP a", sids={mm.SID_BASE: md.Sid()})
    with pytest.raises(ValueError):
        run_forks(plain, [Program(10)])


@native
def test_a_failing_sid_read_raises_and_the_recorder_refuses():
    def boom(_cycle, _reg):
        raise KeyError("sid")

    base, _ = _load(
        "native",
        "a: LDA $D41B\n  JMP a",
        sids={mm.SID_BASE: vm.NativeSid("6581", boom)},
    )
    with pytest.raises(KeyError):
        base.run_program(Program(1000).step(5))
    with pytest.raises(KeyError):
        run_forks(base, [Program(1000).step(5)] * 2)
    m, _ = _load("native", "a: JMP a")
    m.smc_arm(bytes(0x10000), bytes(256))
    for call in (lambda: m.run_program(Program(10)), lambda: run_forks(m, [])):
        with pytest.raises(RuntimeError):
            call()


def test_program_validation():
    p = Program(10)
    for bad in (
        lambda: p.mov(0, 1),
        lambda: p.mov(Reg(16), 1),
        lambda: p.mov(R[0], -1),
        lambda: p.mov(R[0], 1 << 32),
        lambda: p.key(8, 0),
        lambda: p.keys(b"\0"),
        lambda: p.ldio(R[0], 0xD400),
        lambda: p.ld(R[0], 0x10000),
        lambda: p.outmem(0, 0x10000),
        lambda: Program(-1),
        lambda: Program(1, marker=0x10000),
        lambda: p.label("x").label("x"),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        _ = Program(10).jmp("nowhere").code
    assert peekable(0xD020) and peekable(0xDD0D) and not peekable(0xDE00)
    assert Program(1).max_ops == MAX_OPS


@native
def test_native_code_validation_and_reuse():
    core = vm._core  # pylint: disable=protected-access
    for code in (
        [("NOPE",)],
        [("JMP", 5)],
        [("ALU", 0, 16, 0, 0)],
        [("ALU", 11, 0, 0, 0)],
        [("BR", 7, 0, 0, 0, 0)],
        [("HALT", 2, 0)],
        [("RUN", [0x10000], 0, 1, 0)],
    ):
        with pytest.raises(ValueError):
            core.Code(code, 10, None, 10)
    p = d.steer(d.game("angle", "native")[1], 3)
    assert p.native() is p.native() and len(p.native()) == len(p)
    m, labels = d.game("angle", "native", angle=0, limit=0x80)
    assert m.run_program(p.native()).code == 0 and m.ram[d.ANGLE] == 3
    m2, _ = d.game("angle", "native", angle=9, limit=0x80)
    assert m2.run_program(p).code == 0 and m2.ram[d.ANGLE] == 3 and labels


@native
@pytest.mark.parametrize("seed", [0, 2, 5])
def test_random_programs_and_forks_agree_on_both_backends(seed):
    assert md.program_divergence(seed, 400_000) is None


def test_a_python_program_is_refused_while_its_machine_runs():
    calls = []

    def nested(_cycle, _reg):
        with pytest.raises(RuntimeError):
            m.run_program(Program(10))
        calls.append(1)
        return 0

    m, _ = _load(
        "python",
        "a: LDA $D41B\n  JMP a",
        sids={mm.SID_BASE: PyNativeSid("6581", nested)},
    )
    assert m.run_program(Program(1000).step(3)).status == "halt" and calls
