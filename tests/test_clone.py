"""``Machine.clone``: a cycle-exact, independent fork on both backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.machine import Machine
from vagrant_moses.sid import PyNativeSid

pytest.importorskip("jennings")

# pylint: disable=wrong-import-position
import machine_diff as md
from asm import assemble

BACKENDS = ["python"] + ["native"] * vm.NATIVE
SID = {"python": PyNativeSid, "native": vm.NativeSid}
REU_DMA = """
  SEI
  LDA #$1B
  STA $D011
  LDA #$FF
  STA $D015
  LDA #$40
  STA $D001
  LDA #$00
  STA $DF02
  STA $DF04
  STA $DF05
  STA $DF06
  STA $DF08
  LDA #$10
  STA $DF03
  LDA #$60
  STA $DF07
  LDA #$90
  STA $DF01
  INC $D020
  JMP $0800
"""

TIMER_LOOP = """
  SEI
  LDA #<handler
  STA $0314
  LDA #>handler
  STA $0315
  LDA #$7F
  STA $DC0D
  LDA #$00
  STA $DC04
  LDA #$10
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  CLI
loop: INC $C000
  JMP loop
handler: LDA $DC0D
  INC $D020
  PLA
  TAY
  PLA
  TAX
  PLA
  RTI
"""


def _load(m, source):
    code, _ = assemble(source, 0x0800)
    m.ram[0x0800 : 0x0800 + len(code)] = bytes(code)
    return m.reset(0x0800)


def _machine(backend, seed=3, reu=None):
    ntsc, cia_new, _, _ = md.program(seed)
    m = Machine(ntsc, {mm.SID_BASE: md.Sid()}, cia_new, backend=backend, reu=reu)
    return m, md.load(m, seed)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("logged", [False, True])
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_a_clone_ends_as_the_original(backend, logged, seed):
    assert md.clone_divergence(seed, 150_000, backend, logged) is None


def _window(m, lo, hi, end):
    """Clone ``m`` at every cycle from ``lo`` to ``hi``; each clone run to ``end``.

    Returns the final states and how many clone points had an I/O write pending
    and an interrupt pending (Python backend: the native one does not expose them)."""
    clones, pending, irq = [], 0, 0
    m.run(lo)
    for t in range(lo, hi):
        m.run(t + 1)
        pending += bool(getattr(m, "pending", ()))
        irq += getattr(m, "ic", vm._MAX) != vm._MAX  # pylint: disable=protected-access
        clones.append(m.clone())
    m.run(end)
    want = md.state(m)
    for c in clones:
        c.run(end)
        assert md.state(c) == want
    return pending, irq


@pytest.mark.parametrize("backend", BACKENDS)
def test_clones_at_every_cycle_around_an_interrupt(backend):
    m, start = _machine(backend)
    probe, _ = _machine(backend)
    probe.run(start + 20_000)
    t = next(c for c, name in probe.events if name.endswith("_entry") and c > start)
    pending, irq = _window(m, t - 40, t + 40, t + 3000)
    if backend == "python":
        assert pending and irq


@pytest.mark.parametrize("backend", BACKENDS)
def test_clones_at_every_cycle_through_an_reu_dma(backend):
    m = Machine(backend=backend, reu=128)
    m.ram[0x1000:0x1060] = bytes(range(0x60))
    start = _load(m, REU_DMA)
    _window(m, start + 50, start + 250, start + 3000)
    assert bytes(m.reu.ram[:0x60]) == bytes(range(0x60))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_clone_is_independent(backend):
    m, start = _machine(backend, reu=128)
    m.run(start + 5000)
    c = m.clone()
    assert c is not m and c.ram is not m.ram
    c.ram[0x4000] = 0x5A
    c.colour[3] = 7
    c.reu.ram[9] = 1
    m.ram[0x4001] = 0xA5
    assert (m.ram[0x4000], m.colour[3], m.reu.ram[9]) != (0x5A, 7, 1)
    assert c.ram[0x4001] != 0xA5
    c.run(start + 9000)
    assert m.t == start + 5000 and c.t == start + 9000
    assert len(c.events) > len(m.events)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sinks_and_hooks_are_not_carried_over(backend):
    seen, io = [], []
    m = Machine(
        backend=backend,
        on_event=lambda *e: seen.append(e),
        on_io=lambda *a: io.append(a),
    )
    start = _load(m, TIMER_LOOP)
    m.run(start + 30_000)
    c = m.clone()
    n_seen, n_io = len(seen), len(io)
    c.run(start + 60_000)
    assert (len(seen), len(io)) == (n_seen, n_io)
    assert c.events and not m.events
    m.run(start + 60_000)
    assert [tuple(e) for e in c.events] == seen[n_seen:]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sid_devices(backend):
    host = md.Sid()
    logged = SID[backend]("8580", host.read)
    m = Machine(backend=backend, sids={0xD400: logged, 0xD500: host})
    m.io_write(0xD401, 0x42)
    c = m.clone()
    assert c.sids[0xD500] is host
    assert c.sids[0xD400] is not logged
    assert c.sids[0xD400].last == logged.last and not c.sids[0xD400].drain_writes()
    assert c.io_read(0xD402) == m.io_read(0xD402) == 0x42
    assert [w[1:] for w in logged.drain_writes()] == [(1, 0x42)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cloning_while_running_is_refused(backend):
    errors = []

    def sink(_cycle, _name):
        if not errors:
            with pytest.raises(RuntimeError) as err:
                m.clone()
            errors.append(err.value)

    m = Machine(backend=backend, on_event=sink)
    m.run(_load(m, TIMER_LOOP) + 30_000)
    assert errors and "running" in str(errors[0])
    m.clone()


@pytest.mark.skipif(not vm.NATIVE, reason="native machine not built")
def test_native_refuses_while_the_smc_recorder_is_armed():
    m = Machine(backend="native")
    m.smc_arm(bytes(0x10000), bytes(256))
    with pytest.raises(RuntimeError, match="SMC"):
        m.clone()
    m.smc_disarm()
    assert m.clone().cycle == m.cycle


@pytest.mark.parametrize("backend", BACKENDS)
def test_clones_run_in_parallel_threads(backend):
    m, start = _machine(backend, seed=3)
    m.run(start + 10_000)
    clones = [m.clone() for _ in range(8)]
    end = start + (200_000 if backend == "native" else 20_000)

    def go(c):
        c.run(end)
        return md.state(c)

    with ThreadPoolExecutor(8) as pool:
        states = list(pool.map(go, clones))
    m.run(end)
    assert states == [md.state(m)] * 8
