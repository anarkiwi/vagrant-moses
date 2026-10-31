"""NativeSid: the write log, the bus-value model and the reads routed to ``on_read``."""

from __future__ import annotations

import os
import sys
import threading

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.machine import Machine
from vagrant_moses.sid import BUS_TTL, PyNativeSid

from asm import assemble

pytest.importorskip("jennings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import thread_machines  # pylint: disable=wrong-import-position

CLASSES = [PyNativeSid] + [vm.NativeSid] * vm.NATIVE
BACKENDS = ["python"] + ["native"] * vm.NATIVE
ORG = 0x1000


@pytest.fixture(params=CLASSES, ids=lambda c: c.__module__)
def sid_cls(request):
    return request.param


def _chip(cls, model="6581"):
    calls = []

    def on_read(cycle, reg):
        calls.append((cycle, reg))
        return 0x40 + reg

    return cls(model, on_read), calls


def test_writes_and_changed_rows_drain_and_clear(sid_cls):
    sid, _ = _chip(sid_cls)
    assert sid.last == bytes(32)
    for cycle, reg, value in [(1, 7, 0), (2, 7, 5), (3, 39, 5), (4, 7, 0)]:
        sid.write(cycle, reg, value)
    assert sid.drain_writes() == [(1, 7, 0), (2, 7, 5), (3, 7, 5), (4, 7, 0)]
    assert sid.drain_rows() == [(2, 7, 5), (4, 7, 0)]
    assert sid.drain_writes() == sid.drain_rows() == []
    assert sid.last[7] == 0
    sid.write(5, 0x18, 15)
    assert sid.last[0x18] == 15 and sid.drain_rows() == [(5, 0x18, 15)]


def test_rejects_unknown_model_and_bad_byte(sid_cls):
    with pytest.raises(ValueError):
        sid_cls("6582", print)
    sid, _ = _chip(sid_cls)
    with pytest.raises(ValueError):
        sid.write(0, 0, 256)


@pytest.mark.parametrize("model", ["6581", "8580"])
def test_bus_value_holds_for_its_ttl_then_reads_zero(sid_cls, model):
    ttl = BUS_TTL[model]
    sid, calls = _chip(sid_cls, model)
    sid.write(10, 3, 0xAB)
    assert sid.read(10 + ttl - 1, 0) == 0xAB
    sid.write(10, 3, 0xAB)
    assert sid.read(10 + ttl, 0x1F) == 0
    assert calls == []


def test_reading_an_unreadable_register_halves_the_ttl(sid_cls):
    sid, _ = _chip(sid_cls)
    sid.write(0, 3, 0xAB)
    assert sid.read(1, 0) == 0xAB
    half = (BUS_TTL["6581"] - 1) // 2
    assert sid.read(1 + half, 0) == 0


def test_a_readable_register_comes_from_on_read_and_drives_the_bus(sid_cls):
    sid, calls = _chip(sid_cls)
    assert sid.read(100, 0x1B) == 0x5B
    assert sid.read(101, 0x39) == 0x59
    assert calls == [(100, 0x1B), (101, 0x19)]
    assert sid.read(102, 0x00) == 0x59
    assert sid.read(102 + BUS_TTL["6581"], 0x00) == 0


def test_a_bad_on_read_byte_is_an_error(sid_cls):
    sid = sid_cls("6581", lambda _c, _r: 300)
    with pytest.raises((ValueError, OverflowError)):
        sid.read(0, 0x19)


PROGRAM = """
  LDA #$AB
  STA $D400
  LDA $D400
  STA $C000
  LDA $D41B
  STA $C001
  LDA $D400
  STA $C002
  LDA $D45C
  STA $C003
  LDX #$10
  STX $D401
  INC $D401
  INC $D41C
  LDX #$00
  LDY #$00
wait: DEX
  BNE wait
  DEY
  BNE wait
  LDA $D400
  STA $C004
done: JMP done
"""


def _run(backend, dev):
    machine = Machine(sids={mm.SID_BASE: dev}, backend=backend)
    code, _ = assemble(PROGRAM, ORG)
    machine.ram[ORG : ORG + len(code)] = code
    machine.run(machine.reset(ORG) + 4000 + 256 * 1281 * 2)
    return bytes(machine.ram[0xC000:0xC005])


class Chip:
    """The same chip as a plain device and behind NativeSid, for comparison."""

    def __init__(self):
        self.writes, self.reads = [], []

    def value(self, cycle, reg):
        self.reads.append((cycle, reg))
        return 0x40 + reg

    def read(self, cycle, reg):
        return self.value(cycle, reg)

    def write(self, cycle, reg, value):
        self.writes.append((cycle, reg, value))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_machine_reads_the_bus_value_and_calls_python_only_for_the_chip(
    backend, sid_cls
):
    chip, drained = Chip(), []

    def on_read(cycle, reg):
        drained.extend(sid.drain_writes())
        return chip.value(cycle, reg)

    sid = sid_cls("6581", on_read)
    ram = _run(backend, sid)
    assert ram == bytes((0xAB, 0x5B, 0x5B, 0x5C, 0x00))
    assert [reg for _, reg in chip.reads] == [0x1B, 0x1C, 0x1C]
    assert [w[1:] for w in drained] == [(0, 0xAB), (1, 0x10), (1, 0x10), (1, 0x11)]
    rest = sid.drain_writes()
    assert [w[1:] for w in rest] == [(0x1C, 0x5C), (0x1C, 0x5D)]
    assert sid.last[1] == 0x11 and sid.last[0x1C] == 0x5D


@pytest.mark.skipif(not vm.NATIVE, reason="vagrant_moses._core not built")
def test_the_sid_is_refused_to_other_threads_while_the_machine_runs():
    seen = []

    def on_read(_cycle, _reg):
        def poke():
            try:
                sid.drain_writes()
            except RuntimeError as exc:
                seen.append(exc)

        thread = threading.Thread(target=poke)
        thread.start()
        thread.join()
        seen.append(sid.drain_writes())
        return 0

    sid = vm.NativeSid("6581", on_read)
    machine = Machine(sids={mm.SID_BASE: sid}, backend="native")
    code, _ = assemble("  STA $D400\n  LDA $D419\nl: JMP l\n", ORG)
    machine.ram[ORG : ORG + len(code)] = code
    machine.run(machine.reset(ORG) + 100)
    assert isinstance(seen[0], RuntimeError) and seen[1] == [(seen[1][0][0], 0, 0)]
    assert sid.drain_writes() == []


@pytest.mark.skipif(not vm.NATIVE, reason="vagrant_moses._core not built")
def test_threads_with_native_sids_run_in_parallel():
    threads = min(8, os.cpu_count() or 1)
    if threads < 2:
        pytest.skip("needs more than one CPU")
    cycles = 10_000_000
    single = thread_machines.wall(1, cycles, sid=True)
    assert thread_machines.wall(threads, cycles, sid=True) < 3 * single
