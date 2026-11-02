"""Throw-away bus cycles reach the chips; host writes after a reset land on its cycle."""

from __future__ import annotations

import functools
import sys

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.machine import Machine

from asm import assemble

pytest.importorskip("jennings")

ORG = 0x1000
QUIET = "  SEI\n  LDA #$7F\n  STA $DC0D\n  STA $DD0D\n  LDA $DC0D\n  LDA $DD0D\n"
# CIA 1 timer A: a one-shot underflow sets ICR bit 0 with its interrupt masked.
UNDERFLOW = (
    "  LDA #$08\n  STA $DC04\n  LDA #$00\n  STA $DC05\n  LDA #$19\n  STA $DC0E\n"
)
WAIT = "  LDY #$10\nwait: DEY\n  BNE wait\n"


@pytest.fixture(autouse=True, params=["python"] + ["native"] * vm.NATIVE)
def backend(request, monkeypatch):
    """Run every test on each machine backend."""
    monkeypatch.setattr(
        sys.modules[__name__],
        "Machine",
        functools.partial(Machine, backend=request.param),
    )
    return request.param


class Sid:
    """Logs each write's value."""

    def __init__(self):
        self.values = []

    def read(self, _cycle, _register):
        return 0

    def write(self, _cycle, _register, value):
        self.values.append(value)


def _run(src, cycles=20000, entry_cycle=None, **kwargs):
    sid = Sid()
    machine = Machine(sids={mm.SID_BASE: sid}, **kwargs)
    code, _ = assemble(src, ORG)
    machine.ram[ORG : ORG + len(code)] = code
    start = machine.reset(ORG, entry_cycle)
    machine.run(start + cycles)
    return machine, sid


def _events(machine, name):
    return [cycle for cycle, event in machine.events if event == name]


def test_rti_in_cia2_acknowledges_every_nmi():
    """NMI to ``$DD0C`` holding RTI: its read of ``$DD0D`` clears the interrupt."""
    src = (
        QUIET
        + "  LDA #$35\n  STA $01\n  LDA #$0C\n  STA $FFFA\n  LDA #$DD\n  STA $FFFB\n"
        + "  LDA #$40\n  STA $DD0C\n  LDA #$FF\n  STA $DD04\n  LDA #$00\n  STA $DD05\n"
        + "  LDA #$81\n  STA $DD0D\n  LDA #$11\n  STA $DD0E\nloop: JMP loop\n"
    )
    machine, _ = _run(src)
    raises = _events(machine, "nmi")
    assert len(raises) > 50
    assert {b - a for a, b in zip(raises, raises[1:])} == {0x100}
    assert len(_events(machine, "nmi_entry")) == len(_events(machine, "rti"))


@pytest.mark.parametrize(
    "access, cleared",
    [
        ("LDX #$0E\n  LDA $DCFF,X", True),
        ("LDX #$0E\n  LDA $DCFE,X", False),
        ("LDX #$0E\n  LDA $DD00,X", False),
        ("LDA #$00\n  LDX #$00\n  STA $DC0D,X", True),
        ("LDA #$00\n  STA $DC0D", False),
        (
            "LDY #$00\n  LDA #$00\n  STA $FB\n  LDA #$DC\n  STA $FC\n  LDA #$00\n"
            "  LDY #$0D\n  STA ($FB),Y",
            True,
        ),
        ("LDX #$0E\n  LSR $DCFF,X", True),
    ],
)
def test_an_indexed_throw_away_read_acknowledges_cia1(access, cleared):
    """The unfixed-address read of a page cross (or of any indexed store) reads ICR."""
    src = (
        QUIET
        + UNDERFLOW
        + WAIT
        + f"  {access}\n  LDA $DC0D\n  STA $D400\nloop: JMP loop\n"
    )
    _, sid = _run(src, cycles=2000)
    assert sid.values == [0 if cleared else 1]


def test_brk_reads_its_signature_byte():
    """BRK's second cycle reads the byte after it: here from colour RAM, observed."""
    seen = []
    src = "  LDA #$00\n  STA $D800\n  JMP $D800\n"
    machine, _ = _run(src, cycles=100, on_io=lambda a, v, w: seen.append((a, w)))
    assert (0xD801, False) in seen
    assert _events(machine, "brk_entry")


def test_a_host_write_right_after_reset_lands_on_the_reset_cycle():
    """Starting a CIA timer from the host after ``reset(cycle)`` equals a CPU write then."""
    setup = [(0xDC04, 0x40), (0xDC05, 0x00), (0xDC0D, 0x81), (0xDC0E, 0x19)]
    start = vm.boot_cycle(False)
    host = Machine()
    host.ram[ORG : ORG + 3] = bytes((0x4C, ORG & 0xFF, ORG >> 8))
    host.reset(ORG, start)
    for addr, value in setup:
        host.io_write(addr, value)
    host.run(start + 200)
    src = "".join(f"  LDA #${v:02X}\n  STA ${a:04X}\n" for a, v in setup) + "l: JMP l\n"
    cpu, _ = _run(src, cycles=200 + 23, entry_cycle=start - 23)
    assert _events(host, "cia_irq") == _events(cpu, "cia_irq") != []
