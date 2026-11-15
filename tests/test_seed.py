"""CIA 1's keyboard matrix and chip-state seeding, on each machine backend."""

from __future__ import annotations

import functools
import sys

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.machine import Machine, boot_cycle

from asm import assemble

pytest.importorskip("jennings")

BACKENDS = ["python"] + ["native"] * vm.NATIVE


@pytest.fixture(autouse=True, params=BACKENDS)
def backend(request, monkeypatch):
    """Run every test on each machine backend."""
    monkeypatch.setattr(
        sys.modules[__name__],
        "Machine",
        functools.partial(Machine, backend=request.param),
    )
    return request.param


ORG, OUT = 0x1000, 0xC000
RETURN, A, S, SPACE = (0, 1), (1, 2), (1, 5), (7, 4)
SCAN = """
  SEI
  LDA #$FF
  STA $DC02
  LDA #$00
  STA $DC03
  LDX #$00
rows: LDA sel,X
  STA $DC00
  LDA $DC01
  STA $C000,X
  INX
  CPX #$08
  BNE rows
  LDA #$00
  STA $DC00
  LDA $DC01
  STA $C010
  LDA #$FE
  STA $DC02
  LDA $DC01
  STA $C011
  LDA #$00
  STA $DC02
  LDA #$FF
  STA $DC03
  LDX #$00
cols: LDA sel,X
  STA $DC01
  LDA $DC00
  STA $C008,X
  INX
  CPX #$08
  BNE cols
done: JMP done
sel: .byte $FE,$FD,$FB,$F7,$EF,$DF,$BF,$7F
"""


def _load(machine, src):
    code, labels = assemble(src, ORG)
    machine.ram[ORG : ORG + len(code)] = code
    return labels


def test_a_scan_sees_held_keys_through_both_ports():
    machine = Machine()
    _load(machine, SCAN)
    for row, col in (RETURN, A, S, SPACE):
        machine.set_key(row, col, True)
    machine.run(machine.reset(ORG) + 2000)
    out = machine.ram[OUT : OUT + 0x12]
    assert list(out[:8]) == [0xFD, 0xDB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xEF]
    assert list(out[8:16]) == [0xFF, 0xFE, 0xFD, 0xFF, 0x7F, 0xFD, 0xFF, 0xFF]
    assert (out[0x10], out[0x11]) == (0xC9, 0xCB)


def test_no_keys_leave_the_ports_as_driven():
    machine = Machine()
    _load(machine, SCAN)
    machine.run(machine.reset(ORG) + 2000)
    assert set(machine.ram[OUT : OUT + 0x12]) == {0xFF}


def test_keymatrix_set_key_and_release_all():
    machine = Machine()
    assert machine.keymatrix == bytes(8)
    machine.set_key(*A, True)
    machine.set_key(*S, True)
    machine.set_key(*S, False)
    assert machine.keymatrix == bytes((0, 0x04, 0, 0, 0, 0, 0, 0))
    machine.io_write(0xDC02, 0xFF)
    machine.io_write(0xDC00, 0xFD)
    assert machine.io_read(0xDC01) == 0xFB
    machine.keymatrix = bytes(range(8))
    assert machine.keymatrix == bytes(range(8))
    machine.release_all()
    assert machine.keymatrix == bytes(8)
    assert machine.io_read(0xDC01) == 0xFF
    with pytest.raises(ValueError):
        machine.keymatrix = bytes(7)
    with pytest.raises(ValueError):
        machine.set_key(8, 0, True)


def test_keys_leave_cia2_and_the_pb_timer_outputs_alone():
    machine = Machine()
    machine.keymatrix = b"\xff" * 8
    machine.io_write(0xDD02, 0xFF)
    machine.io_write(0xDD00, 0x00)
    assert machine.io_read(0xDD01) == 0xFF
    machine.io_write(0xDC02, 0xFF)
    machine.io_write(0xDC00, 0x00)
    machine.io_write(0xDC0E, 0x02)
    assert machine.io_read(0xDC01) & 0xC0 == 0x00
    machine.io_write(0xDC0E, 0x07)
    assert machine.io_read(0xDC01) & 0xC0 == 0x40


@pytest.mark.parametrize("cia_new", [False, True])
def test_a_seeded_timer_underflows_counter_cycles_on(cia_new):
    machine = Machine(cia_new=cia_new)
    _load(machine, "l: JMP l\n")
    machine.run(machine.reset(ORG) + 100)
    now, counter, latch = machine.cycle, 500, 1000
    machine.cia1.seed_timer(0, counter, latch, 0x01)
    machine.cia1.seed_icr(0x01)
    assert machine.cia1.timer(0) == (counter, latch)
    assert machine.cia1.icr_mask == 0x01 and machine.cia1.regs[0x0E] == 0x01
    assert machine.events == []
    machine.run(now + 3000)
    first = now + counter - (1 if cia_new else 0)
    raises = [c for c, e in machine.events if e == "cia_irq"]
    assert raises == [first]


def test_seed_icr_raises_nothing_until_the_next_underflow():
    machine = Machine()
    _load(machine, "l: JMP l\n")
    machine.run(machine.reset(ORG) + 100)
    now = machine.cycle
    machine.cia2.seed_timer(1, 50, 400, 0x01)
    machine.run(now + 200)
    machine.cia2.seed_icr(0x02)
    machine.run(now + 1000)
    assert [c for c, e in machine.events if e == "nmi"] == [now + 50 + 401]


def test_a_stopped_or_one_shot_seed_and_bad_arguments():
    machine = Machine()
    _load(machine, "l: JMP l\n")
    machine.run(machine.reset(ORG) + 100)
    now = machine.cycle
    machine.cia1.seed_icr(0x03)
    machine.cia1.seed_timer(0, 30, 100, 0x00)
    machine.cia1.seed_timer(1, 40, 100, 0x19)
    machine.run(now + 1000)
    assert machine.cia1.timer(0) == (30, 100)
    assert [c for c, e in machine.events if e == "cia_irq"] == [now + 40]
    with pytest.raises(ValueError):
        machine.cia1.seed_timer(2, 0, 0, 0)
    with pytest.raises(ValueError):
        machine.vic.seed_beam(mm.PAL.lines, 0)
    with pytest.raises(ValueError):
        machine.vic.seed_beam(0, mm.PAL.cycles_per_line)


def test_seed_raster_compare_sets_the_latch_not_the_beam():
    machine = Machine()
    machine.vic.seed_raster_compare(0x123)
    assert machine.vic.regs[0x12] == 0x23 and machine.vic.regs[0x11] & 0x80
    machine.vic.seed_raster_compare(0x40)
    assert machine.vic.regs[0x12] == 0x40 and not machine.vic.regs[0x11] & 0x80


ALIGNED = """
  SEI
  LDA $DC0D
  LDA #$0F
  STA $D019
  LDA #<irq
  STA $0314
  LDA #>irq
  STA $0315
  LDA #$81
  STA $DC0D
  LDA #$01
  STA $D01A
  CLI
loop: INC $C000
  JMP loop
irq: LDA $DC0D
  LDA #$0F
  STA $D019
  INC $C001
  JMP $EA81
"""


def _position(cycle, video):
    """``(line, line cycle)`` of ``cycle`` on a machine that ran from power-on."""
    return (
        cycle // video.cycles_per_line - 1
    ) % video.lines, cycle % video.cycles_per_line


@pytest.mark.parametrize(
    "ntsc, line, cycle, compare",
    [
        (False, 0x33, 20, 0x40),
        (False, 0x40, 5, 0x40),
        (False, 0x3F, 62, 0x40),
        (False, 0, 0, 0),
        (False, 311, 62, 0x100),
        (False, 0xF7, 30, 0x30),
        (True, 0x33, 11, 0x34),
        (True, 0, 1, 0x105),
    ],
)
def test_a_seeded_machine_runs_as_one_that_got_there(ntsc, line, cycle, compare):
    video, latch = mm.video(ntsc), 0x1B37
    boot = boot_cycle(ntsc)
    frame = video.lines * video.cycles_per_line
    there = (line + 1) * video.cycles_per_line + cycle - 1
    there += -(-(boot + 5000 - there) // frame) * frame
    assert _position(there + 1, video) == (line, cycle)

    natural = Machine(ntsc)
    for addr, value in (
        (0xD011, 0x1B | compare >> 1 & 0x80),
        (0xD012, compare & 0xFF),
        (0xDC04, latch & 0xFF),
        (0xDC05, latch >> 8),
        (0xDC0E, 0x11),
    ):
        natural.io_write(addr, value)
    _load(natural, ALIGNED)
    natural.reset(ORG, there)
    counter = natural.io_read(0xDC04) | natural.io_read(0xDC05) << 8

    seeded = Machine(ntsc)
    seeded.io_write(0xD011, 0x1B)
    _load(seeded, ALIGNED)
    seeded.reset(ORG, boot)
    seeded.vic.seed_raster_compare(compare)
    seeded.vic.seed_beam(line, cycle)
    seeded.cia1.seed_timer(0, counter, latch, 0x01)

    natural.run(there + 3 * frame)
    seeded.run(boot + 3 * frame)
    events = [
        [(c - t, e) for c, e in m.events] for m, t in ((natural, there), (seeded, boot))
    ]
    assert events[0] == events[1]
    assert {e for _, e in events[0]} >= {"video_irq", "cia_irq", "video_irq_entry"}
    assert natural.ram[0xC000:0xC002] == seeded.ram[0xC000:0xC002]
