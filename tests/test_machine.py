"""The cycle-exact machine against documented 6510, CIA and VIC-II timing."""

from __future__ import annotations

import builtins
import collections
import functools
import sys

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.machine import BR, X2, Machine, boot_cycle, opcode_layout

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


ORG = 0x1000
LINE, FRAME = mm.PAL.cycles_per_line, mm.PAL.cycles_per_frame
QUIET = """
  SEI
  LDA #$7F
  STA $DC0D
  LDA $DC0D
  LDA #$1B
  STA $D011
"""
LOOP = """
  LDX #$00
loop: INX
  STX $D418
  JMP loop
"""


class Sid:
    """A SID stand-in logging each write as ``(cycle, register, value)``."""

    def __init__(self, value=0):
        self.writes, self.value = [], value

    def read(self, _cycle, register):
        return self.value | register

    def write(self, cycle, register, value):
        self.writes.append((cycle, register, value))


def _boot(src, seconds=0.25, **kwargs):
    sid = Sid()
    machine = Machine(sids={mm.SID_BASE: sid}, **kwargs)
    code, labels = assemble(src, ORG)
    machine.ram[ORG : ORG + len(code)] = code
    start = machine.reset(ORG)
    machine.run(start + int(seconds * mm.PAL.clock_hz))
    return machine, sid, labels


def _events(machine, name):
    return [cycle for cycle, event in machine.events if event == name]


def _line(cycle):
    """The PAL raster line the beam is on at ``cycle``, and the line cycle."""
    return (cycle // LINE - 1) % mm.PAL.lines, cycle % LINE


def _vector(name, vec=mm.RAM_IRQ_VECTOR_LO):
    return f"  LDA #<{name}\n  STA ${vec:04X}\n  LDA #>{name}\n  STA ${vec + 1:04X}\n"


def test_every_opcode_takes_its_documented_cycles():
    from deity_informant import CYCLETIME, EXTRACYCLES
    from jennings.opcodes import OPCODES

    for op in range(256):
        ents, exec_at, kind = opcode_layout(op)
        if OPCODES[op][0] == "JAM":
            assert ents == (vm.HALT,)
            continue
        fetch = next(i for i, k in enumerate(ents) if k in vm.FETCHES or k == BR)
        base = 1 + fetch - (X2 in ents)
        if kind == "branch":
            assert (base, EXTRACYCLES[op]) == (2, 2)
        else:
            assert base == CYCLETIME[op], hex(op)
            assert (X2 in ents) == bool(EXTRACYCLES[op]), hex(op)
        assert 0 <= exec_at < len(ents) or OPCODES[op][0] in ("BRK", "ANE")


def test_layouts_put_writes_last_and_reads_before():
    assert opcode_layout(0x8D)[0] == (vm.R, vm.R, vm.W, vm.F)
    assert opcode_layout(0xEE)[0][-3:] == (vm.W, vm.W, vm.F)
    assert opcode_layout(0xEE)[2] == "rmw"
    assert opcode_layout(0x20)[0] == (vm.R, vm.R, vm.W, vm.W, vm.R, vm.F)
    assert opcode_layout(0x00)[0] == vm.BRK_ENTRIES


def test_boot_cycle_is_sidplayfps_power_on_delay():
    assert boot_cycle(False) == 339711
    assert boot_cycle(True) != boot_cycle(False)


def test_reset_leaves_the_port_and_registers_as_sidplayfp():
    machine, _, _ = _boot("loop: JMP loop", seconds=0.001)
    assert (machine[0], machine[1]) == (mm.DDR_DEFAULT, mm.BANK_DEFAULT)
    assert machine.mpu.sp == 0xFF and machine.cycle > boot_cycle(False)
    assert machine.mpu.pc in (ORG, ORG + 1, ORG + 2)


def test_cia_timer_irq_raises_every_latch_plus_one_cycles():
    body = """
  LDA #$34
  STA $DC04
  LDA #$12
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  CLI
loop: JMP loop
irq: INC $D400
  JMP $EA31
"""
    src = QUIET + _vector("irq") + body
    machine, sid, _ = _boot(src)
    raises = _events(machine, "cia_irq")
    assert len(raises) > 10
    assert {b - a for a, b in zip(raises, raises[1:])} == {0x1235}
    assert len(_events(machine, "cia_irq_entry")) == len(raises)
    assert len(sid.writes) == 2 * len(raises)  # INC writes twice: RMW dummy write


def test_one_shot_timer_raises_once():
    src = (
        QUIET
        + "  LDA #$81\n  STA $DC0D\n  LDA #$19\n  STA $DC0E\n  CLI\nloop: JMP loop\n"
    )
    machine, _, _ = _boot(src)
    assert len(_events(machine, "cia_irq")) == 1


def test_nmi_is_edge_triggered_until_icr_read():
    body = """
  LDA #$00
  STA $DD04
  STA $DD05
  LDA #$81
  STA $DD0D
  LDA #$01
  STA $DD0E
loop: JMP loop
nmi: INC $D400
  RTI
"""
    src = QUIET + _vector("nmi", mm.RAM_NMI_VECTOR_LO) + body
    machine, _, _ = _boot(src)
    assert len(_events(machine, "nmi")) == len(_events(machine, "nmi_entry")) == 1
    assert _events(machine, "rti")


def test_an_irq_waits_for_cli_then_one_more_instruction():
    body = """
  LDA #$00
  STA $DC04
  LDA #$01
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$19
  STA $DC0E
  LDX #$00
  LDA #$11
  LDY #$22
wait: DEX
  BNE wait
  CLI
  STA $D401
  STY $D402
loop: JMP loop
irq: LDA #$33
  STA $D400
  JMP $EA31
"""
    src = QUIET + _vector("irq") + body
    machine, sid, _ = _boot(src)
    raise_at = _events(machine, "cia_irq")[0]
    assert _events(machine, "cia_irq_entry")[0] - raise_at > 1000
    assert [reg for _, reg, _ in sid.writes if reg < 3] == [1, 0, 2]


@pytest.mark.parametrize("stall", [False, True])
def test_irq_latency_is_the_finishing_instruction_then_seven_cycles(stall):
    blank = "" if stall else "  LDA #$0B\n  STA $D011\n"
    body = """
  LDA #$83
  STA $DC04
  LDA #$01
  STA $DC05
  LDA #$81
  STA $DC0D
  LDA #$11
  STA $DC0E
  CLI
loop: NOP
  NOP
  NOP
  NOP
  JMP loop
irq: LDA $DC0D
  JMP $EA81
"""
    src = QUIET + blank + _vector("irq") + body
    machine, _, _ = _boot(src)
    pairs = zip(_events(machine, "cia_irq")[2:], _events(machine, "cia_irq_entry")[2:])
    latency = collections.Counter(entry - at for at, entry in pairs)
    if stall:
        assert max(latency) > 7
    else:
        assert set(latency) == {6, 7, 8}


def _loop_gaps(src):
    _, sid, _ = _boot(src, seconds=0.1)
    cycles = [c for c, reg, _ in sid.writes if reg == 0x18][1:]
    return cycles, [b - a for a, b in zip(cycles, cycles[1:])]


def test_badlines_steal_forty_to_forty_three_cycles_twenty_five_times_a_frame():
    cycles, gaps = _loop_gaps(QUIET + LOOP)
    stolen = collections.Counter(g - 9 for g in gaps if g != 9)
    assert set(stolen) <= {40, 41, 42, 43}
    stalled = [c for c, g in zip(cycles[1:], gaps) if g != 9]
    per_frame = collections.Counter((c - 1) // FRAME for c in stalled)
    assert set(list(per_frame.values())[1:-1]) == {25}
    assert all(0x30 <= _line(c - 1)[0] <= 0xF7 for c in stalled)
    assert all(_line(c - 1)[0] & 7 == 3 for c in stalled)


def test_a_blank_screen_steals_nothing():
    _, gaps = _loop_gaps(QUIET + "  LDA #$0B\n  STA $D011\n" + LOOP)
    assert set(gaps) == {9}


@pytest.mark.parametrize("expand, height", [(0, 21), (1, 42)])
def test_a_sprite_steals_cycles_on_each_of_its_lines(expand, height):
    body = f"""
  LDA #$0B
  STA $D011
  LDA #$01
  STA $D015
  LDA #$80
  STA $D001
  LDA #${expand:02X}
  STA $D017
"""
    src = QUIET + body + LOOP
    cycles, gaps = _loop_gaps(src)
    frame = collections.defaultdict(set)
    for c, g in zip(cycles[1:], gaps):
        if g != 9:
            frame[(c - 1) // FRAME].add(_line(c - 1)[0])
    full = [lines for lines in frame.values() if len(lines) == height]
    assert full and all(min(lines) in (0x80, 0x81) for lines in full)


@pytest.mark.parametrize("line", [0x00, 0x37, 0x100])
def test_raster_irq_raises_at_the_start_of_its_line(line):
    body = f"""
  LDA #${0x1B | (line >> 1 & 0x80):02X}
  STA $D011
  LDA #${line & 0xFF:02X}
  STA $D012
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: INC $D019
  JMP $EA81
"""
    src = QUIET + _vector("irq") + body
    machine, _, _ = _boot(src)
    raises = _events(machine, "video_irq")[1:]
    assert len(raises) >= 10
    assert {_line(c) for c in raises} == {(line, 1 if line == 0 else 0)}
    assert {b - a for a, b in zip(raises, raises[1:])} == {FRAME}


@pytest.mark.parametrize("ack", ["INC", "DEC", "ASL", "LSR"])
def test_read_modify_write_acknowledges_the_raster_irq(ack):
    body = f"""
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: {ack} $D019
  JMP $EA81
"""
    src = QUIET + _vector("irq") + body
    machine, _, _ = _boot(src)
    assert len(_events(machine, "video_irq")) >= 10


def test_an_unacknowledged_raster_irq_is_raised_once():
    body = """
  LDA #$01
  STA $D01A
  CLI
loop: JMP loop
irq: JMP $EA81
"""
    src = QUIET + _vector("irq") + body
    machine, _, _ = _boot(src)
    assert len(_events(machine, "video_irq")) == 1


def test_brk_runs_through_the_irq_vector():
    body = """
  BRK
  NOP
  LDA #$22
  STA $D401
loop: JMP loop
brk: LDA #$11
  STA $D400
  JMP $EA81
"""
    src = _vector("brk") + body
    machine, sid, _ = _boot(src, seconds=0.01)
    assert _events(machine, "brk_entry") and len(_events(machine, "brk_entry")) == 1
    assert [(reg, value) for _, reg, value in sid.writes] == [(0, 0x11), (1, 0x22)]


def test_a_jam_stops_the_cpu():
    machine, sid, _ = _boot("  LDA #$01\n  STA $D400\n  .byte $02\n  STA $D401\n")
    assert machine.jammed and [reg for _, reg, _ in sid.writes] == [0]


def test_ane_uses_the_magic_constant():
    src = "  LDA #$00\n  LDX #$FF\n  .byte $8B, $3C\n  STA $D400\nloop: JMP loop\n"
    _, sid, _ = _boot(src, seconds=0.001)
    assert sid.writes[0][1:] == (0, 0x2C)  # (A | $EF) & X & imm


def test_io_page_routes_reads_and_writes():
    src = """
  LDA #$05
  STA $D800
  LDA $D800
  STA $C000
  LDA $D41B
  STA $C001
  LDA $D45B
  STA $C002
  LDA $DE00
  STA $C003
  LDA $DE01
  STA $C004
  LDA #$34
  STA $01
  LDA #$77
  STA $D400
  LDA $D400
  STA $C005
loop: JMP loop
"""
    machine, sid, _ = _boot(src, seconds=0.001)
    ram = machine.ram
    assert ram[0xC000] == 0x05 and machine.colour[0] == 0x05
    assert ram[0xC001] == ram[0xC002] == 0x1B  # $D45B mirrors the $D400 chip
    assert ram[0xC003] != ram[0xC004]  # unmapped I/O 1 reads the open bus
    assert ram[0xC005] == 0x77 and len(sid.writes) == 0  # $34: RAM at $D400


def test_an_extra_sid_at_io1_takes_its_slot():
    extra = Sid(value=0x40)
    machine = Machine(sids={mm.SID_BASE: Sid(), mm.IO1_BASE: extra})
    code, _ = assemble(
        "  LDA #$09\n  STA $DE04\n  LDA $DE1B\n  STA $C000\nl: JMP l\n", ORG
    )
    machine.ram[ORG : ORG + len(code)] = code
    machine.run(machine.reset(ORG) + 100)
    assert [w[1:] for w in extra.writes] == [(4, 9)]
    assert machine.ram[0xC000] == 0x5B


def test_events_reach_a_custom_sink():
    seen = []
    machine = Machine(on_event=lambda cycle, name: seen.append(name))
    code, _ = assemble("  BRK\n  NOP\nl: JMP l\n", ORG)
    machine.ram[ORG : ORG + len(code)] = code
    machine.ram[0x0314:0x0316] = (ORG + 2).to_bytes(2, "little")
    machine.run(machine.reset(ORG) + 100)
    assert seen[0] == "brk_entry" and machine.events == []


def test_the_cpu_extra_is_required(monkeypatch):
    real = builtins.__import__

    def no_jennings(name, *args, **kwargs):
        if name.startswith("jennings"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_jennings)
    with pytest.raises(ImportError, match=r"vagrant-moses\[cpu\]"):
        Machine()
