"""The native machine's SMC recorder: writers, roles, timing, mapping and indices."""

from __future__ import annotations

import pytest

from vagrant_moses import memmap as mm
from vagrant_moses import smc
from vagrant_moses.machine import Machine

from asm import assemble

pytest.importorskip("jennings")
pytestmark = pytest.mark.skipif(not smc.AVAILABLE, reason="no native SMC recorder")

ORG = 0x1000


def _armed(src, begin=True, image=None, **kwargs):
    m = Machine(backend="native", **kwargs)
    code, labels = assemble(src, ORG)
    m.ram[ORG : ORG + len(code)] = code
    start = m.reset(ORG)
    rec = smc.Recorder(m, bytes(m.ram) if image is None else image)
    if begin:
        rec.begin()
    return m, rec, start, labels


def _keys(rec):
    return {e[:5]: e.first for e in rec.events()}


def test_opcode_table_encodes_length_pointer_and_return():
    t = smc.opcode_table()
    assert t[0x6C] == 3 | 1 << 2 and t[0xB1] == 2 | 2 << 2 and t[0xA1] == 2 | 3 << 2
    assert t[0x60] == 1 | 1 << 4 and t[0x40] == 1 | 2 << 4 and t[0x8D] == 3


def test_a_store_owns_a_byte_until_it_changes_it_again():
    src = """
        LDA #$42
        STA tv
        .byte $A9
    tv: .byte $00
    idle: JMP idle
    """
    m, rec, start, labels = _armed(src)
    passes = []

    def trap(mc):
        passes.append(mc.cycle)
        if len(passes) == 2:
            mc.stop = True
            return False
        rec.next_call()
        mc.mpu.pc = ORG
        return True

    m.set_trap(trap, {labels["idle"]})
    m.run(start + 10_000)
    assert len(passes) == 2 and rec.call == 1
    keys = _keys(rec)
    # the second pass stores the same value: the byte keeps its call-0 writer
    assert keys == {
        ("cpu", 0x8D, "operand", "same_call_earlier", "flat"): 0,
        ("cpu", 0x8D, "operand", "earlier_call", "flat"): 1,
    }
    assert rec.unclassified() == (0, [])


def test_a_store_into_its_own_operand_and_the_pre_lift_phase():
    src = """
        LDA #$11
        STA wv
        INC sv
    n:  NOP
    s:  .byte $8D
    sv: .byte $00, $C0
        .byte $A9
    wv: .byte $00
    idle: JMP idle
    """
    m, rec, start, labels = _armed(src, begin=False)
    m.set_trap(lambda mc: setattr(mc, "stop", True), {labels["n"]})
    m.run(start + 1000)
    rec.begin()
    m.set_trap(lambda mc: setattr(mc, "stop", True), {labels["idle"]})
    m.run(m.cycle + 1000)
    keys = _keys(rec)
    assert ("cpu", 0xEE, "operand", "pre_lift", "flat") in keys
    assert ("cpu", 0x8D, "operand", "pre_lift", "flat") in keys


def test_jsr_frames_and_rts_return_bytes():
    src = """
        JSR sub
    idle: JMP idle
    sub: RTS
    """
    m, rec, start, labels = _armed(src)
    m.set_trap(lambda mc: setattr(mc, "stop", True), {labels["idle"]})
    m.run(start + 1000)
    assert _keys(rec) == {("cpu", 0x20, "ret_addr", "same_call_earlier", "flat"): 0}


def test_port_writes_bank_cells_and_the_host_port_reset():
    src = """
        LDA #$36
        STA $01
    idle: JMP idle
    """
    m, rec, start, labels = _armed(src)
    m.set_trap(lambda mc: setattr(mc, "stop", True), {labels["idle"]})
    m.run(start + 1000)
    assert _keys(rec) == {
        ("cpu_port", 0x85, "bank_reg", "same_call_earlier", "flat"): 0
    }
    rec.host_write(mm.CPU_PORT, 0x37)
    rec.next_call()
    m.mpu.pc = labels["idle"] + 3
    m.ram[labels["idle"] + 3 : labels["idle"] + 6] = bytes.fromhex("4c0010")
    m.stop = False
    m.run(m.cycle + 30)
    assert len(rec.events()) == 1
    rec.host_write(mm.CPU_PORT, 0x36)
    m.run(m.cycle + 30)
    assert ("cpu_port", None, "bank_reg", "pre_lift", "flat") in _keys(rec)


def test_ram_written_under_the_kernal_and_run_banked_out():
    src = """
        LDA #$60
        STA $E000
        LDA #$35
        STA $01
        JSR $E000
        LDA #$37
        STA $01
        JSR $E000
    idle: JMP idle
    """
    m, rec, start, labels = _armed(src)
    m.set_trap(lambda mc: setattr(mc, "stop", True), {labels["idle"]})
    m.run(start + 1000)
    keys = _keys(rec)
    assert ("cpu", 0x8D, "opcode", "same_call_earlier", "ram_under_rom") in keys
    assert ("cpu_port", 0x85, "opcode", "same_call_earlier", "rom_vs_ram") in keys


def test_an_interrupt_frame_is_async_and_indexed_by_frame():
    src = """
        SEI
        LDA #<hdlr
        STA $FFFE
        LDA #>hdlr
        STA $FFFF
        LDA #$35
        STA $01
        CLI
    idle: JMP idle
    hdlr: LDA $DC0D
        RTI
    """
    m, rec, start, _ = _armed(src)
    for reg, v in ((mm.CIA1_TIMER_A_LO, 0x00), (mm.CIA1_TIMER_A_HI, 0x10)):
        m.io_write(reg, v)
    m.io_write(mm.CIA1_ICR, 0x81)
    m.io_write(mm.CIA1_CRA, 0x11)
    rec.frame_clock(0x400)
    m.run(start + 0x1001 * 5)
    keys = _keys(rec)
    first = keys[("irq_push", None, "ret_addr", "async", "flat")]
    assert (
        first
        == keys[("cpu", 0x8D, "hw_vector", "same_call_earlier", "ram_under_rom")]
        >= 3
    )
    with pytest.raises(ValueError):
        rec.frame_clock(0)


def test_dma_writes_code_the_cpu_then_runs():
    src = """
        LDX #$00
    reg: LDA tab,X
        STA $DF02,X
        INX
        CPX #$07
        BNE reg
        LDA #$90
        STA $DF01
        LDA #<tgt
        STA $DF02
        LDA #>tgt
        STA $DF03
        LDA #$00
        STA $DF04
        LDA #$01
        STA $DF07
        LDA #$91
        STA $DF01
    tgt: BRK
    idle: JMP idle
    tab: .byte $00, $20, $00, $00, $00, $01, $00
    """
    m, rec, start, labels = _armed(src, reu=128)
    m.ram[0x2000] = 0xEA
    rec = smc.Recorder(m, bytes(m.ram))
    rec.begin()
    m.set_trap(lambda mc: setattr(mc, "stop", True), {labels["idle"]})
    m.run(start + 5000)
    assert m.mpu.pc == labels["idle"]
    assert _keys(rec) == {("reu_dma", None, "opcode", "same_call_earlier", "flat"): 0}


def test_deadline_unowned_bytes_finish_and_disarm():
    src = "loop: JMP loop\n"
    m = Machine(backend="native")
    code, _ = assemble(src, ORG)
    m.ram[ORG : ORG + 3] = code
    start = m.reset(ORG)
    rec = smc.Recorder(m, bytes(0x10000))
    rec.begin()
    assert rec.deadline is None and not rec.timed_out
    rec.deadline = start + 100
    m.run(start + 100_000)
    assert rec.timed_out and m.cycle < start + 120 and rec.deadline == start + 100
    count, sample = rec.unclassified()
    # the JMP's high operand byte ($10) differs from the image; its low byte ($00) does not
    assert count == 2 and sample == [
        ("unowned", ORG, "opcode", ORG),
        ("unowned", ORG + 2, "operand", ORG),
    ]
    rec.deadline = None
    assert not rec.timed_out
    rec.finish()
    assert m.smc_armed
    rec.disarm()
    assert not m.smc_armed
    with pytest.raises(RuntimeError):
        rec.events()


def test_machines_take_rom_images():
    kernal = bytes([0x60]) * 0x2000
    for backend in ("native", "python"):
        m = Machine(backend=backend, roms={mm.KERNAL_BASE: kernal})
        m.reset(ORG)
        assert m[0xE000] == 0x60 and m[0xA000] == Machine(backend=backend)[0xA000]
