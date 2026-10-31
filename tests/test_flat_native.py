"""The native core against the Python engine on the same program."""

from __future__ import annotations

import pytest

from vagrant_moses import memmap as mm
from vagrant_moses.cpu import flat_mpu, run_to_rts

from asm import assemble

CODE = 0x1000
CODE_RANGE = range(CODE, CODE + 0x100)
STACK_RANGE = range(0x0100, 0x0200)

PROGRAM = """
start: LDA #$01
       STA $2000
       LDX #$05
loop:  LDA $2000,X
       ADC #$03
       STA $2000,X
       PHA
       DEX
       BNE loop
       LDA #$FF
       STA $D418
       LDA $D012
       STA $2010
       LDA #$00
       STA $FB
       LDA #$20
       STA $FC
       LDY #$02
       LDA ($FB),Y
       JSR sub
       LDA #$34
       STA $01
       LDA $E000
       LDA #$37
       STA $01
       LDA $01
       PLA
       PLA
       PLA
       PLA
       PLA
       RTS
sub:   INC $2011
       RTS
"""


def _run(native):
    code, _ = assemble(PROGRAM, CODE)
    ram = bytearray(0x10000)
    ram[0], ram[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    ram[CODE : CODE + len(code)] = code
    mpu, mem = flat_mpu(ram, native=native)
    ram = mem._subject.ram
    ram[0x01FF], ram[0x01FE], mpu.sp = 0x00, 0x00, 0xFD
    log = []
    watched = [*CODE_RANGE, *STACK_RANGE, *range(0x2000, 0x2020), 0xD012, 0xD418, 1]
    mem.subscribe_to_read(watched, lambda a: log.append(("r", a)))
    mem.subscribe_to_write(watched, lambda a, v: log.append(("w", a, v)))
    mpu.pc, mpu.processorCycles = CODE, 40 * mm.PAL.cycles_per_line
    states = []
    while mpu.pc != 0x0001 and len(states) < 500:
        mpu.step()
        states.append((mpu.pc, mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.processorCycles))
    assert mpu.pc == 0x0001 and mpu.sp == 0xFF
    return states, log, bytes(mem[0x2000:0x2020]), mem[1]


@pytest.fixture(scope="module")
def runs():
    return _run(False), _run(True)


def test_instruction_state_is_identical(runs):
    (py, _, _, _), (nat, _, _, _) = runs
    assert len(py) > 50
    assert py == nat


def test_final_memory_is_identical(runs):
    (_, _, py_mem, py_port), (_, _, nat_mem, nat_port) = runs
    assert py_mem == nat_mem and py_port == nat_port == 0x37


def test_data_subscriptions_match_in_order(runs):
    (_, py, _, _), (_, nat, _, _) = runs
    outside = lambda log: [e for e in log if e[1] not in CODE_RANGE]
    assert outside(py) == outside(nat)
    assert ("w", 0xD418, 0xFF) in nat and ("r", 0xD012) in nat


def test_native_fetches_never_reach_subscriptions(runs):
    (_, py, _, _), (_, nat, _, _) = runs
    assert any(e[0] == "r" and e[1] in CODE_RANGE for e in py)
    assert not [e for e in nat if e[1] in CODE_RANGE]


def test_stack_accesses_reach_subscriptions_on_both(runs):
    (_, py, _, _), (_, nat, _, _) = runs
    stack = lambda log: [e for e in log if e[1] in STACK_RANGE]
    assert stack(py) == stack(nat) and stack(nat)


def test_run_to_rts_matches_step_loop():
    code, _ = assemble(PROGRAM, CODE)
    results = []
    for native in (False, True):
        ram = bytearray(0x10000)
        ram[0], ram[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
        ram[CODE : CODE + len(code)] = code
        mpu, mem = flat_mpu(ram, native=native)
        run_to_rts(mpu, mem, CODE, 0, 10000)
        results.append(
            (
                mpu.a,
                mpu.x,
                mpu.y,
                mpu.sp,
                mpu.p,
                mpu.processorCycles,
                bytes(mem[0x2000:0x2020]),
            )
        )
    assert results[0] == results[1]
