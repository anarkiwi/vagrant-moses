"""ADC, SBC, ARR, RRA and ISC in decimal mode on both machines, against Bruce Clark's tutorial.

A 6502 loop runs each instruction on every operand with the decimal flag set (invalid BCD
included) for one accumulator and carry; the test covers every accumulator, operand and carry,
and compares the result, N, V, Z, C and the operand left in memory.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jennings")

import decimal_ref as ref  # pylint: disable=wrong-import-position
from asm import assemble  # pylint: disable=wrong-import-position
from vagrant_moses import machine as vm  # pylint: disable=wrong-import-position

BACKENDS = ["python"] + ["native"] * vm.NATIVE
ORG, A_IN, OPERAND = 0x1000, 0xF0, 0xF1
RESULT, FLAGS, MEMORY = 0x2000, 0x2100, 0x2200
# per iteration: 3 + 2 + 5 + 3 + 5 + 4 + 5 + 3 + 5 + 2 + 3 cycles at most, plus ARR's patch
ITERATION = 48
OPS = {
    "ADC": ("ADC $F1", ref.adc),
    "SBC": ("SBC $F1", ref.sbc),
    "RRA": ("RRA $F1", ref.rra),
    "ISC": ("ISC $F1", ref.isc),
    "ARR": ("ARR #$00", ref.arr),
}


def program(name, carry, patched=ORG):
    patch = f"  STX ${patched:04X}\n" if name == "ARR" else ""
    body = OPS[name][0]
    label = "arr:" if name == "ARR" else ""
    return assemble(
        f"""
        SED
        LDX #$00
next:   STX ${OPERAND:02X}
{patch}        LDA ${A_IN:02X}
        {"SEC" if carry else "CLC"}
{label}  {body}
        PHP
        STA ${RESULT:04X},X
        PLA
        STA ${FLAGS:04X},X
        LDA ${OPERAND:02X}
        STA ${MEMORY:04X},X
        INX
        BNE next
done:   JMP done
""",
        ORG,
    )


def run(backend, name, a, carry):
    code, labels = program(name, carry)
    if name == "ARR":
        code, labels = program(name, carry, labels["arr"] + 1)
    machine = vm.Machine(False, backend=backend)
    start = machine.reset(ORG)
    machine[0], machine[1] = 0x2F, 0x30
    for i, byte in enumerate(code):
        machine.ram[ORG + i] = byte
    machine.ram[A_IN] = a
    machine.run(start + 256 * ITERATION + 16)
    assert machine.mpu.pc == labels["done"]
    ram = machine.ram
    return (
        bytes(ram[RESULT : RESULT + 256]),
        bytes(ram[FLAGS : FLAGS + 256]),
        bytes(ram[MEMORY : MEMORY + 256]),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("block", range(16))
def test_decimal_mode_matches_the_reference(backend, name, block):
    fn = OPS[name][1]
    for a in range(block * 16, block * 16 + 16):
        for carry in (0, 1):
            result, flags, memory = run(backend, name, a, carry)
            for m in range(256):
                want = fn(a, m, carry, 1)
                got = (
                    result[m],
                    flags[m] >> 7 & 1,
                    flags[m] >> 6 & 1,
                    flags[m] >> 1 & 1,
                    flags[m] & 1,
                )
                assert got == tuple(want[:5]), (name, a, m, carry)
                assert flags[m] & 8, "the program runs with D set"
                if name in ("RRA", "ISC"):
                    assert memory[m] == want[5], (name, a, m, carry)
