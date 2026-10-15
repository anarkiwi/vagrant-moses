"""A two-pass label assembler over jennings' one-statement ``Assembler``, for tests.

Lines are ``[label:] [STATEMENT | .byte n]``; lowercase names are labels, usable
as operands and as ``#<name``/``#>name``. Mnemonics and registers are uppercase.
"""

from __future__ import annotations

import re

from jennings.assembler import Assembler
from jennings.devices.mpu6502 import MPU

_ASM = Assembler(MPU())
_NAME = re.compile(r"([<>]?)\b([a-z_][a-z0-9_]*)\b")


def _pass(lines, org, labels):
    pc, out, found = org, bytearray(), {}
    for line in lines:
        label, _, stmt = line.rpartition(":") if ":" in line else ("", "", line)
        if label:
            found[label.strip()] = pc
        stmt = stmt.strip()
        if not stmt:
            continue
        if stmt.startswith(".byte"):
            code = [int(v.strip().lstrip("$"), 16) for v in stmt[5:].split(",")]
        else:

            def value(m, here=pc):
                addr = labels.get(m.group(2), here)
                if m.group(1):
                    return "$%02X" % (addr & 0xFF if m.group(1) == "<" else addr >> 8)
                return "$%04X" % addr

            code = _ASM.assemble(_NAME.sub(value, stmt), pc)
        out += bytes(code)
        pc += len(code)
    return out, found


def assemble(src: str, org: int):
    """``(code, labels)`` of ``src`` assembled at ``org``."""
    lines = [line.split(";")[0] for line in src.splitlines()]
    _, labels = _pass(lines, org, {})
    return _pass(lines, org, labels)
