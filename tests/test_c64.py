"""Behavioral tests for the C64 machine model.

Each test asserts a machine invariant or a helper contract -- the power-on
pattern, the vector-dispatch priority rule, and the shape of the ROM-free IRQ
dispatch path (frame pushed, entry pc, conflict refusal).
"""

from __future__ import annotations

import pytest

import vagrant_moses
from vagrant_moses import c64


class VM:
    """The host the install helpers target: a writable C64 main-memory buffer."""

    def __init__(self):
        self.mem = bytearray(0x10000)


# --- package surface ----------------------------------------------------------


def test_package_exposes_api():
    for name in vagrant_moses.__all__:
        assert hasattr(vagrant_moses, name)
    assert callable(vagrant_moses.poweron_ram)


def test_module_is_reachable_from_the_package():
    assert vagrant_moses.c64 is c64
    assert vagrant_moses.poweron_ram is c64.poweron_ram
    assert vagrant_moses.IRQ_DISPATCH == c64.IRQ_DISPATCH


# --- power-on RAM -------------------------------------------------------------


def test_poweron_ram_size_and_type():
    ram = c64.poweron_ram()
    assert isinstance(ram, bytes)
    assert len(ram) == 0x10000


def test_poweron_ram_banks_alternate_base():
    ram = c64.poweron_ram()
    assert ram[0x0000] == 0x00
    assert ram[0x4000] == 0xFF
    assert ram[0x8000] == 0x00
    assert ram[0xC000] == 0xFF


def test_poweron_ram_stripe_pattern():
    ram = c64.poweron_ram()
    for bank in range(4):
        off = bank * 0x4000
        base = 0xFF if bank & 1 else 0x00
        stripe = base ^ 0xFF
        assert ram[off : off + 2] == bytes([base]) * 2
        assert ram[off + 2 : off + 6] == bytes([stripe]) * 4
        assert ram[off + 6 : off + 10] == bytes([base]) * 4
        assert ram[off + 10 : off + 14] == bytes([stripe]) * 4


# --- vectors ------------------------------------------------------------------


def test_vector_pairs_are_contiguous_lo_hi():
    for pair in (c64.IRQ_VEC, c64.NMI_VEC, c64.HW_IRQ_VEC, c64.HW_NMI_VEC):
        assert pair[1] == pair[0] + 1


def test_read_vector_little_endian():
    mem = bytearray(0x10000)
    mem[0x10] = 0xCD
    mem[0x11] = 0xAB
    assert c64.read_vector(mem, 0x10) == 0xABCD


def test_read_vector_wraps_word_boundary():
    mem = bytearray(0x10000)
    mem[0xFFFF] = 0x12
    mem[0x0000] = 0x34
    assert c64.read_vector(mem, 0xFFFF) == 0x3412


def test_installed_vector_written_cinv_wins():
    mem = bytearray(0x10000)
    assert c64.installed_vector(mem, {0x0314}, (0, 0)) == (0x0314, True)


def test_installed_vector_written_high_byte_counts():
    mem = bytearray(0x10000)
    assert c64.installed_vector(mem, {0x0315}, (0, 0)) == (0x0314, True)


def test_installed_vector_written_hardware():
    mem = bytearray(0x10000)
    assert c64.installed_vector(mem, {0xFFFE}, (0, 0)) == (0xFFFE, False)


def test_installed_vector_written_nmi():
    mem = bytearray(0x10000)
    assert c64.installed_vector(mem, {0x0318}, (0, 0)) == (0x0318, False)


def test_installed_vector_priority_cinv_first():
    mem = bytearray(0x10000)
    assert c64.installed_vector(mem, {0x0318, 0x0314, 0xFFFE}, (0, 0)) == (0x0314, True)


def test_installed_vector_lifted_from_image():
    mem = bytearray(0x10000)
    mem[0x0314] = 0x34
    mem[0x0315] = 0xAB
    assert c64.installed_vector(mem, set(), (0x0314, 0x0400)) == (0x0314, True)


def test_installed_vector_none_when_absent():
    mem = bytearray(0x10000)
    assert c64.installed_vector(mem, set(), (0x8000, 0x9000)) is None


def test_installed_handler_maps_to_target():
    mem = bytearray(0x10000)
    mem[0x0314] = 0x81
    mem[0x0315] = 0xEA
    assert c64.installed_handler(mem, {0x0314}, (0, 0)) == (0xEA81, True)


def test_installed_handler_none_when_absent():
    mem = bytearray(0x10000)
    assert c64.installed_handler(mem, set(), (0x8000, 0x9000)) is None


# --- dispatch stubs -----------------------------------------------------------


def test_irq_stubs_always_has_dispatch_and_return():
    for kernal in (False, True):
        stubs = dict(c64.irq_stubs(0xC000, kernal))
        assert c64.IRQ_DISPATCH in stubs
        assert c64.IRQ_RETURN in stubs
        assert stubs[c64.IRQ_RETURN] == b"\x60"


def test_irq_stubs_pushes_the_irq_frame():
    code = dict(c64.irq_stubs(0xC000, False))[c64.IRQ_DISPATCH]
    assert code[0] == 0x78  # CLI
    pushed = []
    i = 1
    while i < len(code) - 3:  # stop before the trailing JMP
        assert code[i] == 0xA9  # LDA #imm
        pushed.append(code[i + 1])
        assert code[i + 2] == 0x48  # PHA
        i += 3
    assert pushed == [c64.IRQ_RETURN >> 8, c64.IRQ_RETURN & 0xFF, 0x20]


def test_irq_stubs_tail_selects_target():
    vec = 0x1234
    bare = dict(c64.irq_stubs(vec, False))[c64.IRQ_DISPATCH]
    assert bare[-3:] == bytes((0x6C, vec & 0xFF, vec >> 8))  # JMP (vec)
    kernal = dict(c64.irq_stubs(vec, True))[c64.IRQ_DISPATCH]
    assert kernal[-3:] == bytes(
        (0x4C, c64.KERNAL_IRQ & 0xFF, c64.KERNAL_IRQ >> 8)
    )  # JMP $FF48


def test_irq_stubs_kernal_only_addresses():
    kernal = {addr for addr, _ in c64.irq_stubs(0xC000, True)}
    bare = {addr for addr, _ in c64.irq_stubs(0xC000, False)}
    for extra in (c64.KERNAL_IRQ, 0xEA31, 0xEA81, 0xFEBC):
        assert extra in kernal
        assert extra not in bare
    for base in (c64.IRQ_DISPATCH, c64.IRQ_RETURN):
        assert base in kernal and base in bare


def test_kernal_prologue_jumps_to_vector():
    vec = 0x7890
    pro = dict(c64.irq_stubs(vec, True))[c64.KERNAL_IRQ]
    assert pro[-3:] == bytes((0x6C, vec & 0xFF, vec >> 8))  # JMP (vec)
    assert 0x48 in pro[:-3]  # PHA: A is saved
    assert 0x8A in pro[:-3]  # PHX: X is saved


# --- installation -------------------------------------------------------------


def test_install_kernal_irq_stubs_writes_epilogue():
    vm = VM()
    c64.install_kernal_irq_stubs(vm)
    assert vm.mem[0xEA31:0xEA81] == b"\xea" * (0xEA81 - 0xEA31)
    epilogue = vm.mem[0xEA81 : 0xEA81 + 6]
    assert epilogue[-1] == 0x40  # RTI
    assert sorted(epilogue[:-1]) == [0x68, 0x68, 0x68, 0xA8, 0xAA]
    assert vm.mem[0xFEBC : 0xFEBC + 6] == epilogue


def test_install_irq_entry_writes_every_stub_and_returns_pc():
    vec = 0x4567
    for kernal in (False, True):
        vm = VM()
        assert c64.install_irq_entry(vm, vec, kernal) == c64.IRQ_DISPATCH
        for addr, code in c64.irq_stubs(vec, kernal):
            assert vm.mem[addr : addr + len(code)] == code


def test_install_irq_entry_refuses_covered_dispatch_stub():
    for kernal in (False, True):
        vm = VM()
        with pytest.raises(ValueError):
            c64.install_irq_entry(vm, 0x4567, kernal, img=(0xFF00, 0x10000))


def test_install_irq_entry_refuses_covered_kernal_stub():
    vm = VM()
    with pytest.raises(ValueError):
        c64.install_irq_entry(vm, 0x4567, True, img=(0xEA31, 0xEAE0))


def test_install_irq_entry_succeeds_when_image_is_clear():
    vm = VM()
    c64.install_irq_entry(vm, 0x4567, True, img=(0x8000, 0x9000))
    assert vm.mem[0xEA31] == 0xEA
