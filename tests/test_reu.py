"""The 17xx REU: register semantics, DMA cycle gating, and generated programs against VICE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vagrant_moses import machine as vm
from vagrant_moses import reu as rm
from vagrant_moses.machine import Machine
from vagrant_moses.reu import Reu

pytest.importorskip("jennings")

import reugen  # pylint: disable=wrong-import-position
from asm import assemble  # pylint: disable=wrong-import-position

FIXTURE = Path(__file__).parent / "data" / "reu_vice.json"
BACKENDS = ["python"] + (["native"] if vm.NATIVE else [])


class FlatBus:
    """A 64 KiB C64 side for the engine alone, logging every access."""

    def __init__(self):
        self.mem = bytearray(range(256)) * 256
        self.log = []

    def dma_read(self, addr):
        self.log.append(("r", addr))
        return self.mem[addr]

    def dma_write(self, addr, value):
        self.log.append(("w", addr))
        self.mem[addr] = value


def _reu(size_kb=512):
    bus, lines, requests = FlatBus(), [], []
    reu = Reu(size_kb, bus, lines.append, lambda: requests.append(True))
    return reu, bus, lines, requests


def _program(reu, c64, raddr, length, command, control=0x3F, int_mask=0x1F):
    regs = {2: c64 & 0xFF, 3: c64 >> 8, 4: raddr & 0xFF, 5: raddr >> 8 & 0xFF}
    regs |= {6: raddr >> 16, 7: length & 0xFF, 8: length >> 8, 9: int_mask}
    regs |= {10: control}
    for reg, value in regs.items():
        reu.write(0xDF00 + reg, value)
    reu.write(0xDF01, command)


def _cycles(reu, ba=lambda t: False):
    """Cycles the CPU is held, BA following ``ba(t)``."""
    t = 0
    while not reu.step(ba(t)):
        t += 1
    return t


@pytest.mark.parametrize("size_kb", rm.SIZES)
def test_geometry_and_reset_registers(size_kb):
    reu, *_ = _reu(size_kb)
    wrap, dram, mask, unused, status = rm.geometry(size_kb)
    assert len(reu.ram) == size_kb << 10
    assert wrap == (0x20000 if size_kb == 128 else 0x80000)
    assert dram == (wrap if size_kb <= 512 else size_kb << 10)
    assert mask == (0x7FFFF if size_kb <= 512 else (size_kb << 10) - 1)
    assert unused == (0xF8 if size_kb <= 512 else 0)
    assert reu.regs == bytes((status, 0x10, 0, 0, 0, 0, unused, 0xFF, 0xFF, 0x1F, 0x3F))
    with pytest.raises(ValueError):
        rm.geometry(64)


def test_register_reads_mirror_and_clear():
    reu, _, lines, _ = _reu(256)
    reu.write(0xDF06, 0xFF)
    reu.write(0xDF09, 0x00)
    reu.write(0xDF0A, 0x00)
    reu.write(0xDF00, 0x00)
    reu.write(0xDF0C, 0x12)
    assert reu.read(0xDF06) == 0xFF and reu.peek(6) == 0xFF
    assert reu.read(0xDF09) == 0x1F and reu.read(0xDF0A) == 0x3F
    assert reu.read(0xDF0B) == reu.read(0xDF1F) == 0xFF
    reu.write(0xDF27, 0x42)
    assert reu.read(0xDFE7) == 0x42
    reu.status |= rm.END_OF_BLOCK
    reu.write(0xDF09, 0xC0)
    assert lines == [True] and reu.peek(0) == 0xD0
    assert reu.read(0xDF20) == 0xD0
    assert lines == [True, False] and reu.read(0xDF00) == 0x10


def test_ff00_trigger_arms_and_fires_once():
    reu, _, _, requests = _reu()
    reu.ff00()
    _program(reu, 0x1000, 0, 4, 0x80)
    assert reu.armed and not requests
    reu.write(0xDF01, 0x01)
    reu.ff00()
    reu.ff00()
    assert requests == [True] and not reu.armed
    assert _cycles(reu) == 4 and reu.peek(1) == 0x11


def test_dma_holds_the_bus_one_cycle_per_byte_and_two_per_swap():
    reu, bus, _, requests = _reu()
    _program(reu, 0x1000, 0x10, 5, 0x90)
    assert requests and _cycles(reu) == 5
    assert bytes(reu.ram[0x10:0x15]) == bus.mem[0x1000:0x1005]
    assert reu.peek(0) & rm.END_OF_BLOCK and reu.peek(7) == 1
    assert (reu.peek(2), reu.peek(3), reu.peek(4)) == (0x05, 0x10, 0x15)
    _program(reu, 0x2000, 0x10, 5, 0x92)
    assert _cycles(reu) == 10
    assert [k for k, _ in bus.log[-10:]] == ["r", "w"] * 5
    assert reu.read(0xDF00) is not None and not reu.busy


def test_busy_reu_ignores_the_cpu():
    reu, *_ = _reu()
    _program(reu, 0x1000, 0, 3, 0x90)
    assert not reu.step(False) and reu.busy
    assert reu.read(0xDF00) is None
    reu.write(0xDF07, 0x55)
    assert _cycles(reu) == 2 and reu.length == 1


@pytest.mark.parametrize("k,ends", [(0, False), (2, True), (3, True)])
def test_verify_stops_after_a_mismatch(k, ends):
    reu, bus, _, _ = _reu()
    reu.ram[0:4] = bus.mem[0x1000:0x1004]
    reu.ram[k] ^= 0xFF
    _program(reu, 0x1000, 0, 4, 0x93)
    held = _cycles(reu)
    left = 4 - k - 1
    assert held == k + 1 + (1 if left else 0)
    status = reu.read(0xDF00)
    assert status & rm.VERIFY_ERROR
    assert bool(status & rm.END_OF_BLOCK) == ends
    assert reu.length == max(left, 1)


def test_autoload_and_fixed_addresses_and_half_autoload():
    reu, bus, _, _ = _reu()
    _program(reu, 0x1000, 0x20, 3, 0xB0, control=0x80)
    _cycles(reu)
    assert (reu.c64, reu.base, reu.length) == (0x1000, 0x20, 3)
    assert bytes(reu.ram[0x20:0x23]) == bytes([bus.mem[0x1000]] * 3)
    reu.write(0xDF01, 0x91)
    _cycles(reu)
    assert (reu.c64, reu.base, reu.length) == (0x1000, 0x23, 1)
    reu.write(0xDF07, 0x09)
    assert reu.length == 0x0009
    reu.write(0xDF05, 0x00)
    assert reu.base == 0x0020


def test_reu_address_wraps_and_unbacked_reads_float():
    reu, bus, _, _ = _reu(128)
    reu.ram[0x1FFFF], reu.ram[0] = 0xAA, 0xBB
    _program(reu, 0x3000, 0x1FFFF, 2, 0x91)
    _cycles(reu)
    assert (reu.base, reu.bank & 7) == (0x0001, 0)
    assert bus.mem[0x3000:0x3002] == b"\xaa\xbb"
    small, bus, _, _ = _reu(256)
    small.ram[0x3FFFF] = 0x5A
    _program(small, 0x3000, 0x3FFFF, 3, 0x91)
    _cycles(small)
    assert bus.mem[0x3000:0x3003] == bytes((0x5A, 0x5A, 0x5A))
    big, _, _, _ = _reu(1024)
    _program(big, 0x3000, 0xF7FFFF, 2, 0x90)
    _cycles(big)
    assert big.raddr == 0xF00001 and big.bank == 0x00


def test_reads_wait_for_ba_and_writes_take_one_low_cycle():
    low = set(range(3, 8))
    reu, *_ = _reu()
    _program(reu, 0x1000, 0, 6, 0x90)
    assert _cycles(reu, low.__contains__) == 6 + len(low)
    reu, bus, _, _ = _reu()
    _program(reu, 0x1000, 0, 6, 0x91)
    assert _cycles(reu, low.__contains__) == 6 + len(low) - 1
    assert len(bus.log) == 6


def test_write_ending_into_ba_low_takes_a_tail_cycle():
    low = set(range(3, 6))
    reu, *_ = _reu()
    _program(reu, 0x1000, 0, 4, 0x91)
    assert _cycles(reu, low.__contains__) == 4 + 3
    reu, *_ = _reu()
    _program(reu, 0x1000, 0, 3, 0x91)
    assert _cycles(reu, low.__contains__) == 3


def test_interrupt_masks_drive_the_line():
    reu, bus, lines, _ = _reu()
    _program(reu, 0x1000, 0, 2, 0x90, int_mask=0x40)
    _cycles(reu)
    assert not lines and reu.peek(0) & 0xC0 == 0x40
    reu.write(0xDF09, 0xC0)
    assert lines == [True]
    reu.read(0xDF00)
    bus.mem[0x1000] ^= 0xFF
    _program(reu, 0x1000, 0, 2, 0x93, int_mask=0xA0)
    _cycles(reu)
    assert lines == [True, False, True] and reu.peek(0) & 0xE0 == 0xE0


@pytest.mark.parametrize("backend", BACKENDS)
def test_machine_halts_cpu_and_maps_dma_through_the_port(backend):
    src = """
        LDA #$00
        STA $DF02
        LDA #$C0
        STA $DF03
        LDA #$00
        STA $DF04
        STA $DF05
        STA $DF06
        LDA #$02
        STA $DF07
        LDA #$00
        STA $DF08
        LDA #$81
        STA $DF01
        LDA #$34
        STA $01
        INC $FF00
        LDA #$37
        STA $01
        LDA $DF00
        STA $2000
        done: JMP done
    """
    machine = Machine(reu=128, backend=backend)
    assert machine.reu.ram[:4] == bytearray(4)
    machine.reu.ram[0:2] = b"\x12\x34"
    code, labels = assemble(src, 0x1000)
    machine.ram[0x1000 : 0x1000 + len(code)] = code
    machine.ram[0xFF00] = 0x77
    machine.reset(0x1000)
    machine.run(machine.cycle + 400)
    assert machine.mpu.pc == labels["done"]
    assert machine.ram[0xC000:0xC002] == b"\x12\x34"
    assert machine.ram[0xFF00] == 0x77 and machine.ram[0x2000] == 0x40
    assert machine.reu.regs[1] == 0x11 and not machine.reu.armed
    assert Machine(backend=backend).reu is None


def test_generator_covers_every_cell_and_feature():
    programs = reugen.corpus(48) + reugen.corpus(48, display=True)
    cov = reugen.coverage(programs)
    assert len(cov["cells"]) == 4 * 4 * 2 * 2
    assert all(cov["cells"].values())
    seen = {name for (name, _), count in cov["features"].items() if count}
    assert seen == set(reugen.FEATURES)
    assert {program.size_kb for program in programs} == set(rm.SIZES)


@pytest.fixture(scope="module", name="fixture")
def _fixture():
    if not FIXTURE.exists():
        pytest.skip("no VICE digests recorded")
    doc = json.loads(FIXTURE.read_text())
    corpus = doc["corpus"]
    programs = [
        p
        for display in corpus["display"]
        for p in reugen.corpus(corpus["count"], corpus["seed"], corpus["ops"], display)
    ]
    return {p.name: (p, doc["programs"][p.name]) for p in programs}


def _names():
    if not FIXTURE.exists():
        return []
    return sorted(json.loads(FIXTURE.read_text())["programs"])


@pytest.mark.parametrize("name", _names())
def test_programs_match_vice(fixture, name):
    program, want = fixture[name]
    assert program.key == want["key"], "generator changed: rerun tools/vice_reu.py"
    digests = {backend: reugen.run(program, backend).digest() for backend in BACKENDS}
    for backend, got in digests.items():
        assert got == {k: want[k] for k in got}, backend


def test_observation_normalises_port_and_colour_nibbles():
    ram = bytearray(0x10000)
    ram[0:3] = b"\x2f\x37\x99"
    obs = reugen.Observation.of(ram, b"\xf5" * 0x400, b"\x01\x02")
    assert obs.ram[:3].tolist() == [0, 0, 0x99]
    assert set(obs.colour.tolist()) == {5}
    assert (
        obs.digest() != reugen.Observation.of(bytes(0x10000), obs.colour, b"").digest()
    )
    assert np.array_equal(obs.reu, [1, 2])
