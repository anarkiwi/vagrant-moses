"""The banked bus, the memory map and the instruction-stepped CPU helpers."""

from __future__ import annotations

import builtins

import pytest

from vagrant_moses import c64
from vagrant_moses import memmap as mm
from vagrant_moses.bus import Bus, banking, port_read
from vagrant_moses.cpu import access_cycle, flat_mpu, run_to_rts
from vagrant_moses.roms import kernal_stub

CODE = 0x1000


def _shadowed():
    ram = bytearray(0x10000)
    ram[0xA000], ram[0xD000], ram[0xE000] = 0xAB, 0xEF, 0xCD
    return ram


def test_power_on_port_reads_back_as_written():
    assert port_read(mm.DDR_DEFAULT, mm.BANK_DEFAULT) == mm.BANK_DEFAULT
    assert port_read(0x00, 0x00) == 0x1F  # all inputs: bits 0-4 pulled up
    assert port_read(0xFF, 0xC8) == 0xC8  # all outputs read as written


@pytest.mark.parametrize("data", range(8))
def test_rom_space_follows_the_port_as_the_pla_maps_it(data):
    bus = Bus(_shadowed())
    bus[0], bus[1] = mm.DDR_DEFAULT, data
    loram, hiram, charen = data & 1, data & 2, data & 4
    assert bus[0xA000] == (0 if loram and hiram else 0xAB)
    assert bus[0xE000] == (kernal_stub()[0] if hiram else 0xCD)
    assert bus[0xFFFE] | bus[0xFFFF] << 8 == (0xFF48 if hiram else 0)
    io = charen and (loram or hiram)
    chars = not charen and (loram or hiram)
    assert bus[0xD000] == (0xEF if io or not chars else 0)
    assert bus.io == bool(io) and banking(data)[3] == bool(io)
    assert bus[1] == data | 0x10 and bus[0] == mm.DDR_DEFAULT


def test_input_port_bits_float_to_all_roms_in():
    bus = Bus(_shadowed())
    bus[0], bus[1] = 0x00, 0x00
    assert bus.bits == 7 and bus[0xE000] != 0xCD


def test_writes_reach_the_ram_under_a_rom():
    bus = Bus(bytearray(0x10000))
    bus[0], bus[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    bus[0xE000] = 0x5A
    assert bus[0xE000] == 0xEA and bus.ram[0xE000] == 0x5A
    bus[1] = 0x35
    assert bus[0xE000] == 0x5A


def test_direct_ram_changes_are_seen_through_the_bus():
    bus = Bus()
    assert bus.ram == c64.poweron_ram()
    bus.ram[0x2000] = 0x42
    assert bus[0x2000] == 0x42


def test_real_roms_override_the_rom_less_set():
    bus = Bus(bytearray(0x10000), {mm.KERNAL_BASE: b"\x11" * 0x2000})
    bus[0], bus[1] = mm.DDR_DEFAULT, 0x33
    assert bus[0xE000] == 0x11 and bus[0xA000] == 0 and bus[0xD000] == 0


def test_io_hooks_see_the_banked_in_io_page():
    class Logged(Bus):
        """A bus recording I/O traffic."""

        def __init__(self):
            super().__init__(bytearray(0x10000))
            self.log = []

        def io_read(self, addr):
            self.log.append(("r", addr))
            return 0x99

        def io_write(self, addr, value):
            self.log.append(("w", addr, value))

    bus = Logged()
    bus[0], bus[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    bus[0xD020] = 1
    assert bus[0xD020] == 0x99 and bus.ram[0xD020] == 0
    bus[1] = 0x34
    bus[0xD020] = 2
    assert bus[0xD020] == 2
    assert bus.log == [("w", 0xD020, 1), ("r", 0xD020)]


def test_default_io_is_a_ram_backed_register_file():
    bus = Bus(bytearray(0x10000))
    bus[0], bus[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    bus[0xD418] = 0x0F
    assert bus[0xD418] == bus.ram[0xD418] == 0x0F


def test_memory_map_facts():
    assert mm.PAL.cycles_per_frame == 19656 and mm.NTSC.cycles_per_frame == 17095
    assert mm.video(True) is mm.NTSC and mm.video(False).geometry == (312, 63)
    assert mm.raster_compare(0x9B, 0x37) == 0x137
    assert mm.is_vic_reg(0xD011) and not mm.is_vic_reg(0xD400)
    assert mm.is_cia1_reg(mm.CIA1_ICR) and mm.is_cia2_reg(mm.CIA2_CRA)
    assert mm.is_cia_timer(0xDC04) and mm.is_cia_timer(0xDD07)
    assert not mm.is_cia_timer(0xDC0D) and not mm.is_cia_timer(0xD404)
    assert mm.is_irq_vector(0x0315) and mm.is_nmi_vector(0x0318)
    assert mm.is_cpu_vector(0xFFFA) and not mm.is_cpu_vector(0xFFF9)
    assert len(set(mm.VECTORS)) == 8


@pytest.mark.parametrize(
    "bits, nmi, want",
    [
        (0x37, False, (0x0314, True)),
        (0x37, True, (0x0318, False)),
        (0x35, False, (0xFFFE, False)),
        (0x35, True, (0xFFFA, False)),
    ],
)
def test_dispatch_vector_follows_the_kernal_banking(bits, nmi, want):
    assert c64.dispatch_vector(bits, nmi) == want


@pytest.fixture(params=[False, True], ids=["python", "native"])
def native(request):
    return request.param


def _flat(code, **kwargs):
    ram = bytearray(0x10000)
    ram[0], ram[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    ram[CODE : CODE + len(code)] = bytes(code)
    return ram, *flat_mpu(ram, **kwargs)


def test_flat_raster_reads_follow_the_cycle_counter(native):
    _, mpu, mem = _flat([0xAD, 0x12, 0xD0], native=native)  # LDA $D012
    seen = set()
    for _ in range(8):
        mpu.pc = CODE
        mpu.step()
        seen.add(mpu.a)
        mpu.processorCycles += mm.PAL.cycles_per_line
    assert len(seen) == 8
    mpu.processorCycles = 300 * mm.PAL.cycles_per_line
    mem._subject.ram[0xD011] = 0x1B
    assert mem[0xD011] == 0x9B and mem[0xD012] == 300 & 0xFF
    mem[1] = 0x34
    assert mem[0xD011] == 0x1B  # RAM under the I/O page


def test_flat_ntsc_beam_wraps_at_263_lines(native):
    _, mpu, mem = _flat([], ntsc=True, native=native)
    mpu.processorCycles = 263 * mm.NTSC.cycles_per_line
    assert mem[0xD012] == 0


def test_run_to_rts_returns_and_honours_the_cap(native):
    _, mpu, mem = _flat(
        [0xA9, 0x5C, 0x8D, 0x43, 0x00, 0x60], native=native
    )  # LDA #; STA $43; RTS
    run_to_rts(mpu, mem, CODE, 7, max_cycles=1000)
    assert mem[0x43] == 0x5C and mpu.sp == 0xFF
    _, mpu, mem = _flat([0x4C, CODE & 0xFF, CODE >> 8], native=native)  # JMP *
    run_to_rts(mpu, mem, CODE, 0, max_cycles=1000)
    assert 1000 < mpu.processorCycles < 1010


def test_access_cycle_is_the_last_cycle_of_the_instruction(native):
    _, mpu, _ = _flat([0x8D, 0x00, 0xD4], native=native)  # STA $D400: 4 cycles
    mpu.pc, mpu.processorCycles = CODE, 100
    assert access_cycle(mpu) == 103


def test_flat_mpu_banks_roms_through_the_port(native):
    ram = _shadowed()
    ram[0], ram[1] = mm.DDR_DEFAULT, mm.BANK_DEFAULT
    mpu, mem = flat_mpu(ram, native=native)
    assert type(mpu).__name__ == ("NativeMPU" if native else "MPU")
    bus = mem._subject
    assert bus.bits == 7 and bus.io and mem[1] == mm.BANK_DEFAULT
    assert mem[0xE000] == kernal_stub()[0] and mem[0xA000] != 0xAB
    mem[1] = 0x34
    assert not bus.io and (mem[0xE000], mem[0xA000], mem[0xD000]) == (0xCD, 0xAB, 0xEF)
    assert mem[1] == 0x34 | 0x10 and bus.ram[1] == 0x34
    mem[0], mem[1] = 0x00, 0x00
    assert bus.bits == 7 and mem[1] == port_read(0, 0)


def test_native_ram_is_owned_by_the_native_memory():
    ram = bytearray(0x10000)
    ram[0x2000] = 0x11
    _, mem = flat_mpu(ram, native=True)
    assert mem[0x2000] == 0x11 and mem._subject.ram[0x2000] == 0x11
    mem[0x2001] = 0x22
    assert mem._subject.ram[0x2001] == 0x22 and ram[0x2001] == 0
    mem._subject.ram[0x2002] = 0x33
    assert mem[0x2002] == 0x33 and len(mem) == 0x10000
    assert mem.last_load == 0 and mem._subject is mem


def test_native_true_without_native_core(monkeypatch):
    monkeypatch.setattr("jennings.memory.NATIVE", False)
    with pytest.raises(ImportError, match="native core"):
        flat_mpu(bytearray(0x10000), native=True)
    assert type(flat_mpu(bytearray(0x10000))[0]).__name__ == "MPU"


def test_flat_mpu_needs_the_cpu_extra(monkeypatch):
    real = builtins.__import__

    def no_jennings(name, *args, **kwargs):
        if name.startswith("jennings"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_jennings)
    with pytest.raises(ImportError, match=r"vagrant-moses\[cpu\]"):
        flat_mpu(bytearray(0x10000))
