"""KernalHost: the real KERNAL jump-table ABI serviced against a host directory."""

import builtins

import pytest

from vagrant_moses import memmap as mm
from vagrant_moses.kernal import (
    JUMP_TABLE,
    ST,
    KernalHost,
    KernalHostError,
    sys_addr,
)

pytest.importorskip("jennings")

CODE = 0xC000
NAME = 0xC200
K = JUMP_TABLE


def jsr(name):
    """``JSR`` to a KERNAL jump-table entry."""
    return bytes((0x20, K[name] & 0xFF, K[name] >> 8))


def setlfs(lfn, dev, sec):
    """``LDA #lfn; LDX #dev; LDY #sec; JSR SETLFS``."""
    return bytes((0xA9, lfn, 0xA2, dev, 0xA0, sec)) + jsr("SETLFS")


def setnam(length):
    """``LDA #len; LDX #<NAME; LDY #>NAME; JSR SETNAM``."""
    return bytes((0xA9, length, 0xA2, NAME & 0xFF, 0xA0, NAME >> 8)) + jsr("SETNAM")


def lda(value):
    """``LDA #value``."""
    return bytes((0xA9, value))


def ldx(value):
    """``LDX #value``."""
    return bytes((0xA2, value))


def sta(addr):
    """``STA addr``."""
    return bytes((0x8D, addr & 0xFF, addr >> 8))


def carry(host):
    """The 6510 carry flag."""
    return host.mpu.p & 1


@pytest.fixture(name="host")
def fixture_host(tmp_path):
    """A host over an empty directory."""
    return KernalHost(tmp_path)


def run(host, code, name=b""):
    """Run ``code`` at ``CODE`` with ``name`` at ``NAME``, ending in ``JMP *``."""
    end = CODE + len(code)
    body = code + bytes((0x4C, end & 0xFF, end >> 8))
    host.load_prg(CODE.to_bytes(2, "little") + body)
    host.ram[NAME : NAME + len(name)] = name
    host.run(max_steps=10_000)
    assert host.mpu.pc == end
    return host


def open_file(lfn, sec, name):
    """SETLFS on device 8 then SETNAM and OPEN."""
    return setlfs(lfn, 8, sec) + setnam(len(name)) + jsr("OPEN")


def test_write_file_via_chkout_and_chrout(host, tmp_path):
    code = (
        open_file(2, 2, b"0:OUT,S,W")
        + ldx(2)  # file 2
        + jsr("CHKOUT")
        + lda(0x48)  # 'H'
        + jsr("CHROUT")
        + lda(0x49)  # 'I'
        + jsr("CHROUT")
        + jsr("CLRCHN")
        + lda(0x21)  # '!' to the screen
        + jsr("CHROUT")
        + lda(2)
        + jsr("CLOSE")
    )
    run(host, code, b"0:OUT,S,W")
    assert (tmp_path / "8" / "OUT").read_bytes() == b"HI"
    assert host.output == b"!"
    assert not carry(host) and not host.files


def test_secondary_one_opens_for_write_and_replace_prefix(host, tmp_path):
    code = open_file(3, 1, b"@0:NEW") + ldx(3) + jsr("CHKOUT") + lda(0x41)
    run(host, code + jsr("CHROUT") + jsr("CLALL"), b"@0:NEW")
    assert (tmp_path / "8" / "NEW").read_bytes() == b"A"
    assert host.chan_out is None and not host.files


def test_open_errors(host, tmp_path):
    run(host, open_file(2, 0, b"NOPE"), b"NOPE")
    assert carry(host) and host.mpu.a == 4
    run(host, setlfs(2, 8, 0) + lda(0) + jsr("SETNAM") + jsr("OPEN"))
    assert carry(host) and host.mpu.a == 8
    (tmp_path / "8").mkdir()
    (tmp_path / "8" / "F").write_bytes(b"x")
    run(host, open_file(2, 0, b"F") + jsr("OPEN"), b"F")
    assert carry(host) and host.mpu.a == 2
    assert 2 in host.files and not host.files[2][1]


def test_channel_errors(host, tmp_path):
    (tmp_path / "8").mkdir()
    (tmp_path / "8" / "R").write_bytes(b"x")
    run(host, ldx(5) + jsr("CHKIN"))
    assert carry(host) and host.mpu.a == 3
    run(host, ldx(5) + jsr("CHKOUT"))
    assert carry(host) and host.mpu.a == 3
    run(host, open_file(1, 0, b"R") + ldx(1) + jsr("CHKOUT"), b"R")
    assert carry(host) and host.mpu.a == 7
    run(host, open_file(4, 1, b"W") + ldx(4) + jsr("CHKIN"), b"W")
    assert carry(host) and host.mpu.a == 6
    run(host, ldx(1) + jsr("CHKIN"))
    assert not carry(host) and host.chan_in == 1


def test_chrin_file_eof_status(host, tmp_path):
    (tmp_path / "8").mkdir()
    (tmp_path / "8" / "AB").write_bytes(b"AB")
    reads = b"".join(
        jsr("CHRIN") + sta(0xC300 + 2 * i) + jsr("READST") + sta(0xC301 + 2 * i)
        for i in range(3)
    )
    run(host, open_file(2, 2, b"AB") + ldx(2) + jsr("CHKIN") + reads, b"AB")
    assert bytes(host.ram[0xC300:0xC306]) == b"A\x00B\x40\x0d\x42"


def test_close_drops_selected_channels(host):
    code = open_file(2, 1, b"X") + ldx(2) + jsr("CHKOUT") + lda(2) + jsr("CLOSE")
    run(host, code + lda(0x5A) + jsr("CHROUT") + lda(9) + jsr("CLOSE"), b"X")
    assert host.output == b"Z" and host.chan_out is None and not carry(host)


def test_close_drops_input_channel(host, tmp_path):
    (tmp_path / "8").mkdir()
    (tmp_path / "8" / "I").write_bytes(b"q")
    host.tap_key(0x4B)
    code = open_file(2, 0, b"I") + ldx(2) + jsr("CHKIN") + lda(2) + jsr("CLOSE")
    run(host, code + jsr("CHRIN"), b"I")
    assert host.mpu.a == 0x4B and host.chan_in is None


def test_getin_and_keyboard_chrin(host):
    host.tap_key(0x31).tap_key(0x32)
    run(host, jsr("GETIN") + sta(0xC300) + jsr("CHRIN") + sta(0xC301) + jsr("GETIN"))
    assert host.ram[0xC300:0xC302] == b"12"
    assert host.mpu.a == 0
    assert host.mpu.p & 0x02 and not carry(host)
    run(host, jsr("CHRIN"))
    assert host.mpu.a == 0


def test_readst_keeps_status_other_calls_clear_it(host):
    host.ram[ST] = 0x55
    run(host, jsr("READST"))
    assert host.mpu.a == 0x55 and host.ram[ST] == 0x55
    run(host, jsr("CLRCHN"))
    assert host.ram[ST] == 0


def save_code(start, end):
    """Point ``$FB`` at ``start`` and SAVE up to ``end``."""
    return (
        lda(start & 0xFF)
        + sta(0xFB)
        + lda(start >> 8)
        + sta(0xFC)
        + lda(0xFB)  # zero-page pointer
        + bytes((0xA2, end & 0xFF, 0xA0, end >> 8))  # LDX #<end; LDY #>end
        + jsr("SAVE")
    )


def load_code(verify, addr):
    """``LDA #verify; LDX #<addr; LDY #>addr; JSR LOAD``."""
    return bytes((0xA9, verify, 0xA2, addr & 0xFF, 0xA0, addr >> 8)) + jsr("LOAD")


def test_save_load_round_trip(host, tmp_path):
    body = bytes(range(1, 17))
    host.ram[0x2000:0x2010] = body
    run(host, setlfs(1, 8, 0) + setnam(4) + save_code(0x2000, 0x2010), b"DATA")
    assert (tmp_path / "8" / "DATA").read_bytes() == b"\x00\x20" + body
    host.ram[0x2000:0x2010] = bytes(16)
    run(host, setlfs(1, 8, 1) + setnam(4) + load_code(0, 0x3000), b"DATA")
    assert host.ram[0x2000:0x2010] == body and host.ram[0x3000] == 0
    assert (host.mpu.x, host.mpu.y) == (0x10, 0x20) and not carry(host)
    run(host, setlfs(1, 8, 0) + setnam(4) + load_code(0, 0x3000), b"DATA")
    assert host.ram[0x3000:0x3010] == body
    assert (host.mpu.x, host.mpu.y) == (0x10, 0x30)


def test_verify(host, tmp_path):
    (tmp_path / "8").mkdir()
    (tmp_path / "8" / "V").write_bytes(b"\x00\x40\x01\x02")
    host.ram[0x4000:0x4002] = b"\x01\x02"
    run(host, setlfs(1, 8, 1) + setnam(1) + load_code(1, 0) + jsr("READST"), b"V")
    assert host.mpu.a == 0
    host.ram[0x4001] = 9
    run(host, setlfs(1, 8, 1) + setnam(1) + load_code(1, 0) + jsr("READST"), b"V")
    assert host.mpu.a == 0x10 and host.ram[0x4001] == 9


def test_load_save_errors(host, tmp_path):
    run(host, setlfs(1, 8, 1) + setnam(4) + load_code(0, 0), b"GONE")
    assert carry(host) and host.mpu.a == 4
    (tmp_path / "8").mkdir()
    (tmp_path / "8" / "S").write_bytes(b"\x00")
    run(host, setlfs(1, 8, 1) + setnam(1) + load_code(0, 0), b"S")
    assert carry(host) and host.mpu.a == 4
    run(host, setlfs(1, 8, 1) + lda(0) + jsr("SETNAM") + load_code(0, 0))
    assert carry(host) and host.mpu.a == 8
    run(host, setlfs(1, 8, 1) + lda(0) + jsr("SETNAM") + save_code(0, 1))
    assert carry(host) and host.mpu.a == 8


def test_trap_via_jmp_returns_to_caller(host):
    sub = CODE + 8
    code = (
        lda(0x58)  # 'X'
        + bytes((0x20, sub & 0xFF, sub >> 8))  # JSR sub
        + bytes((0x4C, 0x05, 0xC0))  # JMP *
        + bytes((0x4C, 0xD2, 0xFF))  # sub: JMP CHROUT
    )
    host.load_prg(CODE.to_bytes(2, "little") + code)
    cycles = host.mpu.processorCycles
    host.run(max_steps=100)
    assert host.output == b"X" and host.mpu.pc == CODE + 5
    assert host.mpu.processorCycles - cycles == 2 + 6 + 3 + 6 + 3


def test_no_trap_when_kernal_banked_out(host):
    host.ram[K["CHROUT"] : K["CHROUT"] + 3] = bytes((0xA9, 0x77, 0x60))  # LDA; RTS
    run(host, lda(0x35) + sta(mm.CPU_PORT) + lda(0x51) + jsr("CHROUT"))
    assert host.mpu.a == 0x77 and not host.output


def test_screen_touched_and_roms(tmp_path):
    host = KernalHost(tmp_path, roms={mm.KERNAL_BASE: b"\x01" * 0x2000})
    assert host.bus[mm.KERNAL_BASE] == 1
    run(host, lda(0x08) + sta(mm.SCREEN_RAM + 1))
    assert host.screen()[:2] == b"\x00\x08" and len(host.screen()) == 0x400
    assert {0x0, 0xC} <= host.touched


def test_run_until_and_cap(host):
    host.load_prg(b"\x00\xc0\xe8\x4c\x00\xc0")  # INX; JMP $C000
    host.run(until=lambda h: h.mpu.x == 5)
    assert host.mpu.x == 5
    with pytest.raises(KernalHostError):
        host.run(max_steps=10)


def test_run_to_unpacked(host):
    stub = bytes((0xA9, 0x01, 0x4C, 0x02, 0xC0))  # LDA #1; JMP $C002
    src = 0x0801 + 22
    unpack = (
        lda(0xAA)
        + sta(0x0A00)  # write into the PRG's own page
        + sta(0xD000)  # the next page, extending the unpacked run
        + ldx(len(stub) - 1)
        + bytes((0xBD, src & 0xFF, src >> 8))  # loop: LDA src,X
        + bytes((0x9D, 0x00, 0xC0))  # STA $C000,X
        + bytes((0xCA, 0x10, 0xF7))  # DEX; BPL loop
        + bytes((0x4C, 0x00, 0xC0))  # JMP $C000
    )
    assert len(unpack) == 22
    pc, code = host.run_to_unpacked(b"\x01\x08" + unpack + stub)
    assert pc == 0xC000 and len(code) == 0x2000 and code[:5] == stub
    with pytest.raises(KernalHostError):
        host.run_to_unpacked(b"\x01\x08\x4c\x01\x08", max_steps=50)


def test_run_to_unpacked_empty_body_owns_its_page(host):
    host.ram[CODE : CODE + 12] = (
        lda(0x4C)
        + sta(0xC100)  # JMP opcode in the PRG's own page
        + lda(0x00)
        + sta(0xC101)
        + bytes((0x4C, 0x30, 0xC0))  # JMP $C030
    )
    host.ram[0xC030 : 0xC030 + 13] = (
        lda(0x40)
        + sta(0xC102)
        + lda(0x4C)
        + sta(0x4000)
        + bytes((0x8E, 0x01, 0x40))  # STX $4001 (X = 0)
    )
    host.ram[0xC03D : 0xC03D + 9] = (
        lda(0x40) + sta(0x4002) + bytes((0x4C, 0x00, 0xC1))  # JMP $C100
    )
    pc, code = host.run_to_unpacked(b"\x00\xc0", entry=CODE, max_steps=100)
    assert pc == 0x4000 and code[:3] == b"\x4c\x00\x40"


def basic(*lines, base=0x0801):
    """Tokenised BASIC lines ``(number, tokens)`` linked from ``base``."""
    out, addr = bytearray(), base
    for number, tokens in lines:
        addr += 4 + len(tokens) + 1
        out += addr.to_bytes(2, "little") + number.to_bytes(2, "little")
        out += tokens + b"\x00"
    return bytes(out) + b"\x00\x00"


def image(program, base=0x0801):
    """``program`` placed at ``base`` in a zeroed address-space image."""
    mem = bytearray(0x10000)
    mem[base : base + len(program)] = program
    return mem


@pytest.mark.parametrize(
    "tokens", [b"\x9e2085", b"\x9e 2085", b"\x9e  2085:\x80", b"\x8f\x9e2085"]
)
def test_sys_addr_operand(tokens):
    assert sys_addr(image(basic((10, tokens)))) == 2085


def test_sys_addr_skips_line_number_and_follows_links():
    program = basic(
        (0x9E64, b"\x99\x22HI\x22"), (0x649E, b"\x9e(2061)"), (2, b"\x9e49152")
    )
    assert sys_addr(bytes(image(program))) == 49152


def test_sys_addr_over_bus(host):
    host.ram[0x0801:0x0810] = basic((10, b"\x9e2061"))
    assert sys_addr(host.bus) == 2061


@pytest.mark.parametrize(
    "mem",
    [
        image(basic((10, b"\x99"))),
        image(b"\x01\x08\x0a\x00\x99\x00"),
        bytes(0x0803),
        bytes(image(basic((10, b"\x8f\x8f")))[:0x0806]),
        bytes(image(basic((10, b"\x9e")))[:0x0806]),
        b"",
    ],
)
def test_sys_addr_missing(mem):
    with pytest.raises(ValueError):
        sys_addr(mem)


def test_missing_jennings(monkeypatch, tmp_path):
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("jennings"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"vagrant-moses\[cpu\]"):
        KernalHost(tmp_path)


def test_injected_mpu_runs_on_the_host_bus(tmp_path):
    from jennings.devices.mpu6502 import MPU  # pylint: disable=import-outside-toplevel

    mpu = MPU()
    host = KernalHost(tmp_path, mpu=mpu)
    mpu.memory = host.bus
    run(host, lda(0x2A) + jsr("CHROUT"))
    assert host.mpu is mpu and host.output == b"*"
