"""vagrant_moses.cgen: the C64 I/O model, translate_c64 under gcc and llvm-mos, the CLI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vagrant_moses import cgen
from vagrant_moses import memmap as mm
from vagrant_moses.c64 import poweron_ram

from asm import assemble

try:
    from deity_informant.cgen import clang_format, mlir
    from deity_informant.cgen import translate as di_translate
except ImportError:
    mlir = None

needs_cgen = pytest.mark.skipif(
    mlir is None
    or mlir.runner(build=False) is None
    or not shutil.which("clang-format")
    or not shutil.which("gcc"),
    reason="needs the cgen extra, mlir-opt (or its Docker image), clang-format and gcc",
)
OUT = Path(__file__).resolve().parents[1] / "out" / "test_cgen"
MOS = "ghcr.io/anarkiwi/docker-mos-llvm-sdk:v23.4.0"
SID = range(mm.SID_BASE, mm.SID_BASE + 0x20)

DRIVER_H = """#include <stdint.h>
void tune_init(uint8_t song);
void tune_play(void);
extern volatile uint8_t song_number;
void frame_wait(void);
"""
DRIVER_C = """#include "driver.h"
volatile uint8_t song_number;
__attribute__((noinline)) void frame_wait(void) { __asm__ volatile(""); }
int main(void) {
  tune_init(song_number);
  for (;;) {
    frame_wait();
    tune_play();
  }
}
"""
HARNESS = r"""#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include "driver.h"
uint8_t IO[65536];
static jmp_buf env;
uint8_t io_read(uint16_t a) {
  printf("r %u\n", a);
  return IO[a];
}
void io_write(uint16_t a, uint8_t v) {
  IO[a] = v;
  printf("w %u %u\n", a, v);
}
void trap(uint16_t k) {
  printf("t %u\n", k);
  longjmp(env, 1);
}
int main(int argc, char **argv) {
  for (int i = 3; i + 1 < argc; i += 2)
    IO[atoi(argv[i])] = (uint8_t)atoi(argv[i + 1]);
  if (setjmp(env))
    return 0;
  tune_init((uint8_t)atoi(argv[1]));
  for (int f = atoi(argv[2]); f > 0; f--) {
    puts("f");
    tune_play();
  }
  return 0;
}
"""

PLAYER = """
init:   AND #$03
        TAX
        LDA start,X
        STA pos
        LDA #$0F
        STA $D418
        RTS
play:   LDX pos
        LDA notes,X
        LDY regs,X
        STA $D400,Y
        INX
        TXA
        AND #$0F
        STA pos
        RTS
pos:    .byte 00
start:  .byte 00,04,08,0C
notes:  .byte {notes}
regs:   .byte 00,01,04,05,06,07,08,0B,0C,0D,0E,0F,12,13,14,16
"""
NOTES = ("11,22,41,09,F0,33,44,21", "81,12,13,14,15,16,17,18")
POLL = """
init:   LDA $D418
wait:   LDA $D012
        CMP #$FC
        BNE wait
        LDA $DC0D
        LDA $D41B
        STA $D40F
        RTS
play:   RTS
"""
JUMP = """
init:   RTS
play:   LDA $FB
        STA vec
        LDA $FC
        STA vech
        JMP (vec)
vec:    .byte 00
vech:   .byte 00
"""
SONG_X = """
init:   STX $D418
        RTS
play:   RTS
"""


def player(notes=NOTES[0]):
    """The PLAYER source with ``notes`` (16 bytes as two halves)."""
    return PLAYER.format(notes=f"{notes},{notes}")


def program(src, org=0x1000):
    """``(image, labels, loaded)``: ``src`` assembled at ``org`` over the power-on RAM."""
    code, labels = assemble(src, org)
    image = bytearray(poweron_ram())
    image[org : org + len(code)] = code
    return image, labels, (org, org + len(code))


def lift(src, org=0x1000, **kw):
    """``(image, labels, c, h, info)`` of ``src`` through :func:`cgen.translate_c64`."""
    image, labels, loaded = program(src, org)
    c, h, info = cgen.translate_c64(
        image, labels["init"], labels["play"], loaded=loaded, header="tune.h", **kw
    )
    return image, labels, c, h, info


def frames_of(lines):
    """The harness output as one event list per call, init first."""
    calls = [[]]
    for line in lines:
        kind, *args = line.split()
        if kind == "f":
            calls.append([])
        else:
            calls[-1].append((kind, *map(int, args)))
    return calls


def run_gcc(tmp_path, c, h, song, frames, preset=()):
    """The events of ``tune_init(song)`` and ``frames`` plays of ``c`` under gcc."""
    for name, text in (("tune.c", c), ("tune.h", h), ("driver.h", DRIVER_H)):
        (tmp_path / name).write_text(text)
    (tmp_path / "host.c").write_text(HARNESS)
    exe = tmp_path / "run"
    cmd = [
        "gcc",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DCGEN_HOST",
        "-DCGEN_OOB_EXACT",
    ]
    cmd += [f"-I{tmp_path}", "-o", str(exe), str(tmp_path / "tune.c")]
    subprocess.run([*cmd, str(tmp_path / "host.c")], check=True)
    args = [str(song), str(frames), *(str(v) for kv in preset for v in kv)]
    done = subprocess.run([exe, *args], capture_output=True, text=True, check=True)
    return frames_of(done.stdout.splitlines())


def reference(image, labels, song, frames):
    """The SID writes of the 6502 program on :func:`flat_mpu`, one list per call."""
    from vagrant_moses.cpu import (  # pylint: disable=import-outside-toplevel
        flat_mpu,
        run_to_rts,
    )

    mpu, mem = flat_mpu(bytearray(image))
    calls = []
    mem.subscribe_to_write(SID, lambda a, v: calls[-1].append(("w", a, v)))
    for pc, a in [(labels["init"], song)] + [(labels["play"], 0)] * frames:
        calls.append([])
        run_to_rts(mpu, mem, pc, a, 100_000)
    return calls


def writes(calls):
    """Only the writes of each call."""
    return [[e for e in call if e[0] == "w"] for call in calls]


def test_volatile_window():
    """The whole I/O page is volatile."""
    assert cgen.VOLATILE == ((0xD000, 0xE000),)


@pytest.mark.parametrize(
    "addr,reg",
    [
        (0xD3D2, 0xD012),
        (0xD07F, 0xD03F),
        (0xD43B, 0xD41B),
        (0xD7FC, 0xD41C),
        (0xDCFD, 0xDC0D),
        (0xDD31, 0xDD01),
        (0xD9AB, 0xD9AB),
        (0xDF00, 0xDF00),
    ],
)
def test_io_register(addr, reg):
    """Mirrors decode to the chip's unmirrored register."""
    assert mm.io_register(addr) == reg


#: Per chip, ``(mirror base, stride, last address, offsets whose read changes state)``.
READ_EFFECTS = (
    (mm.VIC_BASE, mm.VIC_STRIDE, 0xD3FF, {0x1E, 0x1F}),
    (mm.CIA1_BASE, mm.CIA_STRIDE, 0xDCFF, {mm.TOD_10THS, mm.TOD_HR, mm.ICR}),
    (mm.CIA2_BASE, mm.CIA_STRIDE, 0xDDFF, {mm.PRB, mm.TOD_10THS, mm.TOD_HR, mm.ICR}),
)


@pytest.mark.parametrize("base,stride,last,regs", READ_EFFECTS)
def test_read_effect_registers(base, stride, last, regs):
    """Each chip's registers with read effects are sensitive in every mirror, no others."""
    sens = set(cgen.SENSITIVE)
    assert {r - base for r in cgen.SENSITIVE_REGISTERS if base <= r <= last} == regs
    for mirror in range(base, last + 1, stride):
        assert {a - mirror for a in sens if mirror <= a < mirror + stride} == regs


def test_sid_latch_registers():
    """A SID read only loads the bus latch: sensitive where its page-cross fix-up leaves
    the SID (page ``$D7``), elsewhere reloaded by that fix-up."""
    sid = {a for a in cgen.SENSITIVE if mm.SID_BASE <= a <= mm.SID_LAST}
    assert cgen.SID_LATCH_REGISTERS == {0xD419, 0xD41A, 0xD41B, 0xD41C}
    assert sid == {
        m + r for m in range(0xD700, 0xD800, mm.SID_STRIDE) for r in range(0x19, 0x1D)
    }


def test_sensitive_set():
    """Volatile-value and write-only registers, colour RAM and I/O 1/2 are not sensitive."""
    sens = set(cgen.SENSITIVE)
    assert len(sens) == 16 * 2 + 8 * 4 + 16 * 3 + 16 * 4
    for kept in (0xD01E, 0xD3DF, 0xD7FB, 0xDC08, 0xDC0B, 0xDCFD, 0xDD01, 0xDDF1):
        assert kept in sens
    for dropped in (
        0xD011,
        0xD012,
        0xD013,
        0xD019,
        0xD418,
        0xD419,
        0xD41B,
        0xD6FC,
        0xDC00,
        0xDC01,
        0xDC04,
        0xDC09,
        0xDC0C,
        0xDD00,
        0xD800,
        0xDF00,
    ):
        assert dropped not in sens


def test_song_entry():
    """A song in X or Y goes through a stub below the program."""
    image = bytearray(0x10000)
    assert cgen.song_entry(image, 0x1234, (0x1000, 0x2000)) == 0x1234
    assert cgen.song_entry(image, 0x1234, (0x1000, 0x2000), "y") == 0x0FFC
    assert image[0x0FFC:0x1000] == bytes((0xA8, 0x4C, 0x34, 0x12))
    with pytest.raises(ValueError):
        cgen.song_entry(image, 0x1234, (0x1000, 0x2000), "s")
    with pytest.raises(ValueError):
        cgen.song_entry(image, 2, (2, 0x10), "x")


@needs_cgen
def test_indexed_sid_writes_no_region(tmp_path):
    """``STA $D400,Y`` with Y from data writes I/O, in order, and makes no region."""
    image, labels, c, h, info = lift(player())
    assert all(org + n <= mm.IO_BASE or org > mm.IO_LAST for org, n in info["regions"])
    assert "goto" not in c and clang_format(c) == c
    _, _, loaded = program(player())
    narrow = di_translate(
        image, [(labels["play"], "p")], [(0xD400, 0xD41D)], loaded=loaded
    )
    # The offsets Y takes, every value the program stores to its table, lie in the window.
    assert "io_write(0xD400 + " in narrow[0]
    assert all(
        org + n <= mm.IO_BASE or org > mm.IO_LAST for org, n in narrow[2]["regions"]
    )
    for song in range(4):
        got = run_gcc(tmp_path, c, h, song, 40)
        assert writes(got) == reference(image, labels, song, 40)


@needs_cgen
def test_invariant_c():
    """Other song data and another origin give the same ``.c``."""
    outs = [
        lift(player(n), org)[2:4]
        for n, org in zip(NOTES * 2, (0x1000,) * 2 + (0x3456,) * 2)
    ]
    assert len({c for c, _ in outs}) == 1
    assert len({h for _, h in outs}) == 4


@needs_cgen
def test_sensitive_reads_kept(tmp_path):
    """The raster poll, ICR acknowledge and OSC3 reads stay, in order; others go."""
    _, _, c, h, _ = lift(POLL)
    assert "io_read(0xD012)" in c and "io_read(0xDC0D)" in c and "0xD418" not in c
    got = run_gcc(tmp_path, c, h, 0, 1, [(0xD012, 0xFC), (0xD41B, 0x5A)])
    assert got[0] == [("r", 0xD012), ("r", 0xDC0D), ("r", 0xD41B), ("w", 0xD40F, 0x5A)]


DUMMY = """
init:   LDY #$19
        STA $D400,Y
        LDY #$0D
        STA $DC00,Y
        LDX #$20
        STA $D7FB,X
        RTS
play:   RTS
"""


@needs_cgen
def test_dummy_reads(tmp_path):
    """Indexed stores' throw-away reads: SID ``$D419`` is dropped, CIA ICR ``$DC0D`` and
    SID ``$D71B`` before a page cross into colour RAM are kept."""
    _, _, c, h, _ = lift(DUMMY)
    assert "0xD419" in c and "io_read(0xD419)" not in c
    got = run_gcc(tmp_path, c, h, 0, 0)
    assert [e[:2] for e in got[0]] == [
        ("w", 0xD419),
        ("r", 0xDC0D),
        ("w", 0xDC0D),
        ("r", 0xD71B),
        ("w", 0xD81B),
    ]


@needs_cgen
def test_trap(tmp_path):
    """A computed jump into code the lift cannot keep traps through the host."""
    _, _, c, h, info = lift(JUMP)
    ((kind, k),) = run_gcc(tmp_path, c, h, 0, 2)[1]
    assert kind == "t" and k in info["traps"]


@needs_cgen
def test_driver_api(tmp_path):
    """``tune_init`` takes the song, here in X through the stub."""
    _, _, c, h, _ = lift(SONG_X, song_reg="x")
    assert '#include "driver.h"' in c and "cgen_play(0)" in c
    assert run_gcc(tmp_path, c, h, 5, 0)[0] == [("w", 0xD418, 5)]


@needs_cgen
def test_cli(tmp_path, capsys):
    """The CLI translates a raw image to a ``.c`` and ``.h``."""
    code, labels = assemble(player(), 0x1000)
    (tmp_path / "p.bin").write_bytes(code)
    argv = [str(tmp_path / "p.bin"), "--load", "0x1000", "--init", str(labels["init"])]
    argv += ["--play", str(labels["play"]), "-o", str(tmp_path / "p.c")]
    assert cgen.main(argv) == 0
    assert '#include "p.h"' in (tmp_path / "p.c").read_text()
    assert (tmp_path / "cgen_trap.c").read_text() == cgen.TRAP
    assert "R0_ORG" in (tmp_path / "p.h").read_text()
    assert "traps" in capsys.readouterr().err


def mos_build(name, c, h):
    """``(prg, elf symbols)`` of ``c`` linked with the test driver by llvm-mos."""
    if (
        not shutil.which("docker")
        or subprocess.run(
            ["docker", "pull", "-q", MOS], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("needs Docker and the llvm-mos image")
    from jennings.coverage.elf import (
        load_elf,
    )  # pylint: disable=import-outside-toplevel

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for f, text in (
        ("tune.c", c),
        ("tune.h", h),
        ("driver.h", DRIVER_H),
        ("driver.c", DRIVER_C),
        ("trap.c", cgen.TRAP),
    ):
        (out / f).write_text(text)
    cmd = ["docker", "run", "--rm", "-v", f"{out}:/w", "-w", "/w", MOS, "mos-c64-clang"]
    cmd += ["-std=c99", "-Os", "-Wall", "-Wextra", "-Werror", "-DCGEN_OOB_EXACT", "-I."]
    cmd += ["-o", "t.prg"]
    done = subprocess.run(
        [*cmd, "tune.c", "driver.c", "trap.c"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return (out / "t.prg").read_bytes(), load_elf(out / "t.prg.elf")[1]


def c64_run(prg, syms, song, frames):
    """``(calls, mpu, mem)``: the PRG's SID writes per call on :func:`flat_mpu`."""
    from vagrant_moses.cpu import flat_mpu  # pylint: disable=import-outside-toplevel

    ram = bytearray(poweron_ram())
    load = prg[0] | prg[1] << 8
    ram[load : load + len(prg) - 2] = prg[2:]
    mpu, mem = flat_mpu(ram)
    if not hasattr(mpu, "run_until"):
        pytest.skip("needs jennings' native core")
    calls = [[]]
    mem.subscribe_to_write(SID, lambda a, v: calls[-1].append(("w", a, v)))
    mpu.pc = syms["_start"]
    assert mpu.run_until(max_cycles=100_000, stop_pcs=[syms["main"]]) == "pc"
    mem[syms["song_number"]] = song
    for _ in range(frames + 1):
        reason = mpu.run_until(max_cycles=1_000_000, stop_pcs=[syms["frame_wait"]])
        if reason != "pc":
            return calls, mpu, mem
        calls.append([])
        ret = mem[0x100 + (mpu.sp + 1) % 256] | mem[0x100 + (mpu.sp + 2) % 256] << 8
        mpu.sp, mpu.pc = (mpu.sp + 2) & 0xFF, ret + 1
    return calls[:-1], mpu, mem


@needs_cgen
def test_mos_player():
    """The llvm-mos PRG writes the SID as the 6502 original on the same C64."""
    image, labels, c, h, _ = lift(player())
    calls, _, _ = c64_run(*mos_build("player", c, h), 2, 30)
    assert calls == reference(image, labels, 2, 30)


@needs_cgen
def test_mos_trap():
    """A trap stores ``k`` in ``cgen_trap`` and jams the 6510 in ``trap``."""
    _, _, c, h, info = lift(JUMP)
    prg, syms = mos_build("trap", c, h)
    calls, mpu, mem = c64_run(prg, syms, 0, 3)
    assert len(calls) == 2 and mpu.jammed
    assert syms["trap"] < mpu.pc < syms["trap"] + 16
    assert mem[syms["cgen_trap"]] | mem[syms["cgen_trap"] + 1] << 8 in info["traps"]
