"""C64 machine code as C: deity-informant's static lift with the C64 I/O model and host.

``python -m vagrant_moses.cgen IMAGE --load ADDR --init PC --play PC -o OUT.c``; needs
the ``cgen`` extra (``docs/cgen.md``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import memmap as mm
from .c64 import poweron_ram

#: The I/O window ``[lo, hi)``: its writes are kept in order and none of it is data.
VOLATILE = ((mm.IO_BASE, mm.IO_LAST + 1),)

#: Unmirrored registers whose read changes device state: the VIC-II collision registers
#: (cleared, with their IRQ sources), each CIA's ICR (flags cleared, IRQ released) and TOD
#: tenths and hours (release and set the TOD read latch), and CIA 2's port B (pulses PC2
#: on the user port).
SENSITIVE_REGISTERS = frozenset(
    {mm.VIC_SPRITE_SPRITE, mm.VIC_SPRITE_DATA, mm.CIA2_BASE + mm.PRB}
    | {
        base + reg
        for base in (mm.CIA1_BASE, mm.CIA2_BASE)
        for reg in (mm.TOD_10THS, mm.TOD_HR, mm.ICR)
    }
)

#: SID registers whose read loads the SID's data-bus latch, which its write-only registers
#: read back.
SID_LATCH_REGISTERS = frozenset(range(mm.SID_POTX, mm.SID_ENV3 + 1))

#: Every I/O address whose throw-away read is observable: a mirror of a
#: :data:`SENSITIVE_REGISTERS` register, or of a :data:`SID_LATCH_REGISTERS` register whose
#: indexed fix-up access (the same address, or ``$100`` above it on a page cross) can leave
#: the register and so not reload the latch.
SENSITIVE = tuple(
    a
    for a in range(mm.IO_BASE, mm.IO_LAST + 1)
    if (r := mm.io_register(a)) in SENSITIVE_REGISTERS
    or (r in SID_LATCH_REGISTERS and mm.io_register(a + 0x100) != r)
)

ENTRIES = ("cgen_init", "cgen_play")
_SONG_STUB = {"x": 0xAA, "y": 0xA8}  # TAX, TAY

#: The C64 host (unless ``CGEN_HOST`` is defined) and the driver API, after the lift.
HOST = """
#include "driver.h"

#ifndef CGEN_HOST
volatile uint16_t cgen_trap;

uint8_t io_read(uint16_t a) { return *(volatile uint8_t *)a; }

void io_write(uint16_t a, uint8_t v) { *(volatile uint8_t *)a = v; }

__attribute__((noinline)) void trap(uint16_t k) {
  cgen_trap = k;
  for (;;)
    __asm__ volatile(".byte 0x02");
}
#endif

void tune_init(uint8_t song) { cgen_init(song); }

void tune_play(void) { cgen_play(0); }
"""


def song_entry(image, init, loaded, song_reg="a"):
    """The init entry PC taking the song in ``A``: ``init``, or for ``song_reg`` ``x``/``y``
    a ``TAX``/``TAY; JMP init`` stub written to ``image`` just below ``loaded``."""
    if song_reg == "a":
        return init
    if song_reg not in _SONG_STUB:
        raise ValueError(f"song_reg must be a, x or y, not {song_reg!r}")
    pc = loaded[0] - 4
    if pc < 0:
        raise ValueError("no room below the loaded program for the song stub")
    image[pc : pc + 4] = bytes((_SONG_STUB[song_reg], 0x4C, init & 0xFF, init >> 8))
    return pc


def translate_c64(  # pylint: disable=too-many-arguments
    image, init, play, *, loaded, song_reg="a", header="cgen.h", prefix=None
):
    """``(c_text, h_text, info)`` of the program in ``image`` (64 KiB) loaded at ``loaded``
    (``[lo, hi)``): ``init`` gets the song in ``song_reg``, ``play`` ``A = 0``; the I/O
    model is :data:`VOLATILE` and :data:`SENSITIVE` and the ``.c`` ends with :data:`HOST`.
    """
    from deity_informant import cgen  # pylint: disable=import-outside-toplevel

    image = bytearray(image)
    entries = [
        (song_entry(image, init, loaded, song_reg), ENTRIES[0]),
        (play, ENTRIES[1]),
    ]
    c, h, info = cgen.translate(
        image,
        entries,
        VOLATILE,
        SENSITIVE,
        loaded=tuple(loaded),
        header=header,
        prefix=prefix,
    )
    return c + HOST, h, info


def _int(text):
    return int(text, 0)


def main(argv=None):
    """Translate a raw C64 program image's init and play routines to C."""
    ap = argparse.ArgumentParser(
        prog="python -m vagrant_moses.cgen", description=main.__doc__
    )
    ap.add_argument("image", help="raw program bytes, loaded over the power-on RAM")
    ap.add_argument("--load", type=_int, required=True, help="load address")
    ap.add_argument("--init", type=_int, required=True, help="init PC")
    ap.add_argument(
        "--play", type=_int, required=True, help="play PC (called with A = 0)"
    )
    ap.add_argument("--song-reg", choices=("a", "x", "y"), default="a")
    ap.add_argument("-o", "--out", required=True, help="the .c")
    ap.add_argument("--header", help="the .h (default: the .c's name with .h)")
    args = ap.parse_args(argv)
    data = Path(args.image).read_bytes()
    image = bytearray(poweron_ram())
    image[args.load : args.load + len(data)] = data
    header = Path(args.header) if args.header else Path(args.out).with_suffix(".h")
    print(f"{args.image}: translating ${args.load:04X}+{len(data)}", file=sys.stderr)
    t0 = time.perf_counter()
    c, h, info = translate_c64(
        image,
        args.init,
        args.play,
        loaded=(args.load, args.load + len(data)),
        song_reg=args.song_reg,
        header=header.name,
    )
    Path(args.out).write_text(c, encoding="utf-8")
    header.write_text(h, encoding="utf-8")
    stages = ", ".join(f"{k} {v:.2f} s" for k, v in info["times"].items())
    print(
        f"{args.out}: {len(c)} bytes, {len(info['regions'])} regions, "
        f"{len(info['traps'])} traps in {time.perf_counter() - t0:.2f} s ({stages})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
