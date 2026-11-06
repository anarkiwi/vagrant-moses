# C64 programs as C: `vagrant_moses.cgen`

The C64 layer over [deity-informant](https://github.com/anarkiwi/deity-informant)'s static
6502 → C translator (`deity_informant.cgen`, its `docs/cgen.md`): the C64 I/O model the lift
needs, and a host so the `.c` builds as a C64 program with llvm-mos. Needs the `cgen` extra
and `mlir-opt` (LLVM 22) and `clang-format` on `PATH`.

```python
from vagrant_moses.c64 import poweron_ram
from vagrant_moses.cgen import translate_c64

image = bytearray(poweron_ram())
image[load : load + len(code)] = code
c, h, info = translate_c64(image, init, play, loaded=(load, load + len(code)),
                           header="tune.h")
```

```bash
python -m vagrant_moses.cgen player.bin --load 0x1000 --init 0x1000 --play 0x1003 \
    -o tune.c --header tune.h
```

The CLI loads the raw bytes over the power-on RAM. Container formats (PSID/RSID headers,
subtunes) are the caller's concern: it passes the image and the entry points.

## I/O model

| | Addresses | Meaning |
|---|---|---|
| `VOLATILE` | `$D000-$DFFF` | every write is `io_write`, kept in order; no I/O byte is a data region |
| `SENSITIVE` | the I/O addresses whose read changes device state | reads kept as `io_read`, in order, even when the value is unused |

A read whose value is unused is observable only through a side effect. `SENSITIVE_REGISTERS`
are the registers whose read changes device state: VIC-II `$D01E`/`$D01F` (collisions,
cleared with their IRQ sources); each CIA's ICR (`$xD`: flags cleared, IRQ/NMI released),
TOD tenths (`$x8`, releases the TOD read latch) and hours (`$xB`, sets it); CIA 2's port B
(`$DD01`: pulses PC2 on the user port; CIA 1's PC is not connected). Raster, light pen,
`$D019`, paddles, OSC3, ENV3, the CIA ports, timers, TOD seconds and minutes and SDR return
values that vary, but reading them changes nothing.

`SID_LATCH_REGISTERS` (`$D419-$D41C`) load the SID's data-bus latch, which its write-only
registers read back. A 6502 throw-away read is the unfixed address of an indexed access,
followed in the same instruction by the fixed access at that address or, on a page cross,
`$100` above it; the SID repeats every `$20`, so that access reaches the same register and
reloads the latch. Only in page `$D7` can it leave the SID (for colour RAM), so only those
mirrors are sensitive.

`SENSITIVE` is every address of the I/O window that `memmap.io_register` decodes to a
`SENSITIVE_REGISTERS` register (the VIC-II repeats every `$40` to `$D3FF`, each CIA every
`$10` over its page), plus those `$D7xx` SID mirrors. Other I/O reads (raster polls, OSC3,
timers, write-only SID registers, colour RAM) are dropped by the lift when their value is
unused, and kept as `io_read` in order when it is used: a raster poll or a SID model probe
(`LDA $D012 / CMP #$FC / BNE`, `LDA $D41B`) waits on the same hardware the original does.

## `translate_c64(image, init, play, *, loaded, song_reg="a", header="cgen.h", prefix=None)`

Calls `deity_informant.cgen.translate` with `VOLATILE` and `SENSITIVE` and the entries
`cgen_init` (`init`) and `cgen_play` (`play`), and returns its `(c_text, h_text, info)` with
`HOST` appended to the `.c`. `loaded` is the program's `[lo, hi)`. `song_reg` names the
register `init` takes the song in: `x` or `y` puts a `TAX`/`TAY; JMP init` stub in the four
bytes below `loaded`. `prefix` is cgen's `mlir-opt` command prefix.

The `.c` holds only code and the host, so programs with the same player and other song
data, or relocated, give a byte-identical `.c`; the `.h` holds the data regions. `HOST`:

```c
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
```

- `io_read`/`io_write` access the C64's I/O directly.
- `trap(k)` stores `k` in `cgen_trap` and executes a 6510 JAM inside `trap`, so an emulator
  halts at a known PC with the trap number readable through the ELF symbol; `k` indexes
  `info["traps"]` (the original PC and kind).
- `tune_init`/`tune_play` implement a driver's `driver.h` (`main` calls `tune_init(song)`, then
  `tune_play()` once per frame); play gets `A = 0`, as a PSID player's play call does.
- Defining `CGEN_HOST` drops the C64 accessors and `trap` for a host that supplies its own
  (a gcc test harness logging accesses and leaving `trap` by `longjmp`).

## Validation

`tests/test_cgen.py`: the I/O model (window, read-effect registers per chip in every mirror,
the SID latch mirrors, volatile-value registers not sensitive); under gcc
with a logging host, a player storing SID registers through `STA $D400,Y` with `Y` from data
(no region in the I/O page, writes equal to the 6502 original on `flat_mpu` for each song),
a raster poll whose reads stay in order while an unused `$D418` read is dropped, indexed
stores whose throw-away read of `$D419` is dropped and of `$DC0D` and `$D71B` kept, a trap, the song stub; the `.c` across two data sets and two origins; the CLI; and with Docker,
llvm-mos PRGs run on `flat_mpu`: the player's SID writes match the original's, and a trap
jams inside `trap` with `cgen_trap` holding a trap key.
