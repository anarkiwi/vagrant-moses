# vagrant-moses

A standalone C64 machine model: the memory map, the `$01`-banked bus and ROMs
(sidplayfp's ROM-less set, or the real ones fetched and SHA-256 verified into a cache),
the VIC-II and two CIAs over a shared two-phase event scheduler, a cycle-exact 6510
machine matching libsidplayfp, KERNAL file/keyboard traps, the power-on RAM, the
installed-interrupt-vector discovery, the ROM-free IRQ dispatch path, and native
extraction of files from release media (zip/gzip, d64/d71/d81, t64, p00).

Pure Python with no runtime dependencies (wheels also carry a native Rust `Machine`); the CPU-driven parts (`Machine`, `flat_mpu`,
`KernalHost`) run on jennings via the `cpu` extra; the media CLI's progress bar
needs the `cli` extra. The SID's audio model is not implemented here: `sidorder` judges
envelopes through the optional pylibresidfp (the `sidorder` extra). SID-file knowledge
(PSID/RSID containers, subtune headers, drivers) is deliberately not here.

## Install

```bash
pip install -e ".[cpu]"            # add ".[sidorder]" for vagrant_moses.sidorder
```

## Use

```python
from vagrant_moses import Cia, Machine, Scheduler, Vic, c64

mem = bytearray(c64.poweron_ram())                       # 64 KiB power-on image
vec, kernal = c64.installed_vector(mem, written, (lo, hi))
entry_pc = c64.install_irq_entry(vm, vec, kernal, img=(lo, hi))

sched = Scheduler()
vic = Vic(sched, ntsc=False, irq=lambda on: ..., set_ba=lambda on: ...)
cia = Cia(sched, interrupt=lambda on: ..., new=False)

machine = Machine(sids={0xD400: sid_device})            # cycle-exact C64
machine.ram[0x1000:0x1000 + len(code)] = code
machine.run(machine.reset(0x1000) + 985248)              # one PAL second
```

```python
from vagrant_moses import iter_prgs

for path, prg in iter_prgs("release.zip", blob):        # zip -> d64.gz -> d64 -> PRG
    ...
```

```bash
pip install "vagrant-moses[cli]"
vagrant-moses media extract dl/ -o prg/                 # prints sha256 size $load path
```

## Docs

- [The machine model: memory map, bus, ROMs, chips, machine, KERNAL and IRQ dispatch](docs/c64.md)
- [The native machine: Rust port, hooks, threading, verification](docs/native.md)
- [SID write order and hard restart: event patterns, the order cover, the settled hard restart](docs/sidorder.md)
- [Release media: nested containers, disk geometries, tapes, and the CLI](docs/media.md)

## Develop

```bash
pip install -e ".[dev]"
black --check src tests scripts && pylint src tests scripts && pytest
```
