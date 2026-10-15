# vagrant-moses

A standalone C64 machine model: the volatile I/O models a host needs to drive a
playroutine's interrupts faithfully — the VIC-II raster/badline/sprite-BA state and
the two CIA timers over a shared two-phase event scheduler — plus the power-on RAM,
the installed-interrupt-vector discovery, the ROM-free IRQ dispatch path, and native
extraction of files from release media (zip/gzip, d64/d71/d81, t64, p00).

Pure Python, no runtime dependencies (the CLI's progress bar needs the `cli` extra).
The install helpers write into any object exposing a writable 64 KiB `.mem` (a
`PcodeVM` qualifies directly). The SID audio
model and SID-file knowledge (PSID/RSID containers, subtune headers) are deliberately
not here.

## Install

```bash
pip install -e .
```

## Use

```python
from vagrant_moses import Cia, Scheduler, Vic, c64

mem = bytearray(c64.poweron_ram())                       # 64 KiB power-on image
vec, kernal = c64.installed_vector(mem, written, (lo, hi))
entry_pc = c64.install_irq_entry(vm, vec, kernal, img=(lo, hi))

sched = Scheduler()
vic = Vic(sched, ntsc=False, irq=lambda on: ..., set_ba=lambda on: ...)
cia = Cia(sched, interrupt=lambda on: ..., new=False)
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

- [The machine model: volatile I/O, power-on RAM, vectors, and IRQ dispatch](docs/c64.md)
- [Release media: nested containers, disk geometries, tapes, and the CLI](docs/media.md)

## Develop

```bash
pip install -e ".[dev]"
black --check src tests scripts && pylint src tests scripts && pytest
```
