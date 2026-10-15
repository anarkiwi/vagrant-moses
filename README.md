# vagrant-moses

A standalone C64 machine model: the volatile I/O models a host needs to drive a
playroutine's interrupts faithfully — the VIC-II raster/badline/sprite-BA state and
the two CIA timers over a shared two-phase event scheduler — plus the power-on RAM,
the installed-interrupt-vector discovery, and the ROM-free IRQ dispatch path.

Pure Python, no runtime dependencies. The install helpers write into any object
exposing a writable 64 KiB `.mem` (a `PcodeVM` qualifies directly). The SID audio
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

## Docs

- [The machine model: volatile I/O, power-on RAM, vectors, and IRQ dispatch](docs/c64.md)

## Develop

```bash
pip install -e ".[dev]"
black --check src tests && pylint src tests && pytest
```
