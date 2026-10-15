# vagrant-moses

A standalone C64 machine model: the power-on RAM, the installed-interrupt-vector
discovery, and the ROM-free IRQ dispatch path a host needs to enter a playroutine's
interrupts faithfully.

Pure Python, no runtime dependencies. The install helpers write into any object
exposing a writable 64 KiB `.mem` (a `PcodeVM` qualifies directly). SID-file
knowledge (PSID/RSID containers, subtune headers) is intentionally not here.

## Install

```bash
pip install -e .
```

## Use

```python
from vagrant_moses import c64

mem = bytearray(c64.poweron_ram())              # 64 KiB power-on image
vec, kernal = c64.installed_vector(mem, written, (lo, hi))
entry_pc = c64.install_irq_entry(vm, vec, kernal, img=(lo, hi))
```

## Docs

- [C64 model and its contract](docs/c64.md)

## Develop

```bash
pip install -e ".[dev]"
black --check src tests && pylint src tests && pytest
```
