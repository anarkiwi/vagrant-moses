# Native machine

`vagrant_moses._core.Machine` is the cycle-exact C64 of `vagrant_moses.machine.Machine`
ported to Rust. The pure-Python `Machine` stays as the reference; `Machine(...,
backend="native"|"python")` picks one, native when `_core` imports and the class is
`Machine` itself (a subclass that overrides `_decode` stays Python).

## Layout

| Path | Role |
|---|---|
| `crates/moses-core` | Pure Rust: scheduler, bus, CIA, VIC-II, interrupt lines, the cycle loop. No PyO3. |
| `crates/moses-py` | PyO3 extension `vagrant_moses._core`: `Machine`, `Memory`, `Registers`, `Chip`. |
| `tools/build_native.sh` | Build the extension next to the package for in-tree runs. |
| `tools/diff_machines.py` | Differential run of random programs on both backends. |
| `tools/bench_machine.py` | Playroutine-shaped benchmark of both backends. |

`moses-core` runs the 6510 on `jennings-core` (`Cpu::step` over a `Bus` implementation);
it holds no CPU semantics. `jennings-core` is not on crates.io, so it is a git dependency
pinned to a release tag in both crates' `Cargo.toml`.

## Semantics

Everything is a transcription of the Python modules: `eventsched` (events are indices,
the queue is a stable sorted array), `bus`, `vicii`, `cia` and `machine` (opcode layouts
are passed in from `Machine.LAYOUTS`, so there is one table). VIC, CIA and the scheduler
never call Python. SID devices (`read(cycle, reg)`, `write(cycle, reg, value)`) and
`on_event(cycle, name)` are called with the GIL re-acquired at those events only, after
the events buffered so far, so interleaving with SID writes is preserved. Without
`on_event`, events collect natively and `machine.events` lists them.

`run(limit)` releases the GIL. While it runs, other threads get `RuntimeError` from any
attribute access; independent machines run in parallel.

## Hooks

- `set_trap(callback, watch)`: before decoding, while `pc` is in `watch` (a set of
  addresses or `machine.WATCH_ALL`), call `callback(machine)`; a true return means the
  callback ran a call and moved `pc`, and the fetch is checked again. The callback may
  read and write RAM, registers (`machine.mpu`), `machine[...]` and `machine.watch`.
- `arm_at(idle_pc)`: nothing is watched until the CPU first decodes at `idle_pc`, then
  `WATCH_ALL`.
- `on_port_write(callback)`: once armed, called with the machine after an instruction
  that wrote `$00`/`$01` (a host `machine[0 or 1] = v` calls it at once). It is called
  once per instruction, not once per write.
- `set_decode_hook(callback | None)`: called before every decode, before the trap.
- Any hook may set `machine.stop = True`; `run` returns after the cycle that decoded.
  An exception from a hook, device or sink ends the run and is raised from `run`.

## Known difference

The Python machine runs jennings' Python engine, which re-reads an instruction's bytes
several times by its cache state. The native machine reads each byte once. The two agree
except when an instruction is fetched from a register whose read has side effects (the
open bus at `$DE00-$DFFF`, the CIA ICR); `tests/machine_diff.py` ends its comparison at
the first such fetch.

## Verification

`pytest` runs `tests/test_machine.py` on both backends, `tests/test_native.py` for the
Python surface and hooks, and six differential programs. For the full gate:

```bash
python tools/diff_machines.py --programs 500 --cycles 2000000
python tools/bench_machine.py --seconds 20
```
