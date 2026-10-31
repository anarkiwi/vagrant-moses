# Native machine

`vagrant_moses._core.Machine` is the cycle-exact C64 of `vagrant_moses.machine.Machine`
ported to Rust. The pure-Python `Machine` stays as the reference; `Machine(...,
backend="native"|"python")` picks one, native when `_core` imports and the class is
`Machine` itself (a subclass that overrides `_decode` stays Python).

## Layout

| Path | Role |
|---|---|
| `crates/moses-core` | Pure Rust: scheduler, bus, CIA, VIC-II, interrupt lines, the cycle loop, the `NativeSid` log and bus-value model (`sid.rs`). No PyO3. |
| `crates/moses-py` | PyO3 extension `vagrant_moses._core`: `Machine`, `Memory`, `Registers`, `Chip`, `NativeSid`. |
| `tools/build_native.sh` | Build the extension next to the package for in-tree runs. |
| `tools/diff_machines.py` | Differential run of random programs on both backends. |
| `tools/bench_machine.py` | Playroutine-shaped and SID-write-heavy benchmark of both backends, with no SID, a Python device or a `NativeSid`. |
| `tools/thread_machines.py` | N machines on N threads against one (`--sid` for `NativeSid`). |
| `src/vagrant_moses/sid.py` | `PyNativeSid`, the pure-Python `NativeSid` the Python machine uses. |

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

## NativeSid

`NativeSid(model, on_read)` (`"6581"` or `"8580"`) in `Machine(sids={io_base: dev})` keeps a
SID's traffic out of Python. The chip stays in Python, behind `on_read`.

- `write(cycle, reg, value)` is logged natively. `drain_writes()` returns every write since
  the last drain as `(cycle, reg, value)`; `drain_rows()` returns those that changed
  their register, as `sidtrace` logs them (a write of the value `last[reg]` already holds is
  omitted; `last` starts as `bytes(32)`). Both clear. Registers are the 5-bit offset, so the
  `$D400-$D7FF` mirrors land on the same log.
- A read of `$19-$1C` calls `on_read(cycle, reg)` with the GIL re-acquired, after the events
  buffered so far. The callback drains the log into its chip, clocks it to `cycle` and
  returns the byte; the byte also drives the bus value.
- Any other read (`$00-$18`, `$1D-$1F`) is the bus value, with no Python call. The rule is
  libresidfp's (`SID.cpp`, `SID.h`): a write or a readable-register read sets the value and
  a time to live of `0x1D00` cycles (6581) or `0xA2000` (8580); reading any other register
  halves the time left; the value reads 0 once the time has run out. The deadline is aged
  to the access cycle, not by clocking a chip.
- The pending-write queue is the machine's: an RMW instruction's dummy write precedes its
  final write in the log, as for a Python device.

The native machine reads and writes the state without a lock. While `run` is going, the
machine claims each `NativeSid` it holds; another thread touching one gets `RuntimeError`, as
for the machine itself, and `on_read` (on the running thread) may call its methods. The GIL
is retaken for a `NativeSid` only inside `on_read`, so machines whose chips are never asked
run in parallel. The Python machine takes the same object, and `vagrant_moses.sid.PyNativeSid`
is the equivalent for builds without `_core`; `vagrant_moses.machine.NativeSid` is whichever
applies.

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
for code in RAM and ROM, and for data accesses anywhere. They differ only for an
instruction fetched from `$D000-$DFFF` with I/O mapped, where a re-read has an effect: the
open bus advances, a CIA ICR read clears, and the last value read (which an RMW
instruction writes back first) is the last byte re-read. `tests/machine_diff.py` ends its
comparison at the first such fetch.

## Verification

`pytest` runs `tests/test_machine.py` on both backends and with each kind of SID,
`tests/test_native.py` for the Python surface and hooks, `tests/test_sid.py` for `NativeSid`,
and six differential programs per SID kind. For the full gate:

```bash
python tools/diff_machines.py --programs 500 --cycles 2000000 --sid both
python tools/thread_machines.py --threads 8 --sid
python tools/bench_machine.py --seconds 20
```
