# Native machine

`vagrant_moses._core.Machine` is the cycle-exact C64 of `vagrant_moses.machine.Machine`
ported to Rust. The pure-Python `Machine` stays as the reference; `Machine(...,
backend="native"|"python")` picks one, native when `_core` imports and the class is
`Machine` itself (a subclass that overrides `_decode` stays Python).

## Layout

| Path | Role |
|---|---|
| `crates/moses-core` | Pure Rust: scheduler, bus, CIA, VIC-II, REU, interrupt lines, the cycle loop, the `NativeSid` log and bus-value model (`sid.rs`). No PyO3. |
| `crates/moses-py` | PyO3 extension `vagrant_moses._core`: `Machine`, `Memory`, `Registers`, `Chip`, `NativeSid`. |
| `tools/build_native.sh` | Build the extension next to the package for in-tree runs. |
| `tools/diff_machines.py` | Differential run of random programs on both backends. |
| `tools/harte_cycles.py` | Bus cycles of every opcode against SingleStepTests 6502 cases (`docs/c64.md`, Bus cycles). |
| `tools/bench_machine.py` | Playroutine-shaped and SID-write-heavy benchmark of both backends, with no SID, a Python device or a `NativeSid`. |
| `tools/thread_machines.py` | N machines on N threads against one (`--sid` for `NativeSid`). |
| `src/vagrant_moses/sid.py` | `PyNativeSid`, the pure-Python `NativeSid` the Python machine uses. |

`moses-core` runs the 6510 on `jennings-core` (`Cpu::step` over a `Bus` implementation);
it holds no CPU semantics. `jennings-core` is not on crates.io, so it is a git dependency
pinned to a release tag in both crates' `Cargo.toml`.

## Semantics

Everything is a transcription of the Python modules: `eventsched` (events are indices,
the queue is a stable sorted array), `bus`, `vicii`, `cia`, `reu` and `machine` (opcode layouts
are passed in from `Machine.LAYOUTS`, so there is one table). VIC, CIA and the scheduler
never call Python. SID devices (`read(cycle, reg)`, `write(cycle, reg, value)`) and
`on_event(cycle, name)` are called with the GIL re-acquired at those events only, after
the events buffered so far, so interleaving with SID writes is preserved. Without
`on_event`, events collect natively and `machine.events` lists them. `on_io(addr, value,
write)`, when given, is called the same way at each I/O access the CPU or a DMA makes (the
stream the trace logs, unwindowed); unset, it costs a branch.

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

## SMC recorder

`vagrant_moses.smc.Recorder(machine, image, banked=True)` arms the native machine (no
pure-Python equivalent: `smc.AVAILABLE`) with a lift-time 64 KiB `image`. It lives in
`crates/moses-core/src/smc.rs`; the machine feeds it at four points and nothing calls Python:

- every CPU bus write, interrupt/BRK push (`Core::push`) and REU DMA write to RAM;
- the entry kind at the interrupt sequence's vector choice (NMI, IRQ of any source, BRK);
- every decode, before the trap and again after a trap that moved `pc`.

At a decode the writes since the previous one are attributed, by address, to the previous
instruction (its opcode, pc, call and context) or to the entry that pushed them, when the
byte changed; a store of the value already held keeps the old writer. A change of the bank bits
makes the previous instruction the port's writer. Interrupt contexts are a stack of
`(context, frame sp)`, popped while `sp` is above the top frame: an IRQ or NMI entry opens its
own context, a BRK inherits. The entry's hardware vector bytes, then the bytes the instruction
fetches or reads as control are checked, read side-effect free under the live map (RAM under
I/O, the port registers at `$00/$01`): opcode, operands, the `JMP (ind)` pointer with its
page wrap, the `(zp),Y` and `(zp,X)` pointers, and the `RTS`/`RTI` return bytes; the bank bits
too once they differ from those at `begin()`. A decode repeating the previous one (same `pc`,
`x`, `sp` and bank bits, no write or entry since) is skipped.

A checked byte becomes an `Event(writer, writer_op, role, timing, mapping, first)`:

| Field | Values |
|---|---|
| `writer` | `cpu` (`writer_op` the storing opcode), `irq_push`, `nmi_push`, `brk_push`, `reu_dma`, `cpu_port` (`writer_op` the port store's opcode, `None` once the host set the port) |
| `role` | `opcode`, `operand`, `ptr_indirect`, `kernal_vector` (a pointer read at `$0314-$0319` by ROM code), `ret_addr`, `hw_vector`, `bank_reg` |
| `timing` | `async` (an interrupt push, or written in an IRQ/NMI context), `pre_lift` (before `begin()`), `earlier_call`, `same_insn` (by this instruction, in the main context), `same_call_earlier` |
| `mapping` | `rom_vs_ram` (read through ROM differing from the image, or through I/O: the port's writer), `ram_under_rom` (written while ROM was mapped there), `ram_under_io` (in `$D000-$DFFF`), `flat` |

A RAM byte read with no writer and differing from the image is counted as `unowned`; pushes
whose entry never reached a decode are `unattributed`. `unclassified()` returns the count and
the first 16. Events are deduplicated by a 19-bit key (writer, opcode, role, timing, mapping)
in a bitset; `first` is the call index (`begin()`, `next_call()`) or, after
`frame_clock(cycles)`, `(cycle - t0) // cycles` with `t0` the next decode's cycle.
`host_write(addr, value)` writes as the host (a call frame's return address, the port), owned
by no instruction. `deadline` stops the run at the first decode after that cycle and sets
`timed_out`; `finish()` attributes the last instruction's writes; re-arming starts afresh.

## Trace

`vagrant_moses.smc.Tracer(recorder, table, sentinel, window)` (`smc.TRACE`) adds to an
armed recorder the lift facts of a call-driven run and one record per phase
(`crates/moses-core/src/trace.rs`), fed at the recorder's decodes and at the I/O accesses the
machine makes, with no Python call per instruction. The host marks phases with `phase(kind, sp0)`
(`NONE`, `INIT`, `CALL` entered at `sp0`) from its trap.

`table` is 256 little-endian `u16`, one per opcode: address form in bits 0-3 (1-9: `zp`, `zpx`,
`zpy`, `abs`, `absx`, `absy`, `(zp,X)`, `(zp),Y`, `(ind)`), bit 4 a data read and bit 5 a store
of the form's address, bits 6-8 the register a plain store stores (A, X, Y, A&X) and bits 9-10
the bytes pushed below `sp`. Each `CALL` decode (not the `sentinel`, not after `deadline`)
records the start, stack height `sp - sp0` and bank bits per pc and the edge from the previous
decode, and resolves the instruction's page-1 data addresses (`(ind)` pointer bytes, read and
store addresses in `$0100-$01FF`, as a range per pc) and store addresses from the bytes it
fetches and RAM. The next decode (or `finish()`) values the stores: the RAM byte where the CPU
does not see I/O (and at `$00/$01`), else, outside the window, the stored register.

The trace observes and never answers: every CPU or DMA access to the I/O page at
`window` (`(lo, hi)`, raw addresses, mirrors unfolded) in a phase is logged
`(addr, value, write)` with the value the machine's chips and devices returned or took, and a
write in `CALL` adds its value to the address's stored values. A host `machine[addr]`,
`io_read` or `io_write` is not logged, nor the decode's look at an instruction's bytes.
`snap()` closes a phase: its log and the RAM bytes changed since the previous snap (the first
against RAM when the trace was armed; `idx` as little-endian `u16`, and their values). `facts()` returns `Facts(starts, edges, writes, page1, heights, banks)`,
ascending (heights by their 16-bit two's-complement value). Starts, heights, banks, stored
values and page-1 ranges are fixed-size tables; edges pass a two-entry successor cache per pc
before a set.

`Tracer(..., facts=False)` keeps the log and records but no lift facts.

## Call queue

`vagrant_moses.smc.Calls(recorder, queue, sentinel, ports, call_cycles)` (`smc.CALLS`) runs a
queue of `(phase, pc, a)` calls (`Tracer.INIT`/`Tracer.CALL`, or packed 4 bytes per call) back
to back with no Python call per call (`crates/moses-core/src/calls.rs`). The sentinel traps
natively: it closes the returning call's record (`Tracer.snap()` and the registers, when a
trace is armed), then stores `ports["init"]`/`ports["call"]` to `$01` when it differs, starts
the recorder's calls (`begin()`, then `next_call()`), pushes `sentinel - 1` as host writes,
sets A and the pc, and the recorder's deadline `call_cycles` (0: none) on. An empty queue
stops the run; `left` counts the calls not started and `close()` closes the open record as
halted `jam` or `cycles` once the run is over.

`records()` returns `Records(origin, meta, log, log_end, idx, vals, delta_end)` as
little-endian bytes: per record `meta` `(phase, halt, a, x, y, sp, p)` (halt 0 return, 1 cycles,
2 jam), its log entries `(addr u16, value, write)` ending at the `u64` `log_end`, and its RAM
changes since the previous record (the first since `origin`, the RAM the trace was armed over)
ending at `delta_end`. `entries()` returns each call's `(pc, sp)` after its return frame and
`(pc, port dir, port data)`, ascending. The snap compares RAM to the previous record a 256-byte
page at a time, copying only the pages that changed.

`vagrant_moses.smc.Scan(recorder, table, sentinel)` scans the main context of an
interrupt-driven phase from the decode after `start(frame_cycles)`: that decode's bank bits and
RAM `$0314` and `$FFFE` words, then each decode in the main context (the recorder's interrupt
contexts closed, not the `sentinel`) is counted until the first whose form address or pointer
bytes are I/O under the live bits, or which stores outside page 1 (`table` as the trace's: form
in bits 0-3, bit 4 a read and bit 5 a store of it); `result()` returns `(vector, insns, busy
frame)`.

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
python tools/vice_reu.py --programs 48 --write-fixture   # REU programs vs VICE x64sc
python tools/bench_machine.py --seconds 20
```
