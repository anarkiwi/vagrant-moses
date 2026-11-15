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
| `tools/bench_forks.py` | `run_forks` forks/s and emulated frames/s by thread count. |
| `src/vagrant_moses/sid.py` | `PyNativeSid`, the pure-Python `NativeSid` the Python machine uses. |

`moses-core` runs the 6510 on `jennings-core` (`Cpu::step` over a `Bus` implementation);
it holds no CPU semantics. `jennings-core` is not on crates.io, so it is a git dependency
pinned in both crates' `Cargo.toml` to a jennings release tag.

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

The machine, `NativeSid` and the views (`ram`, `mpu`, chips, `reu`) take part in Python's
cyclic GC: they visit the devices, callbacks and hooks they hold, and a machine clears its
hooks and views to break a cycle, so a machine whose hooks reach back to their owner is
collected. A numpy array over a view is not visited (numpy arrays are not GC containers): an
owner holding one and handing the machine a hook bound to itself is a cycle the GC cannot
see, so such a hook holds its owner weakly.

The chip views (`machine.vic`, `cia1`, `cia2`) read the registers, `raster_y`,
`line_cycle`, `irq_flags`, `irq_mask`, `timer(n)` and `icr_mask`, and carry the `seed_*`
setters; `keymatrix`, `set_key` and `release_all` live on the machine. They behave as the
Python machine's (`docs/c64.md`, Keyboard matrix and chip-state seeding); a CIA call on the
native VIC view, or the reverse, raises `ValueError` (the Python chips lack the method).

## NativeSid

`NativeSid(model, on_read, rows=True)` (`"6581"` or `"8580"`) in `Machine(sids={io_base: dev})` keeps a
SID's traffic out of Python. The chip stays in Python, behind `on_read`.

- `write(cycle, reg, value)` is logged natively. `drain_writes()` returns every write since
  the last drain as `(cycle, reg, value)`; `drain_rows()` returns those that changed
  their register, as `sidtrace` logs them (a write of the value `last[reg]` already holds is
  omitted; `last` starts as `bytes(32)`; with `rows=False` none are kept). Both clear;
  `drain_packed()` is `drain_writes()` as little-endian `i64` triples. Registers are the 5-bit offset, so the
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

## Clone

`clone()` forks the machine: the copy, run for any number of cycles, behaves exactly as the
original run for the same cycles. `Core::fork` copies everything the core models (the CPU,
RAM, banking, colour RAM, VIC-II, both CIAs, interrupt lines, scheduler, pending I/O
writes, open-bus seed, last read, REU, collected events) as plain Rust values; the opcode
layouts are shared. Nothing goes through Python, so a fork is a few memory copies (the
64 KiB RAM and 1 KiB colour RAM, plus the REU's RAM and the events collected so far).

The BASIC, CHARGEN and KERNAL images are shared between a machine and its forks (`Arc`) and
copied on write: a ROM view (`machine.pages`) unshares the image it is taken of before handing
out its memory and pins the machine, whose later forks then copy the images rather than share
them, so a write through a view the caller holds never reaches another machine. RAM is copied:
`machine.ram` is one contiguous zero-copy buffer, which sharing pages would break.

- Each `NativeSid` is copied into a new `NativeSid` with the same bus value, `last` and
  `on_read`, and empty write logs; a Python SID device object is shared by both machines.
- The clone has no `on_event` sink (its events collect in `events`, which starts as a copy
  of the original's), no `on_io`, no trap, decode or port hook, and fresh views.
- It is refused with `RuntimeError` while the machine runs (also from its own hooks or
  sink) or while the SMC recorder is armed.

A clone is idle and unclaimed, so clones run in parallel threads.

## Driver programs

`vagrant_moses.program.Program(budget, marker=None, max_ops=1 << 24)` is a program for a
register machine that drives the keyboard and runs the machine to program counters
(`crates/moses-core/src/prog.rs`; `program.py` is the Python reference). `machine.run_program(p)`
runs it on either backend and returns `Outcome(status, code, cycle, frames, outputs)`; the
native machine runs it inside `Core::run` with the GIL released and no Python call per op.
The builder methods append one op each and return the program; `label(name)` names the next
op as a branch target; `code` is the validated `(name, *operands)` tuple with labels resolved,
`native()` the native machine's compiled form (cached, reusable on any machine).

There are sixteen `u32` registers `R[0]`-`R[15]`, all 0 at the start. A source operand `s` is a
register or a `u32` immediate. `budget` caps the cycles from the program's start: every run op
runs to at most `start + budget`. `marker` is a pc whose arrivals every run op counts (a
once-per-frame pc makes `FRAMES` a frame counter). `max_ops` caps the ops executed.

| Op (builder) | Effect |
|---|---|
| `KEYS rows` (`keys`), `KEYS_CLEAR` (`clear`) | CIA 1's matrix = the 8 row bytes (as `keymatrix`); all released |
| `KEY row col down` (`key`) | hold or release the key at PA line `row`, PB line `col` |
| `KEYR r down` (`keyr`) | the same for key `r & 63` = `row * 8 + col` |
| `RUN_TO pcs [leave]` (`run_to`) | run until the CPU is about to execute a pc in `pcs`; nothing when parked at one already, unless `leave` |
| `COUNT_TO pc n` (`count_to`) | run until the CPU has been about to execute `pc` `n` more times (`n` = 0: nothing) |
| `STEP n` (`step`) | run `n` decodes on: `n` instructions (an interrupt entry is not one; its handler's first instruction is) |
| `LD r addr [idx]` (`ld`) | `r` = the byte at `addr + idx` (16-bit wrap) as the CPU sees it under the live banking, with no side effect; the I/O page as `LDIO` |
| `LDR r addr [idx]` (`ldr`) | `r` = the RAM byte, whatever the banking |
| `LDIO r addr [idx]` (`ldio`) | `r` = the I/O register as a CPU read now would return it, whatever the banking, read on copies of the chip, scheduler and interrupt lines so nothing changes: VIC-II `$D000-$D3FF`, colour RAM `$D800-$DBFF`, CIAs `$DC00-$DDFF` |
| `MOV`/`LDI` `ADD SUB AND OR XOR SHL SHR` `d s` | `d = s`, `d op s` mod 2^32 (shifts by `s & 31`) |
| `ADD8 SUB8 d s` | `(d ± s) & 0xFF` |
| `CYC lo [hi]` (`cyc`) | `hi` = the cycle's high 32 bits, then `lo` = its low 32 bits |
| `FRAMES r` (`frames`) | `r` = marker arrivals counted so far |
| `BEQ BNE BLT BGE a s label` | branch on `a == s`, `!=`, `<`, `>=` (unsigned) |
| `BMI8 BPL8 r label` | branch when bit 7 of `r` is set / clear (after `SUB8 d, cur`: the sign of `target - cur` on a wrapping 8-bit compass) |
| `JMP label`, `DJNZ r label` | jump; `r -= 1` and jump unless it is now 0 |
| `OUT r`, `OUTMEM addr len` | append `r`; append `len` bytes as `LD` reads them |
| `HALT s` | end with status `halt`, code `s`; running off the end is `HALT 0` |

A run op is a stop through the trap machinery: the op's pcs (and the marker) are the watch set,
the program's own native trap counts each decode at one of them, and the decode that completes
the op sets `stop`, so the run returns after that cycle with the instruction at `cpu.pc` decoded
and not executed (VICE checkpoint semantics). The machine is then parked at that pc; any later
run that advances the clock unparks it. `n` of `COUNT_TO` and `STEP` may be a register.

`status` is `halt` (`code` the `HALT` code), `budget` (a run op reached `start + budget`;
`code` its op index), `jam` (the CPU jammed; `code` the run op's index), `ops` (`max_ops`
executed; `code` the next op's index) or `fault` (an `LD`, `LDIO` or `OUTMEM` met an address
with no side-effect-free view, the SID and expansion pages under I/O; `code` the address).
`cycle` is the machine's cycle at the end, `frames` the marker count, `outputs` the values
appended, also those before a fault. The trap, decode and port hooks and `arm_at` are set
aside while the program runs and restored after; `on_event` and `on_io` keep being called.
A SID callback's exception ends the program and is raised. `run_program` is refused while the
machine runs or the SMC recorder is armed.

## Batched forks

`vagrant_moses.run_forks(base, programs, threads=None, keep=False)` (`base.run_forks(...)`)
runs each program on its own fork of the idle `base` and returns the outcomes in program
order; with `keep`, `(outcomes, forks)`, each fork a machine as `clone()` makes it, left where
its program ended. The native machine claims `base`, releases the GIL for the whole batch and
runs the forks on `threads` scoped threads (default: one per core), each taking the next program
from a shared counter, forking `base` itself (forks only read it) and dropping the fork after
its program unless `keep`. Each fork's result depends only on `base` and its program, so the
outcomes do not depend on the thread count or the scheduling. The SIDs must be `NativeSid`s
(a shared Python device would see the forks' writes interleaved): their `on_read` is called
with the GIL re-acquired and is shared by the forks, so it must not depend on its call order.
The Python machine runs the forks in turn on clones, the reference the native batch equals.
`tools/bench_forks.py` reports forks/s and emulated frames/s by thread count.

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
- Any hook may set `machine.stop = True`; `run` returns after the cycle that decoded (the
  Python machine's `run` honours `stop` too).
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

`vagrant_moses.smc.Calls(recorder, queue, sentinel, ports, call_cycles, times, budget)` (`smc.CALLS`) runs a
queue of `(phase, pc, a)` calls (`Tracer.INIT`/`Tracer.CALL`, or packed 4 bytes per call) back
to back with no Python call per call (`crates/moses-core/src/calls.rs`). The sentinel traps
natively: it closes the returning call's record (`Tracer.snap()` and the registers, when a
trace is armed), then stores `ports["init"]`/`ports["call"]` to `$01` when it differs, starts
the recorder's calls (`begin()`, then `next_call()`), pushes `sentinel - 1` as host writes,
sets A and the pc, and starts the unit's [budget](#budget-and-outcome) (`call_cycles` is `budget.unit_cycles` when that is 0). An empty queue
stops the run; `left` counts the calls not started and `close()` closes the open record as
halted `jam` or `cycles` once the run is over. `times` (default: once each) makes each queued call that many
calls back to back, so a long run of one call is queued in constant memory.

`records()` returns `Records(origin, meta, log, log_end, idx, vals, delta_end)` as
little-endian bytes: per record `meta` `(phase, halt, a, x, y, sp, p)` (halt 0 return, 1 cycles,
2 jam), its log entries `(addr u16, value, write)` ending at the `u64` `log_end`, and its RAM
changes since the previous record (the first since `origin`, the RAM the trace was armed over)
ending at `delta_end`; `events` holds the interrupts taken inside the records ([below](#interrupt-frames)). `entries()` returns each call's `(pc, sp)` after its return frame and
`(pc, port dir, port data)`, ascending. `take()` returns `records()` and empties them, the next record's
changes then against the `origin` of the next take (the RAM the last record taken left): a
run made `Machine.run` window by window, taking the records after each, holds one window's
records and gives the records of one run. The snap compares RAM to the previous record a 256-byte
page at a time, copying only the pages that changed.

An interrupt entered inside a call (the call's phase `CALL`) is part of the call's record: its
handler returns through the sentinel, whose trap resumes the interrupted code (the call's own
return, an `RTS` to the sentinel, is told apart by the open interrupt contexts), and it is an
event of the record ([events](#interrupt-frames)); the handler's entry joins `entries()`. BRK
runs inside the context it is taken in; interrupts in an init call are not followed.
`take()` returns the events with the records.

## Interrupt frames

`vagrant_moses.smc.Frames(recorder, sentinel, budget)` (`smc.FRAMES`; the recorder needs a
`Tracer` over the same sentinel) records an interrupt-driven song's frames natively
(`crates/moses-core/src/frames.rs`, hooked at the interrupt entry in `Core::special`): one
record per interrupt, from its entry to its RTI, in `Calls`' packed form with phase `CALL`.

An entry (NMI, or an IRQ from the video chip, a CIA or the REU; BRK is no frame) outside a
frame opens a record: the tracer is synced (RAM that changed outside page 1 since the previous
frame closed sets `dirty`), the origin of the first record of a chunk is the RAM at the entry,
the pushed return address is rewritten to the sentinel through the host-write path, and the
trace phase is `CALL` at the stack pointer after the pushes. The handler's RTI lands on the
sentinel, whose trap pops the innermost context and resumes the interrupted pc; the outermost
pop closes the record (halt return, the machine's registers). An interrupt taken right after
that RTI, before the sentinel is decoded, finds the sentinel as its pushed return: it closes
the frame (registers before the interrupt) and opens the next, whose interrupted pc is the
closed frame's resume pc. Inside a frame an interrupt is an event of the record and a nested
context (`depth` contexts open above the body); a BRK outside a frame sets bit 4 of `foreign`.
RAM changes are one net delta per record; the log holds the device accesses in machine order,
nested handlers' included.

`take()` returns `Chunk(records, entries, events)` for the closed frames and empties them
(the next chunk's changes are against its own `origin`); `close()` stops recording (the closed frames stay to be taken). `open` says a frame is open, `closed` counts the frames closed,
`image` is the RAM at the first entry, `dirty` and `foreign` as above, `entries()` the handler
entries `(pc, sp)` and `(pc, port dir, port data)`. A frame still open at `close()` is closed as
halted (`jam`, else `cycles`) when the CPU jammed or a limit stopped the run, else dropped.

The exact boundary: each context counts the instructions it retires (decodes, not the
sentinel's and not interrupt-entry sequences; BRK and its handler count in the context they
run in). An interrupt is taken at an instruction boundary, so an event's `icount` is the count
of the interrupted context when it was taken; a nested handler counts only its own.

| bytes | fields | dtype (`smc.*_FIELDS`) |
| --- | --- | --- |
| event, 21 | `record` u4 (index in the chunk), `icount` u4, `pos` u4 (device accesses logged in the record before the entry, nested handlers' included), `at` u2 (handler entry), `a x y sp p` u1 (before the interrupt; `sp` after the pushes plus 3, `p` as pushed), `depth` u1 (0: taken in the body), `kind` u1 | `EVENT_FIELDS` |
| entry, 18 | `at` u2, `pc` u2 (interrupted), `a x y sp p` u1, `kind` u1, `insns` u4, `cycles` u4 (of the whole record) | `ENTRY_FIELDS` |

`kind` indexes `smc.SOURCES` (`nmi`, `video_irq`, `cia_irq`, `reu_irq`). All little-endian.

## Budget and outcome

`smc.Budget(unit_cycles, unit_insns, song_cycles, song_insns)` (0: none) limits each unit (a
call or a frame) and the whole run (the sum of its units), counting cycles from the machine's
clock and instructions from the decode counter. A unit may run exactly its limit; the next
decode that would exceed it stops the run before it executes. `Calls.outcome` and
`Frames.outcome` index `smc.STOPS` (`none`, `unit_cycles`, `unit_insns`, `song_cycles`,
`song_insns`, `jam`): the first limit that stopped the run, or `jam` when the CPU jammed. The
open unit closes as halted `cycles` (`jam` on a jam); a frame cut by a limit has its cycles
counted to the decode that hit it. The caller chooses the limits.

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
`tests/test_clone.py` for `clone()` on both backends, `tests/test_program.py` for driver
programs and `run_forks` on both backends, and six differential programs per SID kind.
`tools/diff_machines.py` also clones each program halfway on both backends and checks the clone
ends as the original, and runs random driver programs alone and as forks on both backends. For the full gate:

```bash
python tools/diff_machines.py --programs 500 --cycles 2000000 --sid both
python tools/thread_machines.py --threads 8 --sid
python tools/vice_reu.py --programs 48 --write-fixture   # REU programs vs VICE x64sc
python tools/bench_machine.py --seconds 20
python tools/bench_forks.py --forks 4000 --frames 2
```
