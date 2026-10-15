# Migrating pysidtracker's machine code to vagrant-moses

Every C64 machine symbol pysidtracker defines, with its vagrant-moses replacement.
"stays" marks SID, SID-file, driver, oracle or survey knowledge that remains in
pysidtracker. `vm` is `vagrant_moses`.

## `pysidtracker.cia`, `.vicii`, `.eventsched`

Already re-exports of `vm.cia`, `vm.vicii`, `vm.eventsched`: import those directly
and delete the shims. The models are identical to pysidtracker's last own copies.

## `pysidtracker.roms`

| pysidtracker | vagrant-moses |
|---|---|
| `Rom`, `ROMS`, `ASSETS`, `DEFAULT_ROMS`, `DEFAULT_ROM_MIRROR` | `vm.roms` (same) |
| `KERNAL_BOOT`, `BASIC_BOOT`, `BASIC_RUN` | `vm.roms` (same) |
| `CPU_DDR`, `CPU_PORT`, `DDR_DEFAULT`, `BANK_DEFAULT` | `vm.memmap` (same) |
| `RomFetchError` (a `SidError`) | `vm.roms.RomFetchError` (a `RuntimeError`) |
| `fetch_rom`, `resolve_rom`, `apply_roms`, `rom_image`, `roms_available` | `vm.roms` (same signatures; pass `cache_dir=` explicitly) |
| `default_rom_cache(root)` | stays as a one-line wrapper for the tune-cache location, passed as `cache_dir=`; `vm.roms.default_rom_cache()` is the standalone default |
| `poweron_writes(cache_dir)` | `vm.roms.poweron_writes(cache_dir=...)`; decoder `vm.roms.decode_poweron` |
| `_download`, `_atomic_write` (from `testing`) | `vm.roms._download`, `vm.roms._atomic_write` for ROMs; `testing` keeps its own for tunes |

## `pysidtracker.poweron`

| pysidtracker | vagrant-moses |
|---|---|
| `ram_pattern()` | `bytearray(vm.c64.poweron_ram())` |
| `SID_PORT`, `SID_PORT_SIZE` | `vm.memmap.SID_BASE`, `SID_LAST + 1 - SID_BASE` |
| `PSID_DRIVER_CLEARED`, `power_on_ram`, `psid_init_bank` | stay (psiddrv) |

## `pysidtracker.emu`

| pysidtracker | vagrant-moses |
|---|---|
| `_BankedMemory(subject, roms)` | `vm.bus.Bus(ram, roms)`; the port follows `$00` as well as `$01`, so seed `ram[0] = DDR_DEFAULT` |
| `access_cycle` | `vm.cpu.access_cycle` |
| `run_to_rts` | `vm.cpu.run_to_rts` |
| `wire_mpu(subject, sid_model, ntsc)` | `vm.cpu.flat_mpu(subject, ntsc=ntsc)` plus pysidtracker's SID subscriptions (stay) |
| `SidReads`, `observe_data_access` | stay (reSIDfp chip; jennings engine hook) |

## `pysidtracker.irqhost`

| pysidtracker | vagrant-moses |
|---|---|
| `W`, `PUSHPCH`, `LOWPC`, `PUSHSR`, `R`, `X2`, `F`, `CLI`, `SEI`, `POPSR`, `BR`, `BRK0`, `VECLO`, `VECHI`, `FN`, `RTI`, `HALT`, `ANE`, `BRK_ENTRIES` | `vm.machine` (same) |
| `_FETCHES` | `vm.machine.FETCHES` |
| `opcode_layout`, `boot_cycle` | `vm.machine` (same) |
| `_KERNAL_CALLS`, `kernal_stub`, `KERNAL_VECTORS` | `vm.roms.KERNAL_CALLS`, `kernal_stub`, `KERNAL_VECTORS` |
| `_RAISE` | `vm.machine.SOURCES` |
| `ram_pattern` (re-export) | `vm.c64.poweron_ram` |
| `C64(ntsc, sid_models, extra_sids, cia_new)` | `vm.machine.Machine(ntsc, sids, cia_new, on_event)`, see below |
| `C64.boot` | stays as a driver installer writing `machine.ram`, then `machine.reset(labels["cold"])` |
| `_DRIVER`, `_resolve_ntsc`, `to_rows`, `InterruptHost`, `InterruptPlayer`, `host_cadence`, `_most_raised` | stay |

`C64` members: `port_dir`, `port_data`, `ram`, `io`, `bits`, `sched`, `vic`, `cia1`,
`cia2`, `mpu`, `t`, `jammed`, `run`, `colour`, `seed` and the CPU-state fields keep
their names. `view`/`romview` become `Bus.pages`; `_map` → `remap`; `_port_read` →
`vm.bus.port_read`; `_io_read`/`_io_write` → `io_read`/`io_write`; `_sid_chip` →
`_sid`. `chips`, `sid_slots`, `lastpoke` and `rows` move to pysidtracker: each SID is
a device wrapping `SidReads` that logs `(cycle, chip, reg, value)` on a changed value
and flushes past `0xFFFF` queued writes, and `on_event=lambda c, n: rows.append((c,
None, None, n))` logs the interrupt rows into the same list:

```python
class _Sid:
    def __init__(self, rows, chip, model):
        self.rows, self.chip, self.reads, self.last = rows, chip, SidReads(model), bytearray(32)

    def read(self, cycle, register):
        return self.reads.read(cycle, register)

    def write(self, cycle, register, value):
        if value != self.last[register]:
            self.rows.append((cycle, self.chip, register, value))
            self.last[register] = value
        self.reads.write(cycle, register, value)
        if self.reads.pending() > 0xFFFF:
            self.reads.flush(cycle)

rows = []
sids = {0xD400: _Sid(rows, 0, models[0])}
sids.update({a & 0xFFE0: _Sid(rows, i, models[i]) for i, a in enumerate(extra, 1)})
machine = Machine(ntsc, sids, cia_new, lambda c, n: rows.append((c, None, None, n)))
```

With that adapter and `boot` ending in `machine.reset(labels["cold"])`, pysidtracker's
`test_irqhost.py`, `test_irqhost_oracle.py` and `test_host_check.py` pass unchanged,
every recorded sidtrace fixture row for row.

## `pysidtracker.kernalhost`

| pysidtracker | vagrant-moses |
|---|---|
| `sys_addr(ram_or_image, base)` | `vm.kernal.sys_addr(mem, base)`; pass `image.mem` for a `SidImage`; raises `ValueError` |
| `KernalHostError` | `vm.kernal.KernalHostError` (a `RuntimeError`) |
| `ABI` | `vm.kernal.JUMP_TABLE` |
| `KernalHost(host_dir, ntsc, kernal_rom, basic_rom)` | `vm.kernal.KernalHost(host_dir, roms={0xE000: kernal, 0xA000: basic})`, or `roms=vm.roms.load_roms()` |
| `host.ram()` | `host.ram` |
| `load_prg(SidImage)` | `load_prg(prg_bytes)` |
| `run_to_unpacked(...) -> SidImage` | returns `(entry, bytes)`; wrap with `SidImage._place(data, entry, None, b"")` |
| `tap_key`, `screen`, `run`, `touched`, `mpu` | same |

The trapped ABI changes: pysidtracker serviced invented addresses (`$FFC6` as CHROUT,
`$FFC9` as CHRIN, `$FFCC` as GETIN, `$FFCF` as SETLFS, `$FFD2` as SETNAM) and register
layouts, and its files had no PRG header. `vm.kernal` services the real jump table
and ABI, so programs written against the old layout (pysidtracker's
`test_kernalhost.py`) must use the real entries.

## `pysidtracker.registers`

| pysidtracker | vagrant-moses |
|---|---|
| `PAL_CYCLES_PER_FRAME`, `NTSC_CYCLES_PER_FRAME` | `vm.memmap.PAL.cycles_per_frame`, `NTSC.cycles_per_frame` |
| `PAL_CLOCK_HZ`, `NTSC_CLOCK_HZ` | `vm.memmap.PAL.clock_hz`, `NTSC.clock_hz` |
| `PAL_CYCLES_PER_LINE`, `NTSC_CYCLES_PER_LINE` | `vm.memmap.PAL.cycles_per_line`, `NTSC.cycles_per_line` |
| `KERNAL_CIA1_LATCH_PAL`, `KERNAL_CIA1_LATCH_NTSC` | `vm.memmap.PAL.kernal_latch`, `NTSC.kernal_latch` |
| `CIA1_*`, `CIA2_*`, `VIC_BASE`, `VIC_LAST`, `VIC_CONTROL_1`, `VIC_RASTER`, `VIC_IRQ_STATUS`, `VIC_IRQ_ENABLE` | `vm.memmap` (same) |
| `RAM_IRQ_VECTOR_LO/HI`, `RAM_NMI_VECTOR_LO/HI`, `CPU_NMI_VECTOR`, `CPU_RESET_VECTOR`, `CPU_IRQ_VECTOR` | `vm.memmap` (same) |
| `SID_BASE`, `SID_LAST`, `SID_STRIDE` | `vm.memmap` (same; the I/O window) |
| `is_cia1_reg`, `is_cia2_reg`, `is_cia_timer`, `is_vic_reg`, `is_irq_vector`, `is_nmi_vector`, `is_cpu_vector` | `vm.memmap` (same) |
| SID register indices and masks, `sid_register`, `is_sid_reg`, `is_sid_control`, `cycles_per_frame_for_flags`, `sid_model_for_flags`, `RegisterStore`, `find_register_stores`, store opcodes, `sid_write_band`, `attack_decay`, `sustain_release` | stay |

## `pysidtracker.cadence` and `.trace`

| pysidtracker | vagrant-moses |
|---|---|
| `cadence._Machine.take` vector choice (`ram[1] & 2`, push A/X/Y, `$0314`/`$0318` or `$FFFE`/`$FFFA`) | `vm.c64.dispatch_vector(bits, nmi)` |
| `cadence._Machine._now` | `vm.cpu.access_cycle(mpu)` |
| `cadence._KERNAL_ROM` | `vm.memmap.KERNAL_BASE` |
| `trace._WATCH_ADDRS` | build from `vm.memmap` windows plus `vm.memmap.VECTORS` |
| `trace._raster` | `vm.memmap.raster_compare` |
| `cadence._Cia`, `_next_match` | stay: the analytic fast path; the exact models are `vm.Cia`/`vm.Vic` |
| `TriggerSource`, `Cadence`, `playroutine_cadence`, `cadence_from_latch`, `_iomap`, `_driver_base`, `_kernal_exit`, `_Machine.driver`, `InitTrace`, `trace_init` | stay (playroutine, psiddrv) |
