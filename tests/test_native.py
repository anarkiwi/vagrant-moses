"""The native machine's Python surface: memory views, hooks, errors and threads."""

from __future__ import annotations

import threading
import time

import pytest

from vagrant_moses import machine as vm
from vagrant_moses import memmap as mm
from vagrant_moses.c64 import poweron_ram
from vagrant_moses.machine import Machine

import machine_diff
from asm import assemble

pytest.importorskip("jennings")
pytestmark = pytest.mark.skipif(not vm.NATIVE, reason="vagrant_moses._core not built")

ORG = 0x1000


def _machine(src, **kwargs):
    machine = Machine(backend="native", **kwargs)
    code, _ = assemble(src, ORG)
    machine.ram[ORG : ORG + len(code)] = code
    return machine, machine.reset(ORG)


def test_default_backend_is_native_and_subclasses_stay_python():
    assert type(Machine()).__module__ == "vagrant_moses._core"
    assert type(Machine(backend="python")) is Machine

    class Sub(Machine):
        """A subclass overriding internals runs on the Python machine."""

    assert isinstance(Sub(), Machine)
    with pytest.raises(ValueError):
        Machine(backend="fortran")


def test_ram_is_a_zero_copy_bytearray_like():
    machine = Machine(backend="native")
    ram = machine.ram
    assert len(ram) == 0x10000 and bytes(ram[:0x200]) == poweron_ram()[:0x200]
    view = memoryview(ram)
    ram[0x2000:0x2004] = b"\x01\x02\x03\x04"
    assert view[0x2000] == 1 and ram[0x2003] == 4 and ram[-1] == poweron_ram()[-1]
    view[0x2001] = 9
    assert ram[0x2001] == 9
    ram[0x3000:0x3006:2] = [7, 8, 9]
    assert list(ram[0x3000:0x3006:2]) == [7, 8, 9]
    with pytest.raises(ValueError):
        ram[0:2] = b"abc"
    with pytest.raises(IndexError):
        _ = ram[0x10000]
    with pytest.raises(ValueError):
        ram[0] = 256
    machine.colour[3] = 0x0F
    assert len(machine.colour) == 0x400 and machine[mm.COLOUR_RAM + 3] == 0x0F


def test_bus_access_matches_the_python_machine():
    native, python = Machine(backend="native"), Machine(backend="python")
    for machine in (native, python):
        machine[1] = 0x35
        machine[0xD020] = 0x07
        machine[0xC000] = 0x55
    assert (native.port_data, native.port_dir, native.io, native.bits) == (
        python.port_data,
        python.port_dir,
        python.io,
        python.bits,
    )
    for addr in (0, 1, 0xA000, 0xC000, 0xD000, 0xD012, 0xD41B, 0xDC0D, 0xDE00, 0xE000):
        assert native[addr] == python[addr], hex(addr)


def test_registers_read_and_write():
    machine = Machine(backend="native")
    mpu = machine.mpu
    mpu.pc, mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p = 0x1234, 1, 2, 3, 4, 0x85
    assert (mpu.pc, mpu.a, mpu.x, mpu.y, mpu.sp) == (0x1234, 1, 2, 3, 4)
    assert mpu.p == 0xB5  # B and bit 5 always read set
    mpu.a = 0x1FF
    assert mpu.a == 0xFF


def test_chip_views_follow_the_run():
    machine, start = _machine(
        "  LDA #$34\n  STA $DC04\n  LDA #$12\n  STA $DC05\nl: JMP l\n"
    )
    machine.run(start + 200)
    assert machine.cia1.timer(0)[1] == 0x1234
    assert machine.cia1.regs[4] == 0x34 and len(machine.vic.regs) == 0x40
    assert 0 <= machine.vic.raster_y < 312 and machine.vic.line_cycle < 63
    with pytest.raises(ValueError):
        machine.vic.timer(0)


def test_trap_sees_a_fetch_and_may_redirect_it():
    src = "  LDA #$01\n  STA $C000\n  LDA #$02\n  STA $C001\nl: JMP l\n"
    machine, start = _machine(src)
    calls = []

    def trap(m):
        calls.append((m.mpu.pc, m.t))
        if m.mpu.pc == ORG + 5:
            m.mpu.pc = ORG + 7  # skip the second load
            return True
        return False

    machine.set_trap(trap, {ORG + 5, ORG + 7})
    machine.run(start + 100)
    assert [pc for pc, _ in calls] == [ORG + 5, ORG + 7]
    assert machine.ram[0xC000] == 1 and machine.ram[0xC001] == 1


def test_arm_at_and_port_writes_rebuild_the_watch():
    src = "idle: JMP idle\n"
    machine, start = _machine(src)
    seen, ports = [], []

    def trap(m):
        seen.append(m.mpu.pc)
        m.watch = frozenset({ORG})
        return False

    machine.set_trap(trap, frozenset())
    machine.arm_at(ORG)
    machine.on_port_write(
        lambda m: ports.append(m.port_data) or setattr(m, "watch", m.WATCH_ALL)
    )
    assert not machine.armed
    machine.run(start + 20)
    assert machine.armed and seen and set(seen) == {ORG}
    assert machine.watch == frozenset({ORG})
    machine[1] = 0x35
    assert ports == [0x35] and machine.watch is machine.WATCH_ALL
    machine.watch = []
    assert machine.watch == frozenset()


def test_decode_hook_and_stop():
    machine, start = _machine("  NOP\n  NOP\n  NOP\nl: JMP l\n")
    pcs = []

    def hook(m):
        pcs.append(m.mpu.pc)
        m.stop = len(pcs) == 3

    machine.set_decode_hook(hook)
    machine.run(start + 1000)
    assert pcs == [ORG, ORG + 1, ORG + 2] and machine.stop
    assert machine.t < start + 20
    machine.set_decode_hook(None)
    machine.run(start + 100)
    assert machine.t == start + 100 and len(pcs) == 3


def test_a_failing_callback_ends_the_run_and_raises():
    class Bad:
        def read(self, _cycle, _register):
            raise KeyError("read")

        def write(self, _cycle, _register, _value):
            raise OSError("write")

    machine, start = _machine("  STA $D400\nl: JMP l\n", sids={mm.SID_BASE: Bad()})
    with pytest.raises(OSError):
        machine.run(start + 1000)
    assert machine.t < start + 10
    with pytest.raises(KeyError):
        _ = machine[0xD400]

    machine, start = _machine("l: JMP l\n")
    machine.set_trap(lambda m: 1 / 0, machine.WATCH_ALL)
    with pytest.raises(ZeroDivisionError):
        machine.run(start + 100)


def test_run_is_not_reentrant_and_hooks_reach_the_machine_only_while_paused():
    machine, start = _machine("l: JMP l\n")
    errors = []

    def hook(m):
        for call in (lambda: m.run(m.t + 1), lambda: m.reset(ORG)):
            try:
                call()
            except RuntimeError as exc:
                errors.append(exc)
        m.stop = True

    machine.set_decode_hook(hook)
    machine.run(start + 10)
    assert len(errors) >= 1

    class Peeker:
        def read(self, _cycle, _register):
            return machine.mpu.pc

        def write(self, _cycle, _register, _value):
            pass

    machine, start = _machine("  LDA $D400\nl: JMP l\n", sids={mm.SID_BASE: Peeker()})
    with pytest.raises(RuntimeError):
        machine.run(start + 100)


def test_events_property_and_custom_sink():
    machine, start = _machine("  BRK\n  NOP\nl: JMP l\n")
    machine.ram[0x0314:0x0316] = (ORG + 2).to_bytes(2, "little")
    machine.run(start + 100)
    assert machine.events[0][1] == "brk_entry"
    machine.clear_events()
    assert machine.events == []


def test_reset_defaults_to_the_power_on_cycle():
    machine = Machine(backend="native")
    assert machine.reset(ORG) == vm.boot_cycle(False) == machine.t
    assert machine.reset(ORG, 5000) == 5000


@pytest.mark.parametrize("seed", range(6))
def test_random_programs_agree_with_the_python_machine(seed):
    assert machine_diff.compare(seed, 300_000) is None


def test_the_gil_is_released_and_machines_run_in_parallel():
    code, _ = assemble("l: INC $C000\n  JMP l\n", ORG)

    def spin():
        machine = Machine(backend="native")
        machine.ram[ORG : ORG + len(code)] = code
        machine.run(machine.reset(ORG) + 3_000_000)

    def wall(n):
        threads = [threading.Thread(target=spin) for _ in range(n)]
        began = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return time.perf_counter() - began

    single = min(wall(1) for _ in range(3))
    assert wall(4) < 2.5 * single + 0.2


def test_another_thread_is_refused_while_the_machine_runs():
    machine, start = _machine("l: JMP l\n")
    started, refused = threading.Event(), []

    def hook(m):
        started.set()
        time.sleep(0.05)
        m.stop = True

    def poke():
        started.wait()
        try:
            _ = machine[0]
        except RuntimeError as exc:
            refused.append(exc)

    machine.set_decode_hook(hook)
    thread = threading.Thread(target=poke)
    thread.start()
    machine.run(start + 100)
    thread.join()
    assert len(refused) == 1


def test_pages_follow_the_banking_and_ram_is_one_object():
    native, python = Machine(backend="native"), Machine(backend="python")
    assert native.ram is native.ram and native.colour is native.colour
    for port in (0x37, 0x35, 0x34, 0x33, 0x36, 0x31):
        native[1], python[1] = port, port
        assert native.bits == python.bits
        for page, ref in zip(native.pages, python.pages):
            assert (page is None) == (ref is None)
            if page is not None:
                span = slice(0, 0x10000)
                assert bytes(page[span]) == bytes(ref[span])
        assert (
            (native.pages[0xC] is native.ram)
            and native.pages[0xD] is not native.ram
            or (port & 7 in (4, 0, 1))
        )


def test_io_access_bypasses_the_pending_queue_and_banking():
    sid = machine_diff.Sid()
    native = Machine(sids={mm.SID_BASE: sid}, backend="native")
    python = Machine(sids={mm.SID_BASE: machine_diff.Sid()}, backend="python")
    for machine in (native, python):
        machine[1] = 0x34  # I/O banked out: io_* still reach the chips
        machine.io_write(0xD400 + 4, 0x21)
        machine.io_write(0xD800, 0x1F)
    assert sid.writes == [(native.t, 4, 0x21)] and native.colour[0] == 0x0F
    for addr in (0xD012, 0xD41B, 0xD800, 0xDC04, 0xDE00):
        assert native.io_read(addr) == python.io_read(addr)
