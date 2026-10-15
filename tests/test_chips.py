"""The VIC-II and CIA ports against documented 6526/6569 register behaviour."""

from vagrant_moses.cia import Cia
from vagrant_moses.eventsched import Event, Scheduler
from vagrant_moses.vicii import NTSC_GEOMETRY, PAL_GEOMETRY, Vic


def _cia():
    sched, lines = Scheduler(), []
    cia = Cia(sched, lambda state: lines.append((sched.time(0), state)), new=False)
    cia.reset()
    return sched, cia, lines


def _at(sched, cycle):
    """Run the chips to ``cycle`` and stand at its CPU (PHI2) slot."""
    sched.run(2 * cycle)
    sched.now = 2 * cycle + 1


def test_icr_read_reports_then_clears_the_flags():
    sched, cia, lines = _cia()
    _at(sched, 2)
    cia.write(0x04, 0x05)
    cia.write(0x05, 0x00)
    cia.write(0x0D, 0x81)
    cia.write(0x0E, 0x19)
    _at(sched, 40)
    assert lines == [(lines[0][0], True)]
    assert cia.read(0x0D) == 0x81
    _at(sched, 45)
    assert cia.read(0x0D) == 0x00
    assert lines[-1][1] is False


def test_timer_b_cascades_timer_a_underflows():
    sched, cia, _ = _cia()
    _at(sched, 2)
    for addr, value in ((4, 9), (5, 0), (6, 2), (7, 0), (0x0F, 0x51), (0x0E, 0x11)):
        cia.write(addr, value)
    seen = set()
    for cycle in range(3, 200):
        _at(sched, cycle)
        seen.add(cia.read(0x06))
    assert seen == {0, 1, 2}
    assert cia.read(0x0F) & 0x01 and cia.read(0x0E) & 0x01


def test_one_shot_stops_and_force_load_reloads():
    sched, cia, _ = _cia()
    _at(sched, 2)
    cia.write(0x04, 0x20)
    cia.write(0x05, 0x00)
    cia.write(0x0E, 0x19)
    _at(sched, 100)
    assert cia.read(0x0E) & 0x01 == 0 and cia.read(0x04) == 0x20
    cia.write(0x04, 0x10)
    cia.write(0x0E, 0x10)
    _at(sched, 104)
    assert cia.read(0x04) == 0x10


def test_timer_outputs_toggle_port_b():
    sched, cia, _ = _cia()
    _at(sched, 2)
    cia.write(0x04, 0x03)
    cia.write(0x05, 0x00)
    cia.write(0x0E, 0x17)
    levels = set()
    for cycle in range(3, 40):
        _at(sched, cycle)
        levels.add(cia.read(0x01) & 0x40)
    assert levels == {0x00, 0x40}
    cia.write(0x0F, 0x02)
    assert cia.read(0x01) & 0x80 == 0


def test_unmodelled_registers_read_back_writes():
    sched, cia, _ = _cia()
    _at(sched, 2)
    cia.write(0x08, 0x12)
    cia.write(0x02, 0xFF)
    cia.write(0x00, 0x0F)
    assert (cia.read(0x08), cia.read(0x00)) == (0x12, 0x0F)


def test_vic_register_reads():
    sched = Scheduler()
    vic = Vic(sched, False, lambda _s: None, lambda _s: None)
    _at(sched, 70)
    vic.write(0x1A, 0x01)
    vic.write(0x20, 0x0E)
    assert (vic.read(0x1A), vic.read(0x20), vic.read(0x30)) == (0xF1, 0xFE, 0xFF)
    assert (vic.read(0x12), vic.read(0x13), vic.read(0x15)) == (0, 0, 0)


def test_sprite_crunch_mixes_the_data_counters():
    sched = Scheduler()
    vic = Vic(sched, False, lambda _s: None, lambda _s: None)
    sprites = vic.sprites
    sprites.exp_flop, sprites.mc, sprites.mc_base = 0x00, [0x3F] * 8, [0x00] * 8
    vic.line_cycle = 14
    sprites.line_crunch(0x00, 14)
    assert sprites.mc[0] == 0x15 and sprites.exp_flop == 0xFF


def test_scheduler_orders_same_slot_events_by_scheduling():
    sched, fired = Scheduler(), []
    first, second = Event(lambda: fired.append(1)), Event(lambda: fired.append(2))
    sched.schedule(first, 1, 0)
    sched.schedule(second, 1, 0)
    sched.cancel(first)
    sched.schedule(first, 1, 0)
    sched.run(10)
    assert fired == [2, 1] and sched.time(1) == 1


PAL_FRAME = PAL_GEOMETRY[0] * PAL_GEOMETRY[1]
NTSC_FRAME = NTSC_GEOMETRY[0] * NTSC_GEOMETRY[1]


def test_vic_raster_irq_latches_until_cleared():
    sched, fired = Scheduler(), []
    vic = Vic(sched, False, fired.append, lambda _s: None)
    vic.write(0x1A, 0x01)  # enable the raster flag
    vic.write(0x12, 100)  # compare at line 100
    sched.run(2 * PAL_FRAME)
    sched.run(2 * 2 * PAL_FRAME)  # second frame: flag already set, no re-assert
    assert fired == [True] and vic.irq_flags & 0x01


def test_vic_clearing_the_raster_flag_reasserts_next_frame():
    sched, fired = Scheduler(), []
    vic = Vic(sched, False, fired.append, lambda _s: None)
    vic.write(0x1A, 0x01)
    vic.write(0x12, 100)
    sched.run(2 * PAL_FRAME)
    vic.write(0x19, 0x01)  # clear the raster flag -> the IRQ line falls
    assert fired == [True, False]
    sched.run(2 * 2 * PAL_FRAME)  # re-asserts on the next crossing
    assert fired == [True, False, True]


def test_vic_badline_steals_only_when_display_is_on():
    sched_on = Scheduler()
    ba_on = []
    vic_on = Vic(sched_on, False, lambda _s: None, ba_on.append)
    vic_on.write(0x11, 0x10)  # display on (DEN)
    sched_on.run(2 * PAL_FRAME)
    sched_off = Scheduler()
    ba_off = []
    _ = Vic(sched_off, False, lambda _s: None, ba_off.append)
    sched_off.run(2 * PAL_FRAME)
    assert (
        False in ba_on and True in ba_on
    )  # badlines pulled the bus low, then released
    assert all(v is True for v in ba_off)  # no steals with display off


def test_vic_sprite_dma_pulls_the_bus_low():
    sched = Scheduler()
    ba = []
    vic = Vic(sched, False, lambda _s: None, ba.append)
    vic.write(0x15, 0x01)  # enable sprite 0
    vic.write(0x01, 100)  # its Y coordinate
    sched.run(2 * PAL_FRAME)
    assert False in ba  # the sprite DMA fetch stole the bus


def test_vic_ntsc_beam_fires_the_raster_irq():
    sched, fired = Scheduler(), []
    vic = Vic(sched, True, fired.append, lambda _s: None)
    assert (vic.max_rasters, vic.cycles_per_line) == NTSC_GEOMETRY == (263, 65)
    vic.write(0x1A, 0x01)
    vic.write(0x12, 100)
    sched.run(2 * NTSC_FRAME)
    assert fired == [True]


def test_vic_read_fixed_and_mirror_values():
    vic = Vic(Scheduler(), False, lambda _s: None, lambda _s: None)
    assert vic.read(0x13) == 0 and vic.read(0x14) == 0  # light pen reads zero
    assert vic.read(0x19) & 0x70 == 0x70  # fixed status bits present
    assert 0 <= vic.read(0x12) <= PAL_GEOMETRY[0] - 1  # a valid PAL raster line
