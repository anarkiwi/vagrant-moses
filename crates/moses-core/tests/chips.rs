//! The keyboard matrix and chip-state seeding on bare chips over a scheduler.

use moses_core::cia::Cia;
use moses_core::lines::{Lines, CIA, VIDEO};
use moses_core::sched::{Sched, E_CLEAR_INT, PHI2};
use moses_core::vic::Vic;

/// Fire every chip event due up to cycle `cycle`'s PHI1, then stand at its PHI2.
fn to(cycle: i64, s: &mut Sched, ln: &mut Lines, cia: &mut Cia, vic: &mut Vic) {
    while let Some(ev) = s.pop_due(cycle << 1) {
        if cia.owns(ev) {
            cia.fire(ev, s, ln);
        } else if ev != E_CLEAR_INT {
            vic.fire(ev, s, ln);
        }
    }
    s.now = cycle << 1 | PHI2;
}

fn chips() -> (Sched, Lines, Cia, Vic) {
    let (mut s, ln) = (Sched::new(), Lines::new());
    let (mut cia, mut vic) = (Cia::new(0, false, false), Vic::new(false, 312, 63));
    cia.reset(&mut s);
    vic.reset(&mut s);
    (s, ln, cia, vic)
}

#[test]
fn keys_pull_the_other_port_low() {
    let (mut s, mut ln, mut cia, _) = chips();
    cia.keys[1] = 0x24;
    cia.keys[7] = 0x10;
    for (addr, value) in [(2, 0xFF), (3, 0x00), (0, 0xFD)] {
        cia.write(addr, value, &mut s, &mut ln);
    }
    assert_eq!(cia.read(1, &mut s, &mut ln), 0xDB);
    cia.write(0, 0x00, &mut s, &mut ln);
    assert_eq!(cia.read(1, &mut s, &mut ln), 0xCB);
    for (addr, value) in [(2, 0x00), (3, 0xFF), (1, 0xEF)] {
        cia.write(addr, value, &mut s, &mut ln);
    }
    assert_eq!(cia.read(0, &mut s, &mut ln), 0x7F);
    cia.write(1, 0xDB, &mut s, &mut ln);
    assert_eq!(cia.read(0, &mut s, &mut ln), 0xFD);
}

#[test]
fn a_seeded_timer_underflows_after_its_counter_then_every_latch_plus_one() {
    let (mut s, mut ln, mut cia, mut vic) = chips();
    to(10, &mut s, &mut ln, &mut cia, &mut vic);
    cia.seed_icr(0x01);
    cia.seed_timer(0, 20, 99, 0x01, &mut s, &mut ln);
    assert_eq!(
        (cia.t[0].timer, cia.t[0].latch, cia.icr_mask()),
        (20, 99, 1)
    );
    let mut raises = vec![];
    for cycle in 11..400 {
        to(cycle, &mut s, &mut ln, &mut cia, &mut vic);
        if ln.asserted[CIA] {
            raises.push(cycle);
            cia.read(13, &mut s, &mut ln);
        }
    }
    assert_eq!(raises, [31, 131, 231, 331]);
}

#[test]
fn a_seeded_beam_steals_the_badline_and_raises_the_compare_line() {
    let (mut s, mut ln, mut cia, mut vic) = chips();
    to(5, &mut s, &mut ln, &mut cia, &mut vic);
    vic.write(0x11, 0x1B, &mut s, &mut ln);
    vic.write(0x1A, 0x01, &mut s, &mut ln);
    vic.seed_raster_compare(0x34, &mut s, &mut ln);
    assert!(vic.seed_beam(0x33, 30, &mut s, &mut ln));
    assert!(!ln.rdy);
    assert!(!vic.seed_beam(312, 0, &mut s, &mut ln));
    assert!(vic.seed_beam(0x33, 10, &mut s, &mut ln));
    assert!(ln.rdy);
    let (mut low, mut raised) = (vec![], None);
    for cycle in 6..200 {
        to(cycle, &mut s, &mut ln, &mut cia, &mut vic);
        if !ln.rdy {
            low.push(cycle);
        }
        if ln.asserted[VIDEO] && raised.is_none() {
            raised = Some(cycle);
        }
    }
    assert_eq!(low, (7..50).collect::<Vec<_>>());
    assert_eq!(raised, Some(6 + 63 - 10));
}
