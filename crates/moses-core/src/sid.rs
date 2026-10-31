//! The host-side face of a SID: the write log and the data-bus value a read of an
//! unreadable register returns. Nothing of the chip itself is modelled here.
//!
//! Bus value rule and constants: libresidfp `SID.cpp` (`BUS_TTL_6581`, `BUS_TTL_8580`,
//! `SID::read`, `SID::write`) and `SID.h` (`ageBusValue`). A write, and a read of a readable
//! register (`$19-$1C`), set the value and its time to live; reading any other register
//! halves the time left; the value becomes 0 once the time has run out.

use std::cell::UnsafeCell;

pub const REGS: usize = 32;
const BUS_TTL_6581: i64 = 0x01d00;
const BUS_TTL_8580: i64 = 0xa2000;

/// A write as logged: `(cycle, register, value)`.
pub type Write = (i64, u8, u8);

pub fn readable(reg: u8) -> bool {
    (0x19..=0x1C).contains(&reg)
}

pub struct SidState {
    ttl_model: i64,
    pub writes: Vec<Write>,
    pub rows: Vec<Write>,
    pub last: [u8; REGS],
    bus: u8,
    ttl: i64,
    clock: i64,
}

impl SidState {
    pub fn new(is_8580: bool) -> Self {
        SidState {
            ttl_model: if is_8580 { BUS_TTL_8580 } else { BUS_TTL_6581 },
            writes: Vec::new(),
            rows: Vec::new(),
            last: [0; REGS],
            bus: 0,
            ttl: 0,
            clock: 0,
        }
    }

    fn age(&mut self, cycle: i64) {
        let elapsed = cycle - std::mem::replace(&mut self.clock, cycle);
        if self.ttl != 0 {
            self.ttl -= elapsed;
            if self.ttl <= 0 {
                self.bus = 0;
                self.ttl = 0;
            }
        }
    }

    fn drive(&mut self, value: u8) {
        self.bus = value;
        self.ttl = self.ttl_model;
    }

    /// Log a write; `rows` keeps those that change the register.
    pub fn write(&mut self, cycle: i64, reg: u8, value: u8) {
        self.age(cycle);
        self.drive(value);
        self.writes.push((cycle, reg, value));
        let last = &mut self.last[reg as usize % REGS];
        if std::mem::replace(last, value) != value {
            self.rows.push((cycle, reg, value));
        }
    }

    /// The bus value for a register the chip does not drive; `None` for a readable one.
    pub fn peek(&mut self, cycle: i64, reg: u8) -> Option<u8> {
        if readable(reg) {
            return None;
        }
        self.age(cycle);
        self.ttl /= 2;
        Some(self.bus)
    }

    /// The byte the chip drove for a readable register.
    pub fn settle(&mut self, cycle: i64, value: u8) {
        self.age(cycle);
        self.drive(value);
    }
}

/// A [`SidState`] shared between the machine and its Python owner.
pub struct SidShared(UnsafeCell<SidState>);

// SAFETY: callers of `state` serialise all access (the GIL, and the run claim of the
// machine that owns the device).
unsafe impl Sync for SidShared {}
unsafe impl Send for SidShared {}

impl SidShared {
    pub fn new(is_8580: bool) -> Self {
        SidShared(UnsafeCell::new(SidState::new(is_8580)))
    }

    /// # Safety
    /// No other reference to the state may be live or used while the result is.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn state(&self) -> &mut SidState {
        &mut *self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_value_decays_and_halves() {
        let mut s = SidState::new(false);
        s.write(10, 3, 0xAB);
        assert_eq!(s.peek(11, 0), Some(0xAB));
        assert_eq!(s.ttl, BUS_TTL_6581 / 2 - 1);
        assert_eq!(s.peek(10 + BUS_TTL_6581 / 2, 0x1D), Some(0));
        assert_eq!(s.peek(20_000, 4), Some(0));
    }

    #[test]
    fn readable_registers_are_left_to_the_chip() {
        let mut s = SidState::new(true);
        assert_eq!(s.peek(5, 0x1B), None);
        s.settle(5, 0x55);
        assert_eq!(s.peek(6, 0x1F), Some(0x55));
        assert_eq!(s.peek(5 + BUS_TTL_8580, 0), Some(0));
    }

    #[test]
    fn rows_are_the_changed_writes() {
        let mut s = SidState::new(false);
        for (c, v) in [(1, 0), (2, 5), (3, 5), (4, 0)] {
            s.write(c, 7, v);
        }
        assert_eq!(s.writes.len(), 4);
        assert_eq!(s.rows, vec![(2, 7, 5), (4, 7, 0)]);
    }
}
