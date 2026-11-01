//! The machine's interrupt lines and the CPU's view of them (`ic`, IRQ pin, NMI, RDY).

use crate::sched::{Sched, E_CLEAR_INT, PHI1};

/// Interrupt-clock sentinel: no interrupt is being delayed.
pub const MAX: i32 = 65536;
pub const NMI: usize = 0;
pub const VIDEO: usize = 1;
pub const CIA: usize = 2;
pub const REU: usize = 3;
const IRQS: [usize; 3] = [VIDEO, CIA, REU];

/// Event names: the sources by index, `rti`, `brk_entry`, then each source's entry.
pub const NAMES: [&str; 10] = [
    "nmi",
    "video_irq",
    "cia_irq",
    "reu_irq",
    "rti",
    "brk_entry",
    "nmi_entry",
    "video_irq_entry",
    "cia_irq_entry",
    "reu_irq_entry",
];
pub const EV_RTI: u8 = 4;
pub const EV_BRK_ENTRY: u8 = 5;
/// The entry event of source `k` is `EV_ENTRY + k`.
pub const EV_ENTRY: u8 = 6;

#[derive(Clone)]
pub struct Lines {
    pub ic: i32,
    pub pin: bool,
    pub nmi: bool,
    pub iflag: bool,
    pub rdy: bool,
    pub cc: i32,
    pub asserted: [bool; 4],
    pub irq_cause: usize,
    irq_count: i32,
    pub buf: Vec<(i64, u8)>,
}

impl Lines {
    pub fn new() -> Self {
        Lines {
            ic: MAX,
            pin: false,
            nmi: false,
            iflag: false,
            rdy: true,
            cc: 6,
            asserted: [false; 4],
            irq_cause: CIA,
            irq_count: 0,
            buf: Vec::new(),
        }
    }

    pub fn emit(&mut self, cycle: i64, name: u8) {
        self.buf.push((cycle, name));
    }

    fn mark(&mut self, sched: &Sched, kind: usize) {
        if kind != NMI && !IRQS.iter().any(|&k| self.asserted[k]) {
            self.irq_cause = kind;
        }
        self.asserted[kind] = true;
        self.emit(sched.time(PHI1), kind as u8);
    }

    /// An IRQ source (`VIDEO`, `CIA` or `REU`) changing state.
    pub fn irq(&mut self, sched: &mut Sched, state: bool, kind: usize) {
        if state {
            self.mark(sched, kind);
            if self.irq_count == 0 {
                self.pin = true;
                self.calc();
            }
            self.irq_count += 1;
        } else {
            self.asserted[kind] = false;
            if let Some(&other) = IRQS.iter().find(|&&k| k != kind && self.asserted[k]) {
                self.irq_cause = other;
            }
            self.irq_count -= 1;
            if self.irq_count == 0 {
                self.pin = false;
                sched.schedule(E_CLEAR_INT, 2, PHI1);
            }
        }
    }

    pub fn cia_nmi(&mut self, sched: &Sched, state: bool) {
        if state {
            self.mark(sched, NMI);
            self.nmi = true;
            self.calc();
        }
    }

    pub fn set_ba(&mut self, state: bool) {
        self.rdy = state;
    }

    #[inline(always)]
    pub fn check(&self) -> bool {
        self.nmi || (self.pin && !self.iflag)
    }

    pub fn calc(&mut self) {
        if self.ic == MAX && self.check() {
            self.ic = self.cc;
        }
    }

    pub fn remove_irq(&mut self) {
        if !self.nmi && self.ic != MAX {
            self.ic = MAX;
        }
    }
}

impl Default for Lines {
    fn default() -> Self {
        Self::new()
    }
}
