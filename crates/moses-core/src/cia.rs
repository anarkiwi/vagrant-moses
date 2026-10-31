//! The 6526/8521 CIA timers and interrupt logic as libsidplayfp models them.
//!
//! A port of `timer.cpp`, `interrupt.cpp` and `mos652x.cpp` (VICE's `ciatimer`
//! state machine). TOD and the serial port are not modelled.

use crate::lines::{Lines, CIA};
use crate::sched::{
    Sched, CIA_EVENTS, C_B_TICK, C_CLEAR_IRQ, C_INTERRUPT, C_SET_IRQ, C_SKIP, C_TIMER,
    C_UPDATE_IDR, PHI1, PHI2,
};

const START: u32 = 0x01;
const STEP: u32 = 0x04;
const CR_ONESHOT: u32 = 0x08;
const CR_FLOAD: u32 = 0x10;
const PHI2IN: u32 = 0x20;
const CR_MASK: u32 = START | CR_ONESHOT | CR_FLOAD | PHI2IN;
const COUNT2: u32 = 0x100;
const COUNT3: u32 = 0x200;
const ONESHOT0: u32 = CR_ONESHOT << 8;
const ONESHOT: u32 = CR_ONESHOT << 16;
const LOAD1: u32 = CR_FLOAD << 8;
const LOAD: u32 = CR_FLOAD << 16;
const OUT: u32 = 0x8000_0000;
const UNWANTED: u32 = OUT | CR_FLOAD | LOAD1 | LOAD;
const STEADY: u32 = START | PHI2IN | COUNT2 | COUNT3;
const UNDERFLOW_A: u8 = 0x01;
const UNDERFLOW_B: u8 = 0x02;
const REQUEST: u8 = 0x80;
const PRA: usize = 0;
const PRB: usize = 1;
const DDRA: usize = 2;
const DDRB: usize = 3;
const ICR: usize = 13;
const CRA: usize = 14;
const CRB: usize = 15;

/// One CIA timer: the cycle-exact count/load/one-shot pipeline.
#[derive(Clone)]
pub struct Timer {
    pub timer: u16,
    pub latch: u16,
    pub state: u32,
    last_control: u8,
    pause: i64,
    pb_toggle: bool,
    event: u8,
    skip: u8,
}

impl Timer {
    fn new(event: u8, skip: u8) -> Self {
        Timer {
            timer: 0xFFFF,
            latch: 0xFFFF,
            state: 0,
            last_control: 0,
            pause: 0,
            pb_toggle: false,
            event,
            skip,
        }
    }

    fn reset(&mut self, s: &mut Sched) {
        s.cancel(self.event);
        self.timer = 0xFFFF;
        self.latch = 0xFFFF;
        self.pb_toggle = false;
        self.state = 0;
        self.last_control = 0;
        self.pause = 0;
        s.schedule(self.event, 1, PHI1);
    }

    fn set_control(&mut self, cr: u8) {
        self.state = (self.state & !CR_MASK) | ((cr as u32 & CR_MASK) ^ PHI2IN);
        self.last_control = cr;
    }

    /// One PHI1 tick of the pipeline; true if it underflowed.
    fn clock(&mut self) -> bool {
        let state = self.state;
        if state & COUNT3 != 0 {
            self.timer = self.timer.wrapping_sub(1);
        }
        let mut adj = state & (START | CR_ONESHOT | PHI2IN);
        if state & (START | PHI2IN) == START | PHI2IN {
            adj |= COUNT2;
        }
        if state & COUNT2 != 0 || state & (STEP | START) == STEP | START {
            adj |= COUNT3;
        }
        adj |= (state & (CR_FLOAD | CR_ONESHOT | LOAD1 | ONESHOT0)) << 8;
        self.state = adj;
        let mut underflow = false;
        if self.timer == 0 && adj & COUNT3 != 0 {
            self.state |= LOAD | OUT;
            if self.state & (ONESHOT | ONESHOT0) != 0 {
                self.state &= !(START | COUNT2);
            }
            let toggle = self.last_control & 0x06 == 6;
            self.pb_toggle = toggle && !self.pb_toggle;
            underflow = true;
        }
        if self.state & LOAD != 0 {
            self.timer = self.latch;
            self.state &= !COUNT3;
        }
        underflow
    }

    fn reschedule(&mut self, s: &mut Sched) {
        let state = self.state;
        if state & UNWANTED != 0 {
            s.schedule_same_phase(self.event, 1);
        } else if state & COUNT3 != 0 {
            if self.timer > 2 && state & STEADY == STEADY {
                self.pause = s.time(PHI1) + 1;
                s.schedule_same_phase(self.skip, self.timer as i64 - 1);
            } else {
                s.schedule_same_phase(self.event, 1);
            }
        } else if state & (START | PHI2IN) == START | PHI2IN
            || state & (START | STEP) == START | STEP
        {
            s.schedule_same_phase(self.event, 1);
        } else {
            self.pause = -1;
        }
    }

    fn latch_lo(&mut self, data: u8) {
        self.latch = (self.latch & 0xFF00) | data as u16;
        if self.state & LOAD != 0 {
            self.timer = self.latch;
        }
    }

    fn latch_hi(&mut self, data: u8) {
        self.latch = (self.latch & 0xFF) | (data as u16) << 8;
        if self.state & LOAD != 0 {
            self.timer = self.latch;
        } else if self.state & START == 0 {
            self.state |= LOAD1;
        }
    }

    fn pb(&self, cr: u8) -> bool {
        if cr & 0x04 != 0 {
            self.pb_toggle
        } else {
            self.state & OUT != 0
        }
    }
}

/// The ICR/IDR pair, raising its interrupt 1 cycle late on a 6526.
#[derive(Clone)]
struct Source {
    new: bool,
    last_clear: i64,
    last_set: i64,
    icr: u8,
    idr: u8,
    idr_temp: u8,
    scheduled: bool,
    asserted: bool,
}

/// One CIA: two timers, Timer B cascading on A, and the interrupt source.
#[derive(Clone)]
pub struct Cia {
    pub regs: [u8; 16],
    pub t: [Timer; 2],
    src: Source,
    base: u8,
    nmi: bool,
}

impl Cia {
    /// `nmi` wires the interrupt to the NMI line (CIA 2), else to IRQ.
    pub fn new(index: u8, nmi: bool, new: bool) -> Self {
        let base = crate::sched::E_CIA + index * CIA_EVENTS;
        Cia {
            regs: [0; 16],
            t: [
                Timer::new(base + C_TIMER, base + C_SKIP),
                Timer::new(base + C_TIMER + 2, base + C_SKIP + 2),
            ],
            src: Source {
                new,
                last_clear: 0,
                last_set: 0,
                icr: 0,
                idr: 0,
                idr_temp: 0,
                scheduled: false,
                asserted: false,
            },
            base,
            nmi,
        }
    }

    /// Whether `ev` belongs to this CIA.
    pub fn owns(&self, ev: u8) -> bool {
        ev >= self.base && ev < self.base + CIA_EVENTS
    }

    pub fn reset(&mut self, s: &mut Sched) {
        self.regs = [0; 16];
        self.t[0].reset(s);
        self.t[1].reset(s);
        let src = &mut self.src;
        src.last_clear = 0;
        src.last_set = 0;
        src.icr = 0;
        src.idr = 0;
        src.idr_temp = 0;
        for off in [C_UPDATE_IDR, C_SET_IRQ, C_CLEAR_IRQ, C_INTERRUPT] {
            s.cancel(self.base + off);
        }
        src.scheduled = false;
        src.asserted = false;
        s.cancel(self.base + C_B_TICK);
    }

    pub fn fire(&mut self, ev: u8, s: &mut Sched, ln: &mut Lines) {
        match ev - self.base {
            C_TIMER => self.timer_event(0, s, ln),
            2 => self.timer_event(1, s, ln),
            C_SKIP => self.skip(0, s, ln),
            3 => self.skip(1, s, ln),
            C_INTERRUPT => self.interrupt_fire(s, ln),
            C_UPDATE_IDR => self.update_idr(s),
            C_SET_IRQ => self.set_irq(s, ln),
            C_CLEAR_IRQ => self.clear_irq(s, ln),
            _ => self.cascade(s, ln),
        }
    }

    fn timer_event(&mut self, i: usize, s: &mut Sched, ln: &mut Lines) {
        self.clock(i, s, ln);
        self.t[i].reschedule(s);
    }

    fn skip(&mut self, i: usize, s: &mut Sched, ln: &mut Lines) {
        let t = &mut self.t[i];
        let elapsed = s.time(PHI1) - t.pause;
        t.pause = 0;
        t.timer = t.timer.wrapping_sub(elapsed as u16);
        self.timer_event(i, s, ln);
    }

    fn clock(&mut self, i: usize, s: &mut Sched, ln: &mut Lines) {
        if self.t[i].clock() {
            if i == 0 {
                self.trigger(UNDERFLOW_A, s, ln);
                if self.regs[CRB] & 0x41 == 0x41 && self.t[1].state & START != 0 {
                    s.schedule(self.base + C_B_TICK, 0, PHI2);
                }
            } else {
                self.trigger(UNDERFLOW_B, s, ln);
            }
        }
    }

    /// Bring a skipping or sleeping timer up to the CPU's cycle.
    fn sync(&mut self, i: usize, s: &mut Sched, ln: &mut Lines) {
        if self.t[i].pause > 0 {
            s.cancel(self.t[i].skip);
            let elapsed = s.time(PHI2) - self.t[i].pause;
            if elapsed >= 0 {
                let t = &mut self.t[i];
                t.timer = t.timer.wrapping_sub(elapsed as u16);
                self.clock(i, s, ln);
            }
        }
        if self.t[i].pause == 0 {
            s.cancel(self.t[i].event);
        }
        self.t[i].pause = -1;
    }

    fn wake(&mut self, i: usize, s: &mut Sched) {
        self.t[i].pause = 0;
        s.schedule(self.t[i].event, 0, PHI1);
    }

    fn cascade(&mut self, s: &mut Sched, ln: &mut Lines) {
        self.sync(1, s, ln);
        self.t[1].state |= STEP;
        self.wake(1, s);
    }

    // --- interrupt source --------------------------------------------------

    fn sink(&self, s: &mut Sched, ln: &mut Lines, state: bool) {
        if self.nmi {
            ln.cia_nmi(s, state);
        } else {
            ln.irq(s, state, CIA);
        }
    }

    fn ack0(&self, s: &Sched) -> bool {
        s.time(PHI2) == self.src.last_clear + 1
    }

    fn write0(&self, s: &Sched) -> bool {
        s.time(PHI2) == self.src.last_set + 1
    }

    fn interrupt_fire(&mut self, s: &mut Sched, ln: &mut Lines) {
        if self.src.idr & REQUEST == 0 {
            self.src.idr |= REQUEST;
            if self.src.new {
                self.src.idr_temp |= REQUEST;
                if self.ack0(s) {
                    s.schedule(self.base + C_SET_IRQ, 1, PHI1);
                }
            }
            self.set_irq(s, ln);
        }
        self.src.scheduled = false;
    }

    fn update_idr(&mut self, s: &mut Sched) {
        self.src.idr = self.src.idr_temp;
        if self.ack0(s) {
            s.schedule(self.base + C_UPDATE_IDR, 1, PHI1);
            self.src.idr_temp = 0;
        }
    }

    fn set_irq(&mut self, s: &mut Sched, ln: &mut Lines) {
        if !self.ack0(s) && !self.src.asserted {
            self.sink(s, ln, true);
            self.src.asserted = true;
        }
    }

    fn clear_irq(&mut self, s: &mut Sched, ln: &mut Lines) {
        if self.src.asserted {
            self.sink(s, ln, false);
            self.src.asserted = false;
        }
    }

    /// Flag source `mask` (0 re-evaluates after an ICR write).
    fn trigger(&mut self, mask: u8, s: &mut Sched, _ln: &mut Lines) {
        let src = &mut self.src;
        src.idr |= mask;
        src.idr_temp |= mask;
        let flagged = if mask != 0 { mask } else { src.idr };
        if flagged & src.icr != 0 {
            if !src.scheduled {
                s.schedule(self.base + C_INTERRUPT, if src.new { 0 } else { 1 }, PHI1);
                src.scheduled = true;
            }
        } else if mask == 0 && self.write0(s) && self.src.scheduled {
            s.cancel(self.base + C_INTERRUPT);
            self.src.scheduled = false;
        }
        let src = &mut self.src;
        if !src.new && mask == UNDERFLOW_B && s.time(PHI2) == src.last_clear + 1 {
            src.idr &= !UNDERFLOW_B;
            src.idr_temp &= !UNDERFLOW_B;
        }
    }

    fn icr_set(&mut self, data: u8, s: &mut Sched, ln: &mut Lines) {
        if data & REQUEST != 0 {
            self.src.icr |= data & !REQUEST;
        } else {
            self.src.icr &= !data;
        }
        if !self.ack0(s) {
            self.trigger(0, s, ln);
        }
        self.src.last_set = s.time(PHI2);
    }

    fn icr_clear(&mut self, s: &mut Sched) -> u8 {
        self.src.last_clear = s.time(PHI2);
        s.schedule(self.base + C_CLEAR_IRQ, 0, PHI1);
        if !s.is_pending(self.base + C_UPDATE_IDR) {
            s.schedule(self.base + C_UPDATE_IDR, 1, PHI1);
            self.src.idr_temp = 0;
        }
        let idr = self.src.idr;
        if !self.src.new {
            self.src.idr &= REQUEST;
        }
        idr
    }

    // --- registers ---------------------------------------------------------

    fn port_b(&self) -> u8 {
        let mut data = self.regs[PRB] | !self.regs[DDRB];
        if self.regs[CRA] & 0x02 != 0 {
            data = data & 0xBF
                | if self.t[0].pb(self.regs[CRA]) {
                    0x40
                } else {
                    0
                };
        }
        if self.regs[CRB] & 0x02 != 0 {
            data = data & 0x7F
                | if self.t[1].pb(self.regs[CRB]) {
                    0x80
                } else {
                    0
                };
        }
        data
    }

    /// A CPU read of register `addr` (mirrored every 16 bytes).
    pub fn read(&mut self, addr: u16, s: &mut Sched, ln: &mut Lines) -> u8 {
        let addr = (addr & 0x0F) as usize;
        for i in 0..2 {
            self.sync(i, s, ln);
            self.wake(i, s);
        }
        match addr {
            PRA => self.regs[PRA] | !self.regs[DDRA],
            PRB => self.port_b(),
            4..=7 => {
                let t = &self.t[(addr >= 6) as usize];
                (t.timer >> (8 * (addr & 1))) as u8
            }
            ICR => self.icr_clear(s),
            CRA | CRB => {
                let t = &self.t[(addr == CRB) as usize];
                (self.regs[addr] & 0xEE) | (t.state & 1) as u8
            }
            _ => self.regs[addr],
        }
    }

    /// A CPU write of `data` to register `addr`.
    pub fn write(&mut self, addr: u16, data: u8, s: &mut Sched, ln: &mut Lines) {
        let addr = (addr & 0x0F) as usize;
        self.sync(0, s, ln);
        self.sync(1, s, ln);
        let old = std::mem::replace(&mut self.regs[addr], data);
        match addr {
            4..=7 => {
                let t = &mut self.t[(addr >= 6) as usize];
                if addr & 1 != 0 {
                    t.latch_hi(data)
                } else {
                    t.latch_lo(data)
                }
            }
            ICR => self.icr_set(data, s, ln),
            CRA => {
                if data & 1 != 0 && old & 1 == 0 {
                    self.t[0].pb_toggle = true;
                }
                self.t[0].set_control(data);
            }
            CRB => {
                if data & 1 != 0 && old & 1 == 0 {
                    self.t[1].pb_toggle = true;
                }
                self.t[1].set_control(data | (data & 0x40) >> 1);
            }
            _ => {}
        }
        self.wake(0, s);
        self.wake(1, s);
    }
}
