//! Everything the 6510 sees besides itself: banked memory, the I/O page and the chips.

use std::collections::VecDeque;
use std::sync::Arc;

use jennings_core::Bus;

use crate::cia::Cia;
use crate::lines::{Lines, REU};
use crate::reu::{Port, Reu};
use crate::sched::{Sched, E_BADLINE, E_CLEAR_INT, E_EDGE, E_VIC, PHI1};
use crate::sid::SidShared;
use crate::smc::Recorder;
use crate::vic::Vic;

pub const IO: u8 = 255;
pub const RAM: usize = 0;
pub const BASIC: usize = 1;
pub const CHARGEN: usize = 2;
pub const KERNAL: usize = 3;
const SID_BASE: u16 = 0xD400;
const SID_LAST: u16 = 0xD7FF;
const LORAM: u8 = 1;
const HIRAM: u8 = 2;
const CHAREN: u8 = 4;

/// What answers at a SID base: a host device, or a logging device that asks the host
/// (with the host index) only for the registers a chip drives.
pub enum SidDev {
    Host(usize),
    Logged(Arc<SidShared>, usize),
}

/// The Python side of the machine: SID devices, the event sink and the hooks.
pub trait Host {
    fn sid_read(&mut self, dev: usize, cycle: i64, reg: u8) -> u8;
    fn sid_write(&mut self, dev: usize, cycle: i64, reg: u8, value: u8);
    fn event(&mut self, cycle: i64, name: u8);
    fn trap(&mut self, core: &mut crate::Core) -> bool;
    fn decode(&mut self, core: &mut crate::Core);
    fn port_write(&mut self, core: &mut crate::Core);
    /// A callback failed: the run ends at the next cycle.
    fn failed(&self) -> bool {
        false
    }
}

/// The 6510 port `$01` as read: output bits as written, inputs as the C64 pulls.
pub fn port_read(ddr: u8, data: u8) -> u8 {
    let pins = 0x3F & !ddr | data & ddr;
    let mut value = (data | !ddr) & (pins | 0x17);
    if ddr & 0x20 == 0 {
        value &= !0x20;
    }
    value & !(!ddr & 0xC0)
}

/// `(basic, kernal, chargen, io)` mapped for the port's effective low bits.
pub fn banking(bits: u8) -> (bool, bool, bool, bool) {
    let (loram, hiram, charen) = (bits & LORAM != 0, bits & HIRAM != 0, bits & CHAREN != 0);
    (
        loram && hiram,
        hiram,
        !charen && (loram || hiram),
        charen && (loram || hiram),
    )
}

pub struct Sys {
    pub sched: Sched,
    pub vic: Vic,
    pub cia: [Cia; 2],
    pub ln: Lines,
    pub img: [Box<[u8; 0x10000]>; 4],
    pub map: [u8; 16],
    pub colour: Box<[u8; 0x400]>,
    pub port_dir: u8,
    pub port_data: u8,
    pub bits: i32,
    pub io: bool,
    pub pending: VecDeque<(u16, u8)>,
    pub last_read: u8,
    pub seed: u32,
    /// SID device bases (as keyed by the caller) and their host index.
    pub sids: Vec<(u32, SidDev)>,
    pub events: Vec<(i64, u8)>,
    pub custom_sink: bool,
    pub port_dirty: bool,
    /// The REU at `$DF00`, out of its slot while it runs a DMA cycle.
    pub reu: Option<Box<Reu>>,
    /// A DMA is pending or running: the CPU halts at its next read cycle.
    pub dma: bool,
    /// The instruction being executed is a read-modify-write.
    pub rmw: bool,
    /// The SMC recorder, when armed.
    pub rec: Option<Box<Recorder>>,
}

impl Sys {
    pub fn remap(&mut self) {
        let bits = ((self.port_data | !self.port_dir) & 7) as i32;
        if bits == self.bits {
            return;
        }
        self.bits = bits;
        let (basic, kernal, chars, io) = banking(bits as u8);
        self.io = io;
        let pick = |on: bool, rom: usize| if on { rom as u8 } else { RAM as u8 };
        self.map[0xA] = pick(basic, BASIC);
        self.map[0xB] = self.map[0xA];
        self.map[0xD] = if io { IO } else { pick(chars, CHARGEN) };
        self.map[0xE] = pick(kernal, KERNAL);
        self.map[0xF] = self.map[0xE];
    }

    /// Hand the events the chips buffered to the sink, in order.
    pub fn flush<H: Host>(&mut self, h: &mut H) {
        if self.ln.buf.is_empty() {
            return;
        }
        if self.custom_sink {
            for (cycle, name) in std::mem::take(&mut self.ln.buf) {
                h.event(cycle, name);
            }
        } else {
            self.events.append(&mut self.ln.buf);
        }
    }

    /// Fire every event due at or before half-cycle `limit`, in order.
    pub fn run_sched(&mut self, limit: i64) {
        while let Some(ev) = self.sched.pop_due(limit) {
            match ev {
                E_VIC | E_BADLINE | E_EDGE => self.vic.fire(ev, &mut self.sched, &mut self.ln),
                E_CLEAR_INT => self.ln.remove_irq(),
                _ => {
                    let cia = if self.cia[0].owns(ev) { 0 } else { 1 };
                    self.cia[cia].fire(ev, &mut self.sched, &mut self.ln);
                }
            }
        }
    }

    /// The recorder's trace, which answers the SID window ahead of the devices.
    fn tracer(&mut self) -> Option<&mut crate::trace::Tracer> {
        self.rec.as_deref_mut()?.tracer.as_deref_mut()
    }

    fn sid(sids: &[(u32, SidDev)], addr: u16) -> Option<&SidDev> {
        let find = |base: u32| sids.iter().find(|s| s.0 == base).map(|s| &s.1);
        find((addr & 0xFFE0) as u32).or_else(|| {
            if (SID_BASE..=SID_LAST).contains(&addr) {
                find(SID_BASE as u32)
            } else {
                None
            }
        })
    }

    /// Pass the REU's IRQ line on to the interrupt lines when it changed.
    pub fn reu_line(&mut self) {
        if let Some(r) = self.reu.as_mut() {
            if r.line != r.signalled {
                r.signalled = r.line;
                self.ln.irq(&mut self.sched, r.line, REU);
            }
        }
    }

    fn reu_read(&mut self, addr: u16) -> Option<u8> {
        if addr >> 8 != 0xDF {
            return None;
        }
        let v = self.reu.as_mut()?.read(addr);
        self.reu_line();
        v
    }

    /// One DMA cycle of the REU with BA `ba`; true when the CPU runs this cycle.
    pub fn reu_step<H: Host>(&mut self, h: &mut H, ba: bool) -> bool {
        let Some(mut reu) = self.reu.take() else {
            return true;
        };
        let go = reu.step(ba, &mut BusRef { s: self, h });
        self.reu = Some(reu);
        self.reu_line();
        go
    }

    pub fn io_read<H: Host>(&mut self, h: &mut H, addr: u16) -> u8 {
        let page = addr >> 8;
        let value = if let Some(v) = self.reu_read(addr) {
            v
        } else if page < 0xD4 {
            self.vic.read(addr, &mut self.sched, &mut self.ln)
        } else if !(0xD8..0xDE).contains(&page) {
            let (cycle, reg) = (self.sched.time(PHI1), (addr & 0x1F) as u8);
            let traced = self.tracer().and_then(|t| t.sid_read(addr));
            match (traced, Self::sid(&self.sids, addr)) {
                (Some(v), _) => v,
                (None, None) => {
                    self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    self.seed as u8
                }
                (None, Some(&SidDev::Host(dev))) => {
                    self.flush(h);
                    h.sid_read(dev, cycle, reg)
                }
                (None, Some(SidDev::Logged(state, dev))) => {
                    // SAFETY: serialised by the machine's run claim; no reference outlives the call.
                    let peeked = unsafe { state.state() }.peek(cycle, reg);
                    match peeked {
                        Some(v) => v,
                        None => {
                            let (state, dev) = (Arc::clone(state), *dev);
                            self.flush(h);
                            let v = h.sid_read(dev, cycle, reg);
                            // SAFETY: as above; the host call has returned.
                            unsafe { state.state() }.settle(cycle, v);
                            v
                        }
                    }
                }
            }
        } else if page < 0xDC {
            self.colour[(addr & 0x3FF) as usize]
        } else {
            let cia = &mut self.cia[(page != 0xDC) as usize];
            cia.read(addr, &mut self.sched, &mut self.ln)
        };
        self.last_read = value;
        value
    }

    pub fn io_write<H: Host>(&mut self, h: &mut H, addr: u16, value: u8) {
        let page = addr >> 8;
        if page == 0xDF && (self.reu.is_some() || self.dma) {
            if let Some(r) = self.reu.as_mut() {
                self.dma |= r.write(addr, value);
                self.reu_line();
            }
        } else if page < 0xD4 {
            self.vic.write(addr, value, &mut self.sched, &mut self.ln);
        } else if !(0xD8..0xDE).contains(&page) {
            let (cycle, reg) = (self.sched.time(PHI1), (addr & 0x1F) as u8);
            let traced = self.tracer().is_some_and(|t| t.sid_write(addr, value));
            match (traced, Self::sid(&self.sids, addr)) {
                (true, _) | (false, None) => {}
                (false, Some(&SidDev::Host(dev))) => {
                    self.flush(h);
                    h.sid_write(dev, cycle, reg, value);
                }
                // SAFETY: serialised by the machine's run claim; no reference outlives the call.
                (false, Some(SidDev::Logged(state, _))) => {
                    unsafe { state.state() }.write(cycle, reg, value)
                }
            }
        } else if page < 0xDC {
            self.colour[(addr & 0x3FF) as usize] = value & 0x0F;
        } else {
            let cia = &mut self.cia[(page != 0xDC) as usize];
            cia.write(addr, value, &mut self.sched, &mut self.ln);
        }
    }

    #[inline(always)]
    pub fn rd<H: Host>(&mut self, h: &mut H, addr: u16) -> u8 {
        if addr > 1 {
            let k = self.map[(addr >> 12) as usize];
            if k == IO {
                self.io_read(h, addr)
            } else {
                self.img[k as usize][addr as usize]
            }
        } else if addr != 0 {
            port_read(self.port_dir, self.port_data)
        } else {
            self.port_dir
        }
    }

    /// A write as the CPU makes it: I/O writes queue until a write cycle.
    #[inline(always)]
    pub fn wr(&mut self, addr: u16, value: u8) {
        if self.io && addr >> 12 == 0xD {
            self.pending.push_back((addr, value));
            return;
        }
        let mut value = value;
        if addr == 0xFF00 && self.reu.as_ref().is_some_and(|r| r.armed) {
            if self.rmw {
                value = self.img[self.map[0xF] as usize][0xFF00];
            }
            self.dma |= self.reu.as_mut().is_some_and(|r| r.ff00());
        }
        self.img[RAM][addr as usize] = value;
        if addr < 2 {
            if addr != 0 {
                self.port_data = value;
            } else {
                self.port_dir = value;
            }
            self.remap();
            self.port_dirty = true;
        }
    }
}

impl<H: Host> Port for BusRef<'_, H> {
    fn dma_read(&mut self, addr: u16) -> u8 {
        if addr < 2 {
            self.s.img[RAM][addr as usize]
        } else {
            self.s.rd(self.h, addr)
        }
    }

    fn dma_write(&mut self, addr: u16, value: u8) {
        if self.s.io && addr >> 12 == 0xD {
            self.s.io_write(self.h, addr, value);
        } else {
            let s = &mut *self.s;
            let old = std::mem::replace(&mut s.img[RAM][addr as usize], value);
            if let Some(r) = s.rec.as_deref_mut() {
                r.dma(addr, old, value, s.bits);
            }
        }
    }
}

/// The 6510's bus: [`Sys`] and the [`Host`] its devices live on.
pub struct BusRef<'a, H: Host> {
    pub s: &'a mut Sys,
    pub h: &'a mut H,
}

impl<H: Host> Bus for BusRef<'_, H> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.s.rd(self.h, addr)
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        let s = &mut *self.s;
        if let Some(r) = s.rec.as_deref_mut() {
            r.note(addr, s.img[RAM][addr as usize], false);
        }
        s.wr(addr, v)
    }
}
