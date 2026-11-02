//! The cycle loop: libsidplayfp's `MOS6510` cycle tables, RDY stalls and interrupt delay
//! around jennings' instruction semantics.

use std::collections::VecDeque;

use jennings_core::{AddrSet, Cpu};

use crate::cia::Cia;
use crate::lines::{self, Lines, MAX};
use crate::reu::Reu;
use crate::sched::Sched;
use crate::smc;
use crate::sys::{BusRef, Host, SidDev, Sys, RAM};
use crate::vic::Vic;

// Entry kinds of libsidplayfp's per-opcode cycle tables; those below `R` write.
pub const W: u8 = 0;
pub const PUSHPCH: u8 = 1;
pub const LOWPC: u8 = 2;
pub const PUSHSR: u8 = 3;
pub const R: u8 = 4;
pub const X2: u8 = 5;
pub const F: u8 = 6;
pub const CLI: u8 = 7;
pub const SEI: u8 = 8;
pub const POPSR: u8 = 9;
pub const BR: u8 = 10;
pub const BRK0: u8 = 11;
pub const VECLO: u8 = 12;
pub const VECHI: u8 = 13;
pub const FN: u8 = 14;
pub const RTI: u8 = 15;
pub const HALT: u8 = 16;
pub const ANE: u8 = 17;
pub const BRK_ENTRIES: [u8; 7] = [BRK0, PUSHPCH, LOWPC, PUSHSR, VECLO, VECHI, FN];
/// Index of [`BRK_ENTRIES`] in the layout table.
pub const BRK_LAYOUT: usize = 256;

/// What `Machine._decode` does for an opcode beyond laying out its cycles.
pub const K_NONE: u8 = 0;
pub const K_RMW: u8 = 1;
pub const K_ABSX: u8 = 2;
pub const K_ABSY: u8 = 3;
pub const K_INDY: u8 = 4;
pub const K_BRANCH: u8 = 5;

const ANE_MAGIC: u8 = 0xEF;
const CLI_BASE: i32 = 0x58 << 3;
const SEI_BASE: i32 = 0x78 << 3;
const IRQ_I: u8 = 0x04;
const NMI_VECTOR: u16 = 0xFFFA;
const IRQ_VECTOR: u16 = 0xFFFE;

/// One opcode's cycles after its fetch (`Machine.LAYOUTS`).
#[derive(Clone, Default)]
pub struct Layout {
    pub ents: Vec<u8>,
    pub exec_at: i32,
    pub kind: u8,
    pub base: i32,
    /// `(flag mask, wanted value)` of a conditional branch.
    pub branch: Option<(u8, bool)>,
}

/// The addresses a trap is called at.
pub enum Watch {
    All,
    Set(AddrSet),
}

impl Watch {
    #[inline(always)]
    fn contains(&self, pc: u16) -> bool {
        match self {
            Watch::All => true,
            Watch::Set(s) => s.contains(pc),
        }
    }
}

pub struct Config {
    pub ntsc: bool,
    pub cia_new: bool,
    pub ram: Box<[u8; 0x10000]>,
    /// BASIC, CHARGEN and KERNAL as 64 KiB images indexed by absolute address.
    pub roms: [Box<[u8; 0x10000]>; 3],
    pub layouts: Vec<Layout>,
    pub sids: Vec<(u32, SidDev)>,
    pub custom_sink: bool,
    pub lines: i32,
    pub cycles_per_line: i32,
    /// The size in KiB of an REU at `$DF00`, if any (one of [`crate::reu::SIZES`]).
    pub reu: Option<u32>,
}

pub struct Core {
    pub cpu: Cpu,
    pub s: Sys,
    layouts: Vec<Layout>,
    layout: usize,
    idx: i32,
    base: i32,
    exec_at: i32,
    rmw: bool,
    taken: bool,
    cross: bool,
    vec_addr: u16,
    target: u16,
    d1x1: bool,
    pub t: i64,
    pub jammed: bool,
    pub stop: bool,
    pub ntsc: bool,
    pub armed: bool,
    pub arm_pc: Option<u16>,
    pub watch: Watch,
    pub trap_set: bool,
    pub decode_set: bool,
    pub port_set: bool,
}

impl Core {
    pub fn new(cfg: Config) -> Core {
        let [basic, chargen, kernal] = cfg.roms;
        let mut sched = Sched::new();
        let mut vic = Vic::new(cfg.ntsc, cfg.lines, cfg.cycles_per_line);
        let mut cia = [
            Cia::new(0, false, cfg.cia_new),
            Cia::new(1, true, cfg.cia_new),
        ];
        vic.reset(&mut sched);
        cia[0].reset(&mut sched);
        cia[1].reset(&mut sched);
        vic.reset(&mut sched);
        let (dir, data) = (cfg.ram[0], cfg.ram[1]);
        let mut s = Sys {
            sched,
            vic,
            cia,
            ln: Lines::new(),
            img: [cfg.ram, basic, chargen, kernal],
            map: [RAM as u8; 16],
            colour: Box::new([0; 0x400]),
            port_dir: dir,
            port_data: data,
            bits: -1,
            io: false,
            pending: VecDeque::new(),
            last_read: 0,
            seed: 3686734,
            sids: cfg.sids,
            events: Vec::new(),
            custom_sink: cfg.custom_sink,
            port_dirty: false,
            reu: cfg.reu.and_then(Reu::new).map(Box::new),
            dma: false,
            rmw: false,
            rec: None,
        };
        s.remap();
        let mut layouts = cfg.layouts;
        layouts.resize(BRK_LAYOUT, Layout::default());
        layouts.push(Layout {
            ents: BRK_ENTRIES.to_vec(),
            ..Layout::default()
        });
        Core {
            cpu: Cpu::new(),
            s,
            layouts,
            layout: BRK_LAYOUT,
            idx: 6,
            base: 0,
            exec_at: 0,
            rmw: false,
            taken: false,
            cross: false,
            vec_addr: 0,
            target: 0,
            d1x1: false,
            t: 0,
            jammed: false,
            stop: false,
            ntsc: cfg.ntsc,
            armed: false,
            arm_pc: None,
            watch: Watch::Set(AddrSet::default()),
            trap_set: false,
            decode_set: false,
            port_set: false,
        }
    }

    /// Bring the chips to `cycle` and start the CPU at `entry` (`Machine.reset`).
    pub fn reset<H: Host>(&mut self, h: &mut H, entry: u16, cycle: i64) {
        self.s.run_sched(cycle << 1);
        self.s.flush(h);
        self.t = cycle;
        self.poke(0, 0x2F);
        self.poke(1, 0x37);
        let cpu = &mut self.cpu;
        (cpu.sp, cpu.a, cpu.x, cpu.y) = (0xFF, 0, 0, 0);
        cpu.set_p(0);
        self.target = entry;
        self.layout = BRK_LAYOUT;
        self.idx = 6;
        self.base = 0;
        self.s.ln.cc = 6;
    }

    /// `machine[addr]` from the host.
    pub fn peek<H: Host>(&mut self, h: &mut H, addr: u16) -> u8 {
        let v = self.s.look(h, addr);
        self.s.flush(h);
        v
    }

    /// `machine[addr] = value` from the host.
    pub fn poke(&mut self, addr: u16, value: u8) {
        self.s.wr(addr, value);
    }

    fn check(&self) -> bool {
        self.s.ln.check()
    }

    /// Fetch the opcode at PC (`fetchNextOpcode`) and lay out its cycles.
    fn decode<H: Host>(&mut self, h: &mut H) {
        if self.decode_set {
            self.s.flush(h);
            h.decode(self);
        }
        self.record();
        if self.trap_set {
            if !self.armed && self.arm_pc == Some(self.cpu.pc) {
                self.armed = true;
                self.watch = Watch::All;
            }
            while self.watch.contains(self.cpu.pc) {
                self.s.flush(h);
                if !h.trap(self) {
                    break;
                }
                self.record();
            }
        }
        let pc = self.cpu.pc;
        let op = self.s.look(h, pc) as usize;
        let (kind, base, branch) = {
            let l = &self.layouts[op];
            (l.kind, l.base, l.branch)
        };
        self.layout = op;
        self.exec_at = self.layouts[op].exec_at;
        self.base = base;
        self.idx = 0;
        self.rmw = kind == K_RMW;
        self.s.rmw = self.rmw;
        if let Some((mask, want)) = branch {
            self.taken = (self.cpu.p() & mask != 0) == want;
            let rel = self.s.look(h, pc.wrapping_add(1));
            let nxt = pc.wrapping_add(2);
            let target = nxt.wrapping_add(rel as i8 as u16);
            self.cross = (target ^ nxt) & 0xFF00 != 0;
        } else if kind != K_NONE {
            let mut lo = self.s.look(h, pc.wrapping_add(1));
            if kind == K_INDY {
                lo = self.s.look(h, lo as u16);
            }
            let idx = if kind == K_ABSX {
                self.cpu.x
            } else {
                self.cpu.y
            };
            self.cross = lo as u32 + idx as u32 > 0xFF;
        }
        if !self.check() {
            self.s.ln.ic = MAX;
        } else if self.s.ln.ic != MAX {
            self.s.ln.ic = -MAX;
        }
    }

    /// `interruptsAndNextOpcode`: start an interrupt or the next opcode.
    fn fetch<H: Host>(&mut self, h: &mut H) {
        let ln = &mut self.s.ln;
        if ln.ic != MAX && ln.cc > ln.ic + 2 {
            self.layout = BRK_LAYOUT;
            self.idx = 0;
            self.base = 0;
            self.d1x1 = true;
            ln.ic = MAX;
        } else {
            self.decode(h);
        }
    }

    /// The SMC recorder's look at the instruction about to be decoded.
    #[inline(always)]
    fn record(&mut self) {
        if let Some(mut r) = self.s.rec.take() {
            let c = &self.cpu;
            self.stop |= r.decode(&self.s, c.pc, c.a, c.x, c.y, c.sp, self.t);
            self.s.rec = Some(r);
        }
    }

    fn push(&mut self, value: u8) {
        let addr = 0x100 | self.cpu.sp as usize;
        if let Some(r) = self.s.rec.as_deref_mut() {
            r.note(addr as u16, self.s.img[RAM][addr], true);
        }
        self.s.img[RAM][addr] = value;
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    fn stall(&mut self, cc: i32) {
        let ln = &mut self.s.ln;
        if cc == CLI_BASE {
            ln.iflag = false;
            if ln.pin && ln.ic == MAX {
                ln.ic = -MAX;
            }
        } else if cc == SEI_BASE {
            ln.iflag = true;
            if !ln.nmi && cc <= ln.ic + 2 {
                ln.ic = MAX;
            }
        }
        if ln.ic == cc {
            ln.ic = cc - 1;
        }
    }

    fn special<H: Host>(&mut self, h: &mut H, kind: u8, t: i64) {
        match kind {
            F => self.fetch(h),
            W => {
                if let Some((addr, value)) = self.s.pending.pop_front() {
                    self.s.io_write(h, addr, value);
                }
            }
            X2 => {
                if !self.cross {
                    self.idx += 1;
                    self.s.ln.cc += 1;
                }
            }
            BR => {
                if !self.taken {
                    self.fetch(h);
                } else if !self.cross {
                    self.idx += 1;
                    self.s.ln.cc += 1;
                    if self.s.ln.ic >> 3 == self.s.ln.cc >> 3 {
                        self.s.ln.ic += 2;
                    }
                }
            }
            CLI => {
                self.s.ln.iflag = false;
                self.s.ln.calc();
                self.fetch(h);
            }
            SEI => {
                self.s.ln.iflag = true;
                self.fetch(h);
                if !self.s.ln.nmi && self.s.ln.ic != MAX {
                    self.s.ln.ic = MAX;
                }
            }
            POPSR => {
                self.s.ln.iflag = self.cpu.p() & IRQ_I != 0;
                self.s.ln.calc();
            }
            RTI => {
                self.s.ln.emit(t + 1, lines::EV_RTI);
                self.fetch(h);
            }
            BRK0 => {
                if !self.d1x1 {
                    self.cpu.pc = self.cpu.pc.wrapping_add(2);
                }
            }
            PUSHPCH => self.push((self.cpu.pc >> 8) as u8),
            LOWPC => {
                self.push(self.cpu.pc as u8);
                let nmi = self.s.ln.nmi;
                self.vec_addr = if nmi { NMI_VECTOR } else { IRQ_VECTOR };
                self.s.ln.nmi = false;
                self.s.ln.calc();
                let name = if nmi {
                    lines::EV_ENTRY + lines::NMI as u8
                } else if self.d1x1 {
                    lines::EV_ENTRY + self.s.ln.irq_cause as u8
                } else {
                    lines::EV_BRK_ENTRY
                };
                if let Some(r) = self.s.rec.as_deref_mut() {
                    r.entry = if nmi {
                        smc::W_NMI
                    } else if self.d1x1 {
                        smc::W_IRQ
                    } else {
                        smc::W_BRK
                    };
                }
                self.s.ln.emit(t + 1, name);
            }
            PUSHSR => {
                let p = self.cpu.p() & !0x30 | if self.d1x1 { 0x20 } else { 0x30 };
                self.push(p);
            }
            VECLO => {
                self.target = self.s.rd(h, self.vec_addr) as u16;
                self.d1x1 = false;
            }
            VECHI => {
                self.target |= (self.s.rd(h, self.vec_addr.wrapping_add(1)) as u16) << 8;
                self.s.ln.iflag = true;
                self.cpu.i = 1;
            }
            FN => {
                self.cpu.pc = self.target;
                self.decode(h);
            }
            HALT => self.jammed = true,
            _ => {
                let operand = self.s.rd(h, self.cpu.pc.wrapping_add(1));
                let value = (self.cpu.a | ANE_MAGIC) & self.cpu.x & operand;
                self.cpu.a = value;
                let p = self.cpu.p() & 0x7D | value & 0x80 | if value == 0 { 0x02 } else { 0 };
                self.cpu.set_p(p);
                self.cpu.pc = self.cpu.pc.wrapping_add(2);
            }
        }
    }

    /// Run the CPU and the chips up to cycle `limit`.
    pub fn run<H: Host>(&mut self, h: &mut H, limit: i64) {
        self.stop = false;
        let mut t = self.t;
        while t < limit && !self.jammed && !self.stop && !h.failed() {
            let t2 = t << 1;
            if self.s.sched.next <= t2 {
                self.s.ln.cc = self.base + self.idx;
                self.s.run_sched(t2);
                self.s.flush(h);
            }
            let kind = self.layouts[self.layout].ents[self.idx as usize];
            self.s.sched.now = t2 + 1;
            if kind >= R && (!self.s.ln.rdy || self.s.dma) {
                let mut go = false;
                if self.s.dma {
                    self.s.ln.cc = self.base + self.idx;
                    go = self.s.reu_step(h, !self.s.ln.rdy);
                    self.s.dma = !go;
                    self.s.flush(h);
                }
                if !go {
                    self.stall(self.base + self.idx);
                    t += 1;
                    continue;
                }
            }
            self.idx += 1;
            self.s.ln.cc = self.base + self.idx;
            if kind == R {
                if self.idx == self.exec_at {
                    self.step(h);
                }
            } else {
                self.t = t;
                self.special(h, kind, t);
            }
            self.s.flush(h);
            t += 1;
        }
        self.t = t;
    }

    fn step<H: Host>(&mut self, h: &mut H) {
        self.cpu.step(&mut BusRef { s: &mut self.s, h });
        let s = &mut self.s;
        if self.rmw {
            if let Some(&(addr, _)) = s.pending.front() {
                s.pending.push_front((addr, s.last_read));
            }
        }
        if std::mem::take(&mut s.port_dirty) && self.armed && self.port_set {
            s.flush(h);
            h.port_write(self);
        }
    }
}
