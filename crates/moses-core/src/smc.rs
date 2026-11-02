//! The self-modifying-code recorder: each RAM byte's last writer, and every fetch or control
//! read whose byte departs from the image armed at lift time, as deduplicated event keys.
//!
//! A decode attributes the writes made since the previous decode to the instruction that made
//! them (or to the interrupt entry that pushed them), when they changed the byte; follows the
//! interrupt contexts by stack pointer; and checks the bytes the instruction at `pc` fetches or
//! reads as control (opcode, operands, indirect pointers with the `JMP ($xxFF)` page wrap,
//! `RTS`/`RTI` return bytes) under the live bank view.

use crate::sys::{banking, port_read, Sys, IO, RAM};
use crate::trace::{Scan, Tracer};

/// Writer kinds; [`W_HOST`] marks a byte the host wrote and is never reported.
pub const W_NONE: u8 = 0;
pub const W_CPU: u8 = 1;
pub const W_IRQ: u8 = 2;
pub const W_NMI: u8 = 3;
pub const W_BRK: u8 = 4;
pub const W_DMA: u8 = 5;
pub const W_PORT: u8 = 6;
pub const W_RESET: u8 = 7;
pub const W_HOST: u8 = 8;
pub const WRITERS: [&str; 8] = [
    "none", "cpu", "irq_push", "nmi_push", "brk_push", "reu_dma", "cpu_port", "cpu_port",
];

pub const R_OPCODE: u8 = 0;
pub const R_OPERAND: u8 = 1;
pub const R_PTR: u8 = 2;
pub const R_RET: u8 = 3;
pub const R_HWVEC: u8 = 4;
pub const R_KVEC: u8 = 5;
pub const R_BANK: u8 = 6;
pub const ROLES: [&str; 7] = [
    "opcode",
    "operand",
    "ptr_indirect",
    "ret_addr",
    "hw_vector",
    "kernal_vector",
    "bank_reg",
];

pub const T_PRE: u8 = 0;
pub const T_EARLIER: u8 = 1;
pub const T_SAME_CALL: u8 = 2;
pub const T_SAME_INSN: u8 = 3;
pub const T_ASYNC: u8 = 4;
pub const TIMINGS: [&str; 5] = [
    "pre_lift",
    "earlier_call",
    "same_call_earlier",
    "same_insn",
    "async",
];

pub const M_FLAT: u8 = 0;
pub const M_UNDER_ROM: u8 = 1;
pub const M_UNDER_IO: u8 = 2;
pub const M_ROM_VS_RAM: u8 = 3;
pub const MAPPINGS: [&str; 4] = ["flat", "ram_under_rom", "ram_under_io", "rom_vs_ram"];

/// Unclassified event kinds.
pub const U_UNOWNED: u8 = 0;
pub const U_UNATTRIBUTED: u8 = 1;
pub const UNCLASSIFIED: [&str; 2] = ["unowned", "unattributed"];
/// How many unclassified events are kept as a sample.
pub const SAMPLE: usize = 16;

const C_MAIN: u8 = 0;
const C_IRQ: u8 = 1;
const C_NMI: u8 = 2;

/// Opcode table entries (`opcode_table`): length in bits 0-1, pointer form in bits 2-3,
/// return pull in bits 4-5.
pub const P_IND: u8 = 1;
pub const P_INDY: u8 = 2;
pub const P_INDX: u8 = 3;
pub const RET_RTS: u8 = 1;
pub const RET_RTI: u8 = 2;

const V_RAM: u8 = 0;
const V_ROM: u8 = 1;
pub(crate) const V_IO: u8 = 2;
const KVEC: std::ops::Range<u16> = 0x0314..0x031A;
const IO_WINDOW: std::ops::RangeInclusive<u16> = 0xD000..=0xDFFF;
const NMI_VECTOR: u16 = 0xFFFA;
const IRQ_VECTOR: u16 = 0xFFFE;
/// A decode window holds an instruction's stores and one interrupt entry's pushes.
const WINDOW: usize = 8;
const KEYS: usize = 1 << 19;

/// The last writer of a byte, or of the port's bank bits.
#[derive(Clone, Copy, Default)]
struct Owner {
    call: i32,
    pc: u16,
    kind: u8,
    op: u8,
    ctx: u8,
    under_rom: bool,
}

const RESET: Owner = Owner {
    call: -1,
    pc: 0,
    kind: W_RESET,
    op: 0,
    ctx: C_MAIN,
    under_rom: false,
};

#[derive(Clone, Copy)]
struct Prev {
    pc: u16,
    op: u8,
    x: u8,
    sp: u8,
    bits: i32,
}

/// One deduplicated event: its key and the index (call, or frame once a frame clock is set)
/// it was first seen at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub key: u32,
    pub first: i64,
}

impl Event {
    /// `(writer, writer opcode, role, timing, mapping)` indices; the opcode is the store's
    /// for `cpu` and the port write's for `cpu_port`, `None` for a reset port.
    pub fn fields(&self) -> (u8, Option<u8>, u8, u8, u8) {
        let k = self.key;
        let writer = (k >> 16) as u8;
        let op = (writer == W_CPU || writer == W_PORT).then_some((k >> 8) as u8);
        (
            writer,
            op,
            (k >> 5 & 7) as u8,
            (k >> 2 & 7) as u8,
            (k & 3) as u8,
        )
    }
}

/// What the CPU sees at `addr` under port bits `bits`: RAM, ROM or I/O (`$00`/`$01` are I/O).
pub(crate) fn view(bits: i32, addr: u16) -> u8 {
    if addr < 2 {
        return V_IO;
    }
    let (basic, kernal, chargen, io) = banking(bits as u8);
    let rom = match addr >> 12 {
        0xA | 0xB => basic,
        0xD if io => return V_IO,
        0xD => chargen,
        0xE | 0xF => kernal,
        _ => false,
    };
    if rom {
        V_ROM
    } else {
        V_RAM
    }
}

/// A byte as the CPU sees it now, without a device read: RAM stands in under I/O.
pub(crate) fn peek(s: &Sys, addr: u16) -> (u8, u8) {
    if addr < 2 {
        let v = if addr == 0 {
            s.port_dir
        } else {
            port_read(s.port_dir, s.port_data)
        };
        return (v, V_IO);
    }
    let k = s.map[(addr >> 12) as usize];
    if k == IO {
        (s.img[RAM][addr as usize], V_IO)
    } else {
        let v = if k as usize == RAM { V_RAM } else { V_ROM };
        (s.img[k as usize][addr as usize], v)
    }
}

pub struct Recorder {
    base: Box<[u8; 0x10000]>,
    owners: Box<[Owner]>,
    table: [u8; 256],
    banked: bool,
    pub call: i32,
    /// Cycles per frame of the frame clock (0: events are indexed by call).
    pub frame_cycles: i64,
    t0: i64,
    index: i64,
    lift_bits: i32,
    ctx: u8,
    stack: [(u8, u8); 256],
    depth: usize,
    port: Owner,
    bits: i32,
    prev: Option<Prev>,
    win: [(u16, u8, bool); WINDOW],
    nwin: usize,
    /// The writer kind of the interrupt entry whose pushes the window holds.
    pub entry: u8,
    dirty: bool,
    seen: Box<[u64]>,
    pub events: Vec<Event>,
    pub unclassified: u64,
    pub sample: Vec<(u8, u16, u8, u16)>,
    /// A decode after this cycle stops the run and sets `timed_out`.
    pub deadline: i64,
    pub timed_out: bool,
    /// The lift-fact and record trace, when armed.
    pub tracer: Option<Box<Tracer>>,
    /// The interrupt-driven main-context scan, when armed.
    pub scan: Option<Box<Scan>>,
}

impl Recorder {
    /// A recorder of fetches against `base`; `table` is [`P_IND`]-style per opcode, `banked`
    /// classifies RAM written under ROM or the I/O window by its mapping.
    pub fn new(base: Box<[u8; 0x10000]>, table: [u8; 256], banked: bool) -> Recorder {
        Recorder {
            base,
            owners: vec![Owner::default(); 0x10000].into_boxed_slice(),
            table,
            banked,
            call: -1,
            frame_cycles: 0,
            t0: -1,
            index: -1,
            lift_bits: -1,
            ctx: C_MAIN,
            stack: [(0, 0); 256],
            depth: 0,
            port: RESET,
            bits: -1,
            prev: None,
            win: [(0, 0, false); WINDOW],
            nwin: 0,
            entry: W_NONE,
            dirty: false,
            seen: vec![0; KEYS / 64].into_boxed_slice(),
            events: Vec::new(),
            unclassified: 0,
            sample: Vec::new(),
            deadline: i64::MAX,
            timed_out: false,
            tracer: None,
            scan: None,
        }
    }

    /// End the pre-lift phase: the bank bits now are the lift-time map.
    pub fn begin(&mut self, bits: i32) {
        (self.call, self.lift_bits, self.index) = (0, bits, 0);
    }

    /// The next call: events are indexed by it unless a frame clock runs.
    pub fn next_call(&mut self) {
        self.call += 1;
        if self.frame_cycles == 0 {
            self.index = self.call as i64;
        }
    }

    /// Index events by frame from the next decode on: `(cycle - t0) / frame_cycles`, `t0`
    /// that decode's cycle.
    pub fn frame_clock(&mut self, frame_cycles: i64) {
        (self.frame_cycles, self.t0) = (frame_cycles, -1);
    }

    /// A CPU write (`push` for an entry or BRK push) of `addr`, which held `old`.
    #[inline(always)]
    pub fn note(&mut self, addr: u16, old: u8, push: bool) {
        for w in &mut self.win[..self.nwin] {
            if w.0 == addr {
                w.2 |= push;
                return;
            }
        }
        if self.nwin < WINDOW {
            self.win[self.nwin] = (addr, old, push);
            self.nwin += 1;
        } else {
            self.unclass(U_UNATTRIBUTED, addr, 0, self.prev.map_or(0, |p| p.pc));
        }
    }

    /// A DMA write of `addr` from `old` to `value` under port bits `bits`.
    pub fn dma(&mut self, addr: u16, old: u8, value: u8, bits: i32) {
        if old != value {
            self.owners[addr as usize] = Owner {
                call: self.call,
                pc: self.prev.map_or(0, |p| p.pc),
                kind: W_DMA,
                op: 0,
                ctx: self.ctx,
                under_rom: view(bits, addr) == V_ROM,
            };
            self.dirty = true;
        }
    }

    /// The host wrote `addr` (a call frame or the port): no instruction owns it; a port
    /// write resets the bank-bit writer and the map the next decode compares against.
    pub fn host(&mut self, addr: u16, bits: i32) {
        self.dirty = true;
        if addr < 2 {
            self.port = RESET;
            self.bits = bits;
        } else {
            self.owners[addr as usize] = Owner {
                kind: W_HOST,
                ..Owner::default()
            };
        }
    }

    fn unclass(&mut self, kind: u8, addr: u16, role: u8, pc: u16) {
        self.unclassified += 1;
        if self.sample.len() < SAMPLE {
            self.sample.push((kind, addr, role, pc));
        }
    }

    fn timing(&self, o: &Owner, pc: u16) -> u8 {
        if o.kind == W_IRQ || o.kind == W_NMI || o.ctx != C_MAIN {
            T_ASYNC
        } else if o.call < 0 {
            T_PRE
        } else if o.call < self.call {
            T_EARLIER
        } else if o.pc == pc && self.ctx == C_MAIN {
            T_SAME_INSN
        } else {
            T_SAME_CALL
        }
    }

    fn emit(&mut self, o: Owner, role: u8, pc: u16, mapping: u8) {
        let timing = self.timing(&o, pc);
        let key = (o.kind as u32) << 16
            | (o.op as u32) << 8
            | (role as u32) << 5
            | (timing as u32) << 2
            | mapping as u32;
        let (word, bit) = ((key >> 6) as usize, 1u64 << (key & 63));
        if self.seen[word] & bit == 0 {
            self.seen[word] |= bit;
            self.events.push(Event {
                key,
                first: self.index,
            });
        }
    }

    /// A byte fetched or read as control by the instruction at `pc`.
    fn check(&mut self, addr: u16, (value, fview): (u8, u8), role: u8, pc: u16, from_rom: bool) {
        if self.call < 0 {
            return;
        }
        let role = if role == R_PTR && from_rom && KVEC.contains(&addr) {
            R_KVEC
        } else {
            role
        };
        if fview != V_RAM {
            if fview == V_IO || value != self.base[addr as usize] {
                self.emit(self.port, role, pc, M_ROM_VS_RAM);
            }
            return;
        }
        let o = self.owners[addr as usize];
        match o.kind {
            W_NONE => {
                if value != self.base[addr as usize] {
                    self.unclass(U_UNOWNED, addr, role, pc);
                }
            }
            W_HOST => {}
            _ => {
                let mapping = if self.banked && o.under_rom {
                    M_UNDER_ROM
                } else if self.banked && IO_WINDOW.contains(&addr) {
                    M_UNDER_IO
                } else {
                    M_FLAT
                };
                self.emit(o, role, pc, mapping);
            }
        }
    }

    /// Give the window's changed bytes to the instruction `p` (or the entry that pushed them).
    fn attribute(&mut self, s: &Sys, p: Prev) {
        for i in 0..std::mem::take(&mut self.nwin) {
            let (addr, old, push) = self.win[i];
            if s.img[RAM][addr as usize] == old {
                continue;
            }
            let kind = if !push {
                W_CPU
            } else if self.entry != W_NONE {
                self.entry
            } else {
                self.unclass(U_UNATTRIBUTED, addr, 0, p.pc);
                continue;
            };
            self.owners[addr as usize] = Owner {
                call: self.call,
                pc: p.pc,
                kind,
                op: if kind == W_CPU { p.op } else { 0 },
                ctx: self.ctx,
                under_rom: view(p.bits, addr) == V_ROM,
            };
        }
    }

    /// The decode of the instruction at the CPU's `pc` (registers `a`, `x`, `y`, `sp`) at
    /// cycle `t`; true stops the run.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(&mut self, s: &Sys, pc: u16, a: u8, x: u8, y: u8, sp: u8, t: i64) -> bool {
        if self.frame_cycles > 0 {
            if self.t0 < 0 {
                self.t0 = t;
            }
            self.index = (t - self.t0) / self.frame_cycles;
        }
        let stop = t > self.deadline;
        self.timed_out |= stop;
        if let Some(tr) = self.tracer.as_deref_mut() {
            tr.decode(s, pc, a, x, y, sp, self.timed_out);
        }
        if let Some(p) = self.prev {
            let idle = p.pc == pc && p.x == x && p.sp == sp && p.bits == s.bits;
            if idle && self.nwin == 0 && self.entry == W_NONE && !self.dirty {
                self.scan(s, pc, x, y, t);
                return stop;
            }
            self.attribute(s, p);
            if self.bits >= 0 && s.bits != self.bits {
                self.port = Owner {
                    call: self.call,
                    pc: p.pc,
                    kind: W_PORT,
                    op: p.op,
                    ctx: self.ctx,
                    under_rom: false,
                };
            }
        }
        self.dirty = false;
        self.bits = s.bits;
        while self.depth > 0 && sp > self.stack[self.depth - 1].1 {
            self.depth -= 1;
        }
        let top = if self.depth > 0 {
            self.stack[self.depth - 1].0
        } else {
            C_MAIN
        };
        let entry = std::mem::take(&mut self.entry);
        if entry != W_NONE {
            let ctx = match entry {
                W_IRQ => C_IRQ,
                W_NMI => C_NMI,
                _ => top,
            };
            if self.depth < self.stack.len() {
                self.stack[self.depth] = (ctx, sp);
                self.depth += 1;
            }
            let va = if entry == W_NMI {
                NMI_VECTOR
            } else {
                IRQ_VECTOR
            };
            for a in [va, va + 1] {
                self.check(a, peek(s, a), R_HWVEC, pc, false);
            }
        }
        self.ctx = if self.depth > 0 {
            self.stack[self.depth - 1].0
        } else {
            C_MAIN
        };
        let fetched = [0u16, 1, 2].map(|k| peek(s, pc.wrapping_add(k)));
        let op = fetched[0].0;
        let from_rom = fetched[0].1 == V_ROM;
        let d = self.table[op as usize];
        self.check(pc, fetched[0], R_OPCODE, pc, from_rom);
        for k in 1..(d & 3) as u16 {
            self.check(
                pc.wrapping_add(k),
                fetched[k as usize],
                R_OPERAND,
                pc,
                from_rom,
            );
        }
        let b1 = fetched[1].0;
        let ptrs = match d >> 2 & 3 {
            P_IND => {
                let w = b1 as u16 | (fetched[2].0 as u16) << 8;
                Some([w, w & 0xFF00 | (w as u8).wrapping_add(1) as u16])
            }
            P_INDY => Some([b1 as u16, b1.wrapping_add(1) as u16]),
            P_INDX => {
                let p = b1.wrapping_add(x);
                Some([p as u16, p.wrapping_add(1) as u16])
            }
            _ => None,
        };
        for a in ptrs.into_iter().flatten() {
            self.check(a, peek(s, a), R_PTR, pc, from_rom);
        }
        let ret = d >> 4 & 3;
        if ret != 0 {
            for i in 0..2 {
                let a = 0x100 | sp.wrapping_add(ret + i) as u16;
                self.check(a, peek(s, a), R_RET, pc, from_rom);
            }
        }
        if self.call >= 0 && s.bits != self.lift_bits {
            self.emit(self.port, R_BANK, pc, M_FLAT);
        }
        self.scan(s, pc, x, y, t);
        self.prev = Some(Prev {
            pc,
            op,
            x,
            sp,
            bits: s.bits,
        });
        stop
    }

    fn scan(&mut self, s: &Sys, pc: u16, x: u8, y: u8, t: i64) {
        let main = self.ctx == C_MAIN;
        if let Some(sc) = self.scan.as_deref_mut() {
            sc.decode(s, pc, x, y, main, t);
        }
    }

    /// Attribute the last instruction's writes once the run is over; pushes of an entry
    /// whose handler was never decoded stay unattributed.
    pub fn finish(&mut self, s: &Sys) {
        self.entry = W_NONE;
        if let Some(p) = self.prev {
            self.attribute(s, p);
        }
        if let Some(tr) = self.tracer.as_deref_mut() {
            tr.finish(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_follow_the_pla() {
        assert_eq!(view(7, 0xA000), V_ROM);
        assert_eq!(view(6, 0xA000), V_RAM);
        assert_eq!(view(7, 0xD000), V_IO);
        assert_eq!(view(3, 0xD000), V_ROM);
        assert_eq!(view(0, 0xD000), V_RAM);
        assert_eq!(view(5, 0xE000), V_RAM);
        assert_eq!(view(7, 0x0001), V_IO);
        assert_eq!(view(7, 0x8000), V_RAM);
    }

    #[test]
    fn an_event_key_round_trips_and_dedupes() {
        let mut r = Recorder::new(Box::new([0; 0x10000]), [0; 256], true);
        r.begin(7);
        let o = Owner {
            call: 0,
            pc: 0x1000,
            kind: W_CPU,
            op: 0x8D,
            ctx: C_MAIN,
            under_rom: false,
        };
        r.emit(o, R_OPERAND, 0x1000, M_FLAT);
        r.emit(o, R_OPERAND, 0x1000, M_FLAT);
        r.next_call();
        r.emit(o, R_OPERAND, 0x1000, M_FLAT);
        r.emit(RESET, R_BANK, 0x1000, M_FLAT);
        let got: Vec<_> = r.events.iter().map(|e| (e.fields(), e.first)).collect();
        assert_eq!(
            got,
            [
                ((W_CPU, Some(0x8D), R_OPERAND, T_SAME_INSN, M_FLAT), 0),
                ((W_CPU, Some(0x8D), R_OPERAND, T_EARLIER, M_FLAT), 1),
                ((W_RESET, None, R_BANK, T_PRE, M_FLAT), 1),
            ]
        );
    }

    #[test]
    fn a_full_window_counts_the_overflow() {
        let mut r = Recorder::new(Box::new([0; 0x10000]), [0; 256], false);
        for a in 0..=WINDOW as u16 {
            r.note(0x200 + a, 0, false);
        }
        r.note(0x200, 0, true);
        assert_eq!((r.nwin, r.unclassified), (WINDOW, 1));
        assert!(r.win[0].2);
    }
}
