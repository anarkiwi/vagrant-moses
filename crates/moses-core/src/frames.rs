//! Interrupt frames: each NMI or IRQ from its entry to its RTI is a record, and an interrupt
//! taken inside one is an event of it, at the instruction boundary the machine took it at.
//!
//! The pushed return address of an entered handler is rewritten to the sentinel, whose trap
//! resumes the interrupted code, so the handler's RTI is seen with no per-instruction hook. A
//! context stack counts the instructions each context retires: an event's `icount` is the
//! interrupted context's count at the entry.

use std::collections::BTreeSet;

use crate::calls::{host_write, Records, HALT_CYCLES, HALT_JAM, HALT_RETURN};
use crate::machine::Core;
use crate::smc::STOP_NONE;
use crate::sys::RAM;
use crate::trace::{PH_CALL, PH_NONE};

/// Interrupt sources by kind: NMI, then the IRQ sources of [`crate::lines`].
pub const SOURCES: [&str; 4] = ["nmi", "video_irq", "cia_irq", "reu_irq"];
/// The kind of a BRK entry, which is no interrupt frame.
pub const K_BRK: u8 = 4;
/// No interrupt entry is pending.
pub const K_NONE: u8 = 0xFF;
pub const EVENT_BYTES: usize = 21;
pub const ROW_BYTES: usize = 18;

/// An interrupt taken inside a record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Event {
    pub record: u32,
    pub icount: u32,
    pub pos: u32,
    pub at: u16,
    /// `[a, x, y, sp, p]` before the interrupt.
    pub regs: [u8; 5],
    pub depth: u8,
    pub kind: u8,
}

impl Event {
    /// `record`, `icount`, `pos` (`u32`), `at` (`u16`), the registers, `depth`, `kind`;
    /// little-endian.
    pub fn bytes(&self) -> [u8; EVENT_BYTES] {
        let mut b = [0; EVENT_BYTES];
        b[0..4].copy_from_slice(&self.record.to_le_bytes());
        b[4..8].copy_from_slice(&self.icount.to_le_bytes());
        b[8..12].copy_from_slice(&self.pos.to_le_bytes());
        b[12..14].copy_from_slice(&self.at.to_le_bytes());
        b[14..19].copy_from_slice(&self.regs);
        (b[19], b[20]) = (self.depth, self.kind);
        b
    }
}

/// A frame's entry state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Row {
    pub at: u16,
    /// The interrupted pc.
    pub pc: u16,
    pub regs: [u8; 5],
    pub kind: u8,
    pub insns: u32,
    pub cycles: u32,
}

impl Row {
    /// `at`, `pc` (`u16`), the registers, `kind`, `insns`, `cycles` (`u32`); little-endian.
    pub fn bytes(&self) -> [u8; ROW_BYTES] {
        let mut b = [0; ROW_BYTES];
        b[0..2].copy_from_slice(&self.at.to_le_bytes());
        b[2..4].copy_from_slice(&self.pc.to_le_bytes());
        b[4..9].copy_from_slice(&self.regs);
        b[9] = self.kind;
        b[10..14].copy_from_slice(&self.insns.to_le_bytes());
        b[14..18].copy_from_slice(&self.cycles.to_le_bytes());
        b
    }
}

/// The frame an interrupt entry pushed, as the stack holds it.
pub struct Pushed {
    pub at: u16,
    pub ret: u16,
    /// The registers before the interrupt.
    pub regs: [u8; 5],
    /// The stack pointer after the pushes.
    pub sp: u8,
    pub kind: u8,
}

impl Pushed {
    pub fn read(core: &Core, kind: u8) -> Pushed {
        let (c, sp) = (&core.cpu, core.cpu.sp);
        let byte = |k: u8| core.s.img[RAM][0x100 | sp.wrapping_add(k) as usize];
        Pushed {
            at: c.pc,
            ret: u16::from_le_bytes([byte(2), byte(3)]),
            regs: [c.a, c.x, c.y, sp.wrapping_add(3), byte(1)],
            sp,
            kind,
        }
    }
}

#[derive(Clone, Copy)]
struct Ctx {
    resume: u16,
    sp: u8,
    /// Instructions retired before the context last gave way, and the decode count it resumed
    /// at.
    acc: u64,
    mark: u64,
}

fn insns(core: &Core) -> u64 {
    core.s.rec.as_deref().map_or(0, |r| r.insns)
}

/// The open interrupt contexts of a unit, outermost first, with the instructions each retired.
/// Contexts below `floor` are the unit's own body, which no return pops.
pub struct Intr {
    stack: Vec<Ctx>,
    floor: usize,
    sentinel: u16,
    pub events: Vec<Event>,
}

impl Intr {
    pub fn new(floor: usize, sentinel: u16) -> Intr {
        Intr {
            stack: Vec::new(),
            floor,
            sentinel,
            events: Vec::new(),
        }
    }

    /// Whether a unit is open.
    pub fn open(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Whether an interrupt context is open (the sentinel's return is its RTI).
    pub fn nested(&self) -> bool {
        self.stack.len() > self.floor
    }

    /// A new unit whose body, entered at stack pointer `sp`, retires from the decode count
    /// `now`.
    pub fn start(&mut self, now: u64, resume: u16, sp: u8) {
        self.stack.clear();
        self.events.clear();
        self.push(now, resume, sp);
    }

    pub fn clear(&mut self) {
        self.stack.clear();
        self.events.clear();
    }

    fn count(&self, now: u64) -> u32 {
        self.stack
            .last()
            .map_or(0, |c| c.acc + now - c.mark)
            .min(u32::MAX as u64) as u32
    }

    fn push(&mut self, now: u64, resume: u16, sp: u8) {
        if let Some(c) = self.stack.last_mut() {
            c.acc += now - c.mark;
        }
        self.stack.push(Ctx {
            resume,
            sp,
            acc: 0,
            mark: now,
        });
    }

    /// The innermost context returns at decode count `now`: its resume pc, and the stack
    /// pointer of the context now innermost, which resumes counting.
    fn unwind(&mut self, now: u64) -> (u16, Option<u8>) {
        let resume = self.stack.pop().map_or(0, |c| c.resume);
        let below = self.stack.last_mut().map(|c| {
            c.mark = now;
            c.sp
        });
        (resume, below)
    }

    /// The innermost context returns: its resume pc. The context below is traced from here.
    pub fn pop(&mut self, core: &mut Core) -> u16 {
        let (resume, below) = self.unwind(insns(core));
        if let Some(sp) = below {
            if let Some(tr) = core
                .s
                .rec
                .as_deref_mut()
                .and_then(|r| r.tracer.as_deref_mut())
            {
                tr.cut();
                tr.sp0 = sp;
            }
        }
        resume
    }

    /// Enter the handler `p` pushed: its return is the sentinel's trap, resuming `resume`.
    pub fn descend(&mut self, core: &mut Core, p: &Pushed, resume: u16) {
        let at = |k: u8| 0x100 | p.sp.wrapping_add(k) as u16;
        host_write(core, at(2), self.sentinel as u8);
        host_write(core, at(3), (self.sentinel >> 8) as u8);
        self.push(insns(core), resume, p.sp);
        if let Some(tr) = core
            .s
            .rec
            .as_deref_mut()
            .and_then(|r| r.tracer.as_deref_mut())
        {
            tr.cut();
            (tr.phase, tr.sp0) = (PH_CALL, p.sp);
        }
    }

    /// The interrupt `p` taken inside the open unit: an event of it, then its context.
    pub fn nest(&mut self, core: &mut Core, p: &Pushed) {
        let pos = core
            .s
            .rec
            .as_deref()
            .and_then(|r| r.tracer.as_deref())
            .map_or(0, |t| t.logged());
        self.events.push(Event {
            record: 0,
            icount: self.count(insns(core)),
            pos: pos as u32,
            at: p.at,
            regs: p.regs,
            depth: (self.stack.len() - 1).min(255) as u8,
            kind: p.kind,
        });
        self.descend(core, p, p.ret);
    }

    /// The unit's events, as records of index `record`.
    pub fn drain(&mut self, record: u32, into: &mut Vec<Event>) {
        into.extend(self.events.drain(..).map(|e| Event { record, ..e }));
    }
}

/// The records, entry states and events of the closed frames.
#[derive(Default)]
pub struct Chunk {
    pub records: Records,
    pub entries: Vec<Row>,
}

pub struct Frames {
    ix: Intr,
    row: Row,
    pub chunk: Chunk,
    /// The RAM at the first entry.
    pub image: Option<Vec<u8>>,
    /// RAM outside page 1 changed between two frames.
    pub dirty: bool,
    /// The kinds taken outside any frame (bit `k`), BRK included.
    pub foreign: u8,
    pub closed: u64,
    /// Whether interrupts are still recorded.
    pub live: bool,
    /// `(pc, sp)` of each handler entry, the 3-byte interrupt frame pushed.
    pub entry_sp: BTreeSet<(u16, u8)>,
    /// `(pc, port dir, port data)` of each handler entry.
    pub entry_port: BTreeSet<(u16, u8, u8)>,
}

impl Frames {
    /// Frames whose handlers return to `sentinel`.
    pub fn new(sentinel: u16) -> Frames {
        Frames {
            ix: Intr::new(0, sentinel),
            row: Row::default(),
            chunk: Chunk::default(),
            image: None,
            dirty: false,
            foreign: 0,
            closed: 0,
            live: true,
            entry_sp: BTreeSet::new(),
            entry_port: BTreeSet::new(),
        }
    }

    /// Whether a frame is open.
    pub fn open(&self) -> bool {
        self.ix.open()
    }

    /// The closed frames' records, emptied: the next frame's changes are against their RAM.
    pub fn take(&mut self) -> Chunk {
        std::mem::take(&mut self.chunk)
    }

    /// The sentinel's trap: the innermost handler's RTI resumes the interrupted code.
    pub fn trap(&mut self, core: &mut Core) -> bool {
        if !self.live || !self.ix.nested() || core.cpu.pc != self.ix.sentinel {
            return false;
        }
        let resume = self.pop(core, None);
        core.cpu.pc = resume;
        true
    }

    fn pop(&mut self, core: &mut Core, end: Option<[u8; 5]>) -> u16 {
        let resume = self.ix.pop(core);
        if !self.ix.open() {
            self.close(core, HALT_RETURN, end, core.t);
        }
        resume
    }

    /// An interrupt of `kind` was entered, the CPU at its handler.
    pub fn enter(&mut self, core: &mut Core, kind: u8) {
        if !self.live {
            return;
        }
        if kind == K_BRK {
            if !self.ix.open() {
                self.foreign |= 1 << K_BRK;
            }
            return;
        }
        let mut p = Pushed::read(core, kind);
        if self.ix.nested() && p.ret == self.ix.sentinel {
            p.ret = self.pop(core, Some(p.regs));
        }
        if self.ix.open() {
            self.ix.nest(core, &p);
        } else {
            self.begin(core, &p);
        }
        self.entry_sp.insert((p.at, p.sp));
        self.entry_port
            .insert((p.at, core.s.port_dir, core.s.port_data));
    }

    fn begin(&mut self, core: &mut Core, p: &Pushed) {
        let Some(mut tr) = core.s.rec.as_deref_mut().and_then(|r| r.tracer.take()) else {
            return;
        };
        let ram = &core.s.img[RAM];
        self.dirty |= self.closed > 0 && tr.differs_outside(ram, 1);
        let rs = &mut self.chunk.records;
        if rs.meta.is_empty() && rs.origin.is_empty() {
            tr.snap(&core.s);
            rs.origin = ram.to_vec();
        }
        if self.image.is_none() {
            self.image = Some(ram.to_vec());
        }
        if let Some(r) = core.s.rec.as_deref_mut() {
            r.tracer = Some(tr);
            r.begin_unit(core.t);
        }
        self.row = Row {
            at: p.at,
            pc: p.ret,
            regs: p.regs,
            kind: p.kind,
            ..Row::default()
        };
        self.ix.clear();
        self.ix.descend(core, p, p.ret);
    }

    /// Close the open frame with `halt` and the registers `end` (default: the machine's).
    fn close(&mut self, core: &mut Core, halt: u8, end: Option<[u8; 5]>, now: i64) {
        let c = &core.cpu;
        let [a, x, y, sp, p] = end.unwrap_or([c.a, c.x, c.y, c.sp, c.p()]);
        let rs = &mut self.chunk.records;
        let pushed = rs.push(core, [PH_CALL, halt, a, x, y, sp, p]);
        let (insns, cycles) = core
            .s
            .rec
            .as_deref_mut()
            .map_or((0, 0), |r| r.end_unit(now));
        if pushed {
            self.row.insns = insns.min(u32::MAX as u64) as u32;
            self.row.cycles = cycles.clamp(0, u32::MAX as i64) as u32;
            self.chunk.entries.push(self.row);
            self.ix.drain((rs.meta.len() - 1) as u32, &mut rs.events);
            self.closed += 1;
        }
        self.ix.clear();
    }

    /// The run is over with a frame open: close it as halted (a jam, else out of cycles) at
    /// the cycle of the decode that hit a limit, else where the run stopped.
    pub fn cut(&mut self, core: &mut Core) {
        if self.ix.open() {
            let halt = if core.jammed { HALT_JAM } else { HALT_CYCLES };
            let now = match core.s.rec.as_deref() {
                Some(r) if r.cause != STOP_NONE => r.cause_t,
                _ => core.t,
            };
            self.close(core, halt, None, now);
        }
    }

    /// Drop the open frame, as a run's end cuts it.
    pub fn discard(&mut self, core: &mut Core) {
        if !self.ix.open() {
            return;
        }
        if let Some(r) = core.s.rec.as_deref_mut() {
            r.end_unit(core.t);
            if let Some(tr) = r.tracer.as_deref_mut() {
                tr.phase = PH_NONE;
                tr.forget();
            }
        }
        self.ix.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_context_counts_only_its_own_instructions() {
        let mut ix = Intr::new(0, 0xCFF0);
        ix.push(100, 0x1000, 0xFC);
        assert_eq!(ix.count(107), 7);
        ix.push(107, 0x2000, 0xF9);
        assert_eq!(ix.count(107), 0);
        assert_eq!(ix.count(115), 8);
        ix.push(115, 0x3000, 0xF6);
        assert_eq!(ix.unwind(120), (0x3000, Some(0xF9)));
        assert_eq!(ix.count(125), 8 + 5);
        assert_eq!(ix.unwind(130), (0x2000, Some(0xFC)));
        assert_eq!(ix.count(131), 7 + 1);
        assert_eq!(ix.unwind(131), (0x1000, None));
        assert!(!ix.open());
    }

    #[test]
    fn contexts_below_the_floor_are_not_nested() {
        let mut ix = Intr::new(1, 0xCFF0);
        ix.start(0, 0, 0xFD);
        assert!(ix.open() && !ix.nested());
        ix.push(4, 0x1234, 0xFA);
        assert!(ix.nested());
    }

    #[test]
    fn events_and_rows_pack_little_endian() {
        let e = Event {
            record: 1,
            icount: 2,
            pos: 3,
            at: 0xC0DE,
            regs: [4, 5, 6, 7, 8],
            depth: 9,
            kind: 3,
        };
        assert_eq!(
            e.bytes(),
            [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0xDE, 0xC0, 4, 5, 6, 7, 8, 9, 3]
        );
        let r = Row {
            at: 0xC0DE,
            pc: 0x1234,
            regs: [1, 2, 3, 4, 5],
            kind: 1,
            insns: 0x0102_0304,
            cycles: 7,
        };
        assert_eq!(
            r.bytes(),
            [0xDE, 0xC0, 0x34, 0x12, 1, 2, 3, 4, 5, 1, 4, 3, 2, 1, 7, 0, 0, 0]
        );
        let mut into = vec![];
        let mut ix = Intr::new(0, 0);
        ix.events.push(e);
        ix.drain(5, &mut into);
        assert_eq!((into[0].record, ix.events.len()), (5, 0));
    }
}
