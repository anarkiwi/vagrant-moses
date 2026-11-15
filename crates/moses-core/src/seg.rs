//! The segment profiler: the CPU's work cut into segments that no chip or host can
//! intervene in, each keyed by its entry state and the bytes it reads, counted per key
//! (`docs/native.md`, Segment profile).
//!
//! A segment opens at a decode and closes at a later decode. The machine feeds it every
//! CPU read and write (a running hash each), the decodes, and *cuts*: an I/O access, a stall
//! cycle or a scheduler event that changed the interrupt lines, an interrupt or BRK entry.
//! A cut excludes the instruction it happened in: the segment closes at the previous decode
//! (the mark), so a recorded segment never contains one.

use std::collections::hash_map::{Entry, HashMap};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;

use jennings_core::{AddrSet, Cpu};

use crate::machine::Core;
use crate::sys::Host;

/// Why a segment ended (`ENDS`).
pub const END_CAP: usize = 0;
pub const END_PC: usize = 1;
pub const END_HOST: usize = 2;
pub const END_RUN: usize = 3;
pub const END_IO: usize = 4;
pub const END_EVENT: usize = 5;
pub const END_IRQ: usize = 6;
pub const ENDS: [&str; 7] = ["cap", "pc", "host", "run", "io", "event", "irq"];
/// The ends that stopped a segment because chip state was about to reach the CPU.
pub const EARLY: [usize; 3] = [END_IO, END_EVENT, END_IRQ];
/// Length buckets: bucket `b` holds segments of `2^b..2^(b+1)` instructions.
pub const HIST: usize = 32;
pub const MAX_CAP: u32 = 1 << 20;
/// Direct-mapped write-back slots in front of a machine's table.
const SLOTS: usize = 256;
const K: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED: u64 = 0x243F_6A88_85A3_08D3;
const JSR: u8 = 0x20;
const RTS: u8 = 0x60;
const RTI: u8 = 0x40;

/// One step of a stream hash: a bijection of the state for each input, so streams that
/// differ in one element never collide.
#[inline(always)]
fn step(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(K)
}

#[inline(always)]
fn mix(h: u64, v: u64) -> u64 {
    let h = (h ^ v).wrapping_mul(K);
    h ^ h >> 32
}

/// The host a profiled run goes through: `H` with [`Host::SEG`] set, so the run loop's
/// profiler feeds compile in only while a profiler is armed.
pub struct Profiled<'a, H: Host>(pub &'a mut H);

impl<H: Host> Host for Profiled<'_, H> {
    const SEG: bool = true;
    fn sid_read(&mut self, dev: usize, cycle: i64, reg: u8) -> u8 {
        self.0.sid_read(dev, cycle, reg)
    }
    fn sid_write(&mut self, dev: usize, cycle: i64, reg: u8, value: u8) {
        self.0.sid_write(dev, cycle, reg, value)
    }
    fn event(&mut self, cycle: i64, name: u8) {
        self.0.event(cycle, name)
    }
    fn trap(&mut self, core: &mut Core) -> bool {
        self.0.trap(core)
    }
    fn decode(&mut self, core: &mut Core) {
        self.0.decode(core)
    }
    fn port_write(&mut self, core: &mut Core) {
        self.0.port_write(core)
    }
    fn trap_writes(&self) -> bool {
        self.0.trap_writes()
    }
    fn io(&mut self, addr: u16, value: u8, write: bool) {
        self.0.io(addr, value, write)
    }
    fn failed(&self) -> bool {
        self.0.failed()
    }
}

/// `pc`, `a`, `x`, `y`, `sp` and `p` packed in bits 0-55.
#[inline(always)]
pub fn regs(c: &Cpu) -> u64 {
    c.pc as u64
        | (c.a as u64) << 16
        | (c.x as u64) << 24
        | (c.y as u64) << 32
        | (c.sp as u64) << 40
        | (c.p() as u64) << 48
}

/// A segment's identity: the entry registers, bank bits and interrupt-line bits
/// ([`regs`] with the bits at 56-58 and the lines at 59-62), and the hash and count of the
/// `(address, value)` reads it made, fetches included.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Key {
    pub entry: u64,
    pub reads: u64,
    pub n: u32,
}

impl Key {
    pub fn pc(&self) -> u16 {
        self.entry as u16
    }

    #[inline(always)]
    fn digest(&self) -> u64 {
        mix(self.reads ^ self.entry.rotate_left(29), self.n as u64)
    }
}

impl Hash for Key {
    fn hash<S: Hasher>(&self, s: &mut S) {
        s.write_u64(self.digest());
    }
}

/// The [`Key`] hash as it is: it is mixed already.
#[derive(Default)]
pub struct Pass(u64);

impl Hasher for Pass {
    fn write(&mut self, bytes: &[u8]) {
        self.0 = bytes.iter().fold(self.0, |h, &b| mix(h, b as u64));
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type Table = HashMap<Key, Ent, BuildHasherDefault<Pass>>;

/// A key's occurrences, its cycles and instructions, the hash of its effect (write stream,
/// exit registers, cycles) and the first source it was seen in (`batch << 32 | fork`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ent {
    pub n: u64,
    pub cycles: u32,
    pub insns: u32,
    pub effect: u64,
    pub first: u64,
}

/// Additive counts. `saved` is the cycles of the occurrences after a key's first in its
/// machine (`[0]`) and in its batch (`[1]`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub run: u64,
    pub segs: [u64; 7],
    pub cycles: [u64; 7],
    pub insns: u64,
    pub hist: [[u64; 2]; HIST],
    pub conflicts: u64,
    pub saved: [u64; 2],
    pub forks: u64,
}

impl Totals {
    fn add(&mut self, o: &Totals) {
        self.run += o.run;
        for i in 0..ENDS.len() {
            self.segs[i] += o.segs[i];
            self.cycles[i] += o.cycles[i];
        }
        for (h, x) in self.hist.iter_mut().zip(&o.hist) {
            (h[0], h[1]) = (h[0] + x[0], h[1] + x[1]);
        }
        self.insns += o.insns;
        self.conflicts += o.conflicts;
        self.saved[0] += o.saved[0];
        self.saved[1] += o.saved[1];
        self.forks += o.forks;
    }
}

fn saved(t: &Table) -> u64 {
    t.values().map(|e| (e.n - 1) * e.cycles as u64).sum()
}

/// Add `e`'s count to `k`'s in `into`; a differing effect counts a conflict.
fn add(into: &mut Table, k: Key, e: Ent, conflicts: &mut u64) {
    match into.entry(k) {
        Entry::Occupied(mut o) => {
            let o = o.get_mut();
            o.n += e.n;
            o.first = o.first.min(e.first);
            *conflicts += (o.effect != e.effect) as u64;
        }
        Entry::Vacant(v) => {
            v.insert(e);
        }
    }
}

fn merge(into: &mut Table, from: Table, conflicts: &mut u64) {
    for (k, e) in from {
        add(into, k, e, conflicts);
    }
}

/// The state at a decode inside the open segment.
#[derive(Clone, Copy, Default)]
struct Mark {
    t: i64,
    reads: u64,
    n: u32,
    writes: u64,
    insns: u32,
    exit: u64,
}

/// The profiler one machine runs with.
pub struct Segs {
    cap: u32,
    pcs: Option<Arc<AddrSet>>,
    calls: bool,
    /// The source this machine's segments are first seen in: `batch << 32 | fork`.
    pub src: u64,
    pub open: bool,
    entry: u64,
    t0: i64,
    reads: u64,
    n: u32,
    writes: u64,
    cut: u8,
    /// The opcode of the instruction decoded last.
    pub op: u8,
    mark: Mark,
    /// Recent keys, counted here and added to `table` when evicted (`n` 0: empty).
    slots: Box<[(Key, Ent)]>,
    table: Table,
    pub tot: Totals,
    global: Table,
    pub batches: u32,
}

/// The forks of one batch, folded.
#[derive(Default)]
pub struct Batch {
    pub tot: Totals,
    pub table: Table,
}

impl Batch {
    /// Fold in the segments of one fork.
    pub fn fold(&mut self, mut f: Segs) {
        f.flush();
        let mut tot = f.tot;
        tot.saved[0] += saved(&f.table);
        tot.forks += 1;
        self.tot.add(&tot);
        merge(&mut self.table, f.table, &mut self.tot.conflicts);
    }

    /// Fold in another part of the same batch.
    pub fn join(&mut self, o: Batch) {
        self.tot.add(&o.tot);
        merge(&mut self.table, o.table, &mut self.tot.conflicts);
    }
}

/// One entry pc's keys: how many, their occurrences, cycles and saved cycles summed, and
/// the first source any was seen in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Top {
    pub pc: u16,
    pub keys: u64,
    pub n: u64,
    pub cycles: u64,
    pub saved: u64,
    pub first: u64,
}

/// Totals with the machine's own runs as one more source and batch; `global` the cycles
/// of the occurrences after each key's first anywhere, `recurrent` the cycles of every
/// occurrence of a key seen more than once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub tot: Totals,
    pub distinct: u64,
    pub global: u64,
    pub recurrent: u64,
    pub batches: u32,
    pub top: Vec<Top>,
}

impl Segs {
    /// Segments of at most `cap` instructions (1 to [`MAX_CAP`]) that also end after an
    /// instruction when the next pc is in `pcs`, or with `calls`, after a `JSR`, `RTS` or `RTI`.
    pub fn new(cap: u32, pcs: Option<Arc<AddrSet>>, calls: bool) -> Segs {
        Segs {
            cap: cap.clamp(1, MAX_CAP),
            pcs,
            calls,
            src: 0,
            open: false,
            entry: 0,
            t0: 0,
            reads: SEED,
            n: 0,
            writes: SEED,
            cut: 0,
            op: 0,
            mark: Mark::default(),
            slots: vec![(Key::default(), Ent::default()); SLOTS].into_boxed_slice(),
            table: Table::default(),
            tot: Totals::default(),
            global: Table::default(),
            batches: 0,
        }
    }

    /// A profiler for a fork: the same boundaries, nothing recorded.
    pub fn spawn(&self) -> Segs {
        Segs::new(self.cap, self.pcs.clone(), self.calls)
    }

    /// The batch index the next batch of forks records as.
    pub fn next_batch(&self) -> u64 {
        self.batches as u64 + 1
    }

    #[inline(always)]
    pub fn read(&mut self, addr: u16, v: u8) {
        self.reads = step(self.reads, (addr as u64) << 8 | v as u64);
        self.n = self.n.wrapping_add(1);
    }

    #[inline(always)]
    pub fn write(&mut self, addr: u16, v: u8) {
        self.writes = step(self.writes, (addr as u64) << 8 | v as u64);
    }

    /// The instruction running now is not replayable for reason `r`.
    #[inline(always)]
    pub fn cut(&mut self, r: usize) {
        if self.cut == 0 {
            self.cut = r as u8;
        }
    }

    fn advance(&mut self, exit: u64, t: i64) {
        self.mark = Mark {
            t,
            reads: self.reads,
            n: self.n,
            writes: self.writes,
            insns: self.mark.insns + 1,
            exit,
        };
    }

    /// The decode at `pc` (registers `exit`) at cycle `t`; `host`: a hook that may change
    /// the machine runs at it.
    pub fn decode(&mut self, exit: u64, pc: u16, t: i64, host: bool) {
        if !self.open {
            return;
        }
        if self.cut != 0 {
            return self.emit(self.cut as usize);
        }
        self.advance(exit, t);
        let end = if host {
            END_HOST
        } else if self.mark.insns >= self.cap {
            END_CAP
        } else if self.pcs.as_ref().is_some_and(|s| s.contains(pc))
            || self.calls && matches!(self.op, JSR | RTS | RTI)
        {
            END_PC
        } else {
            return;
        };
        self.emit(end);
    }

    /// An interrupt sequence starts after the instruction that just ended.
    pub fn interrupt(&mut self, exit: u64, t: i64) {
        if !self.open {
            return;
        }
        if self.cut == 0 {
            self.advance(exit, t);
        }
        self.emit(END_IRQ);
    }

    /// Open a segment at entry state `entry` at cycle `t`.
    pub fn open(&mut self, entry: u64, t: i64) {
        self.open = true;
        (self.entry, self.t0, self.cut) = (entry, t, 0);
        (self.reads, self.n, self.writes) = (SEED, 0, SEED);
        self.mark = Mark {
            t,
            reads: SEED,
            n: 0,
            writes: SEED,
            insns: 0,
            exit: entry,
        };
    }

    /// The run is over after `cycles`: an open segment closes at its mark.
    pub fn end_run(&mut self, cycles: i64) {
        self.tot.run += cycles.max(0) as u64;
        if self.open {
            self.emit(END_RUN);
        }
    }

    fn emit(&mut self, end: usize) {
        self.open = false;
        let m = self.mark;
        if m.insns == 0 {
            return;
        }
        let cycles = (m.t - self.t0) as u64;
        let tot = &mut self.tot;
        tot.segs[end] += 1;
        tot.cycles[end] += cycles;
        tot.insns += m.insns as u64;
        let h = &mut tot.hist[31 - m.insns.leading_zeros() as usize];
        (h[0], h[1]) = (h[0] + 1, h[1] + cycles);
        let key = Key {
            entry: self.entry,
            reads: m.reads,
            n: m.n,
        };
        let effect = mix(mix(m.writes, m.exit), cycles);
        let slot = &mut self.slots[key.digest() as usize % SLOTS];
        if slot.1.n != 0 && slot.0 == key {
            slot.1.n += 1;
            tot.conflicts += (slot.1.effect != effect) as u64;
            return;
        }
        let ent = Ent {
            n: 1,
            cycles: cycles as u32,
            insns: m.insns,
            effect,
            first: self.src,
        };
        let (k, e) = std::mem::replace(slot, (key, ent));
        if e.n != 0 {
            add(&mut self.table, k, e, &mut tot.conflicts);
        }
    }

    /// Add the slots' counts to the table.
    fn flush(&mut self) {
        for (k, e) in self.slots.iter_mut() {
            if e.n != 0 {
                add(
                    &mut self.table,
                    *k,
                    std::mem::take(e),
                    &mut self.tot.conflicts,
                );
            }
        }
    }

    /// Every key this machine recorded in its own runs, with its counts.
    pub fn table(&mut self) -> &Table {
        self.flush();
        &self.table
    }

    /// Take in a finished batch of forks.
    pub fn absorb(&mut self, b: Batch) {
        self.batches += 1;
        let mut tot = b.tot;
        tot.saved[1] += saved(&b.table);
        self.tot.add(&tot);
        merge(&mut self.global, b.table, &mut self.tot.conflicts);
    }

    /// The profile so far, with the `top` entry pcs by saved cycles.
    pub fn summary(&mut self, top: usize) -> Summary {
        self.flush();
        let mut tot = self.tot.clone();
        let own = saved(&self.table);
        tot.saved[0] += own;
        tot.saved[1] += own;
        let (mut distinct, mut global, mut recurrent) = (0, 0, 0);
        let mut pcs: HashMap<u16, Top> = HashMap::new();
        let mut visit = |k: &Key, e: &Ent, n: u64, first: u64| {
            let c = e.cycles as u64;
            distinct += 1;
            global += (n - 1) * c;
            recurrent += if n > 1 { n * c } else { 0 };
            let t = pcs.entry(k.pc()).or_insert(Top {
                pc: k.pc(),
                first,
                ..Top::default()
            });
            t.keys += 1;
            t.n += n;
            t.cycles += n * c;
            t.saved += (n - 1) * c;
            t.first = t.first.min(first);
        };
        for (k, e) in &self.global {
            match self.table.get(k) {
                Some(l) => visit(k, e, e.n + l.n, e.first.min(l.first)),
                None => visit(k, e, e.n, e.first),
            }
        }
        for (k, e) in &self.table {
            if !self.global.contains_key(k) {
                visit(k, e, e.n, e.first);
            }
        }
        let mut top_pcs: Vec<Top> = pcs.into_values().collect();
        top_pcs.sort_unstable_by_key(|t| (std::cmp::Reverse(t.saved), t.pc));
        top_pcs.truncate(top);
        Summary {
            tot,
            distinct,
            global,
            recurrent,
            batches: self.batches,
            top: top_pcs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A segment of `insns` instructions from `t`, each 4 cycles, reading `byte` once each.
    fn segment(g: &mut Segs, t: i64, insns: u32, byte: u8, end: Option<usize>) {
        g.open(0x1000, t);
        for i in 0..insns as i64 {
            g.read(0x1000 + i as u16, byte);
            g.write(0x2000, byte);
            g.decode(0x1001 + i as u64, 0x1001 + i as u16, t + 4 * (i + 1), false);
        }
        if let Some(r) = end {
            g.cut(r);
            g.read(0xD012, 0);
            g.decode(0, 0x1100, t + 4 * insns as i64 + 4, false);
        }
    }

    #[test]
    fn the_cap_closes_a_segment_and_counts_it() {
        let mut g = Segs::new(3, None, false);
        segment(&mut g, 0, 7, 1, None);
        assert_eq!(g.tot.segs[END_CAP], 1);
        assert_eq!((g.tot.cycles[END_CAP], g.tot.insns), (12, 3));
        assert!(!g.open);
        assert_eq!(g.tot.hist[1], [1, 12]);
    }

    #[test]
    fn a_cut_drops_the_instruction_it_happened_in() {
        let mut g = Segs::new(100, None, false);
        segment(&mut g, 0, 5, 1, Some(END_IO));
        assert_eq!((g.tot.segs[END_IO], g.tot.cycles[END_IO]), (1, 20));
        g.open(0x1000, 50);
        g.cut(END_EVENT);
        g.decode(0, 0x1003, 54, false);
        assert_eq!(g.tot.segs.iter().sum::<u64>(), 1);
    }

    #[test]
    fn one_read_byte_changes_the_key_and_equal_runs_share_it() {
        let mut g = Segs::new(100, None, false);
        for (t, byte) in [(0, 1), (100, 1), (200, 2)] {
            segment(&mut g, t, 4, byte, Some(END_IO));
        }
        let mut n: Vec<u64> = g.table().values().map(|e| e.n).collect();
        n.sort_unstable();
        assert_eq!(n, [1, 2]);
        assert_eq!(g.tot.conflicts, 0);
        let s = g.summary(5);
        assert_eq!((s.distinct, s.global, s.recurrent), (2, 16, 32));
        assert_eq!(s.tot.saved, [16, 16]);
        assert_eq!(s.top[0].pc, 0x1000);
    }

    #[test]
    fn pcs_calls_and_hooks_end_a_segment_after_an_instruction() {
        let mut set = AddrSet::default();
        set.insert(0x1002);
        let mut g = Segs::new(100, Some(Arc::new(set)), true);
        g.open(1, 0);
        g.decode(2, 0x1002, 4, false);
        assert_eq!(g.tot.segs[END_PC], 1);
        g.open(1, 10);
        g.op = JSR;
        g.decode(2, 0x2000, 16, false);
        assert_eq!(g.tot.segs[END_PC], 2);
        g.open(1, 20);
        g.op = 0xEA;
        g.decode(2, 0x2000, 22, true);
        g.open(1, 30);
        g.interrupt(2, 32);
        g.open(1, 40);
        g.decode(2, 0x3000, 42, false);
        g.end_run(50);
        assert_eq!(
            (
                g.tot.segs[END_HOST],
                g.tot.segs[END_IRQ],
                g.tot.segs[END_RUN]
            ),
            (1, 1, 1)
        );
        assert_eq!(g.tot.run, 50);
    }

    #[test]
    fn batches_fold_in_any_order_to_the_same_profile() {
        let base = Segs::new(100, None, false);
        let forks = |order: &[usize]| {
            let mut parts = [Batch::default(), Batch::default()];
            for (j, &i) in order.iter().enumerate() {
                let mut f = base.spawn();
                f.src = 1 << 32 | i as u64;
                segment(&mut f, 0, 3, (i % 2) as u8, Some(END_EVENT));
                segment(&mut f, 100, 3, (i % 2) as u8, Some(END_EVENT));
                parts[j % 2].fold(f);
            }
            let [mut a, b] = parts;
            a.join(b);
            let mut g = base.spawn();
            segment(&mut g, 0, 3, 0, Some(END_EVENT));
            g.absorb(a);
            g.summary(3)
        };
        let s = forks(&[0, 1, 2, 3]);
        assert_eq!(s, forks(&[3, 1, 0, 2]));
        assert_eq!((s.distinct, s.tot.forks, s.batches), (2, 4, 1));
        assert_eq!(s.tot.saved, [4 * 12, 6 * 12]);
        assert_eq!(s.global, (5 + 4 - 2) * 12);
        assert_eq!(s.top[0].first, 0);
    }
}
