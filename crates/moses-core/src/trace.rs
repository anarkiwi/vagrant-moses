//! The recorder's trace: per-instruction lift facts of the call phase and per-phase records
//! (the SID window's accesses in order and the RAM change since the previous record).
//!
//! Fed at the recorder's decodes and at bus accesses to the SID window, which it answers
//! itself: a register reads the last value written there, else the armed image's byte. Its
//! state is fixed-size bitsets and tables except the edge set, which grows only by new edges.

use crate::smc::{peek, view, V_IO};
use crate::sys::{Sys, RAM};
use std::collections::HashSet;

pub const PH_NONE: u8 = 0;
pub const PH_INIT: u8 = 1;
pub const PH_CALL: u8 = 2;

/// Opcode table entries (`u16`): address form in bits 0-3 ([`A_ZP`]..[`A_IND`]), a data read
/// of the form's address ([`T_READ`]), a store to it ([`T_WRITE`]), the register a plain store
/// stores in bits 6-8 ([`S_A`]..[`S_AX`]) and the bytes pushed in bits 9-10.
pub const A_ZP: u16 = 1;
pub const A_ZPX: u16 = 2;
pub const A_ZPY: u16 = 3;
pub const A_ABS: u16 = 4;
pub const A_ABSX: u16 = 5;
pub const A_ABSY: u16 = 6;
pub const A_INDX: u16 = 7;
pub const A_INDY: u16 = 8;
pub const A_IND: u16 = 9;
pub const T_READ: u16 = 1 << 4;
pub const T_WRITE: u16 = 1 << 5;
pub const S_A: u16 = 1;
pub const S_X: u16 = 2;
pub const S_Y: u16 = 3;
pub const S_AX: u16 = 4;

const SID_MIRROR: std::ops::RangeInclusive<u16> = 0xD400..=0xD7FF;
/// Height bits per pc: `sp - sp0` in `-255..=255`, indexed by its low 9 bits.
const HWORDS: usize = 8;
/// Value bits per address.
const VWORDS: usize = 4;

/// An instruction's stores, resolved at its decode and valued at the next.
#[derive(Clone, Copy)]
struct Stores {
    bits: i32,
    n: usize,
    addrs: [u16; 3],
    stored: Option<u8>,
}

/// One phase's record: the SID log and the RAM bytes that changed since the previous one.
pub struct Record {
    pub log: Vec<(u16, u8, bool)>,
    pub idx: Vec<u16>,
    pub vals: Vec<u8>,
}

/// Lift facts: starts, `(pc << 16 | next)` edges, stored values, page-1 data ranges, stack
/// heights and bank bits per pc, all ascending.
pub struct Facts {
    pub starts: Vec<u16>,
    pub edges: Vec<u32>,
    pub writes: Vec<(u16, Vec<u8>)>,
    pub page1: Vec<(u16, u16, u16)>,
    pub heights: Vec<(u16, Vec<i16>)>,
    pub banks: Vec<(u16, Vec<u8>)>,
}

fn set(bits: &mut [u64], i: usize) {
    bits[i >> 6] |= 1 << (i & 63);
}

fn ones(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(w, &v)| {
        (0..64)
            .filter(move |b| v >> b & 1 != 0)
            .map(move |b| w * 64 + b)
    })
}

/// The bytes the instruction at `pc` fetches, as the CPU sees them.
fn fetch(s: &Sys, pc: u16) -> [u8; 3] {
    [0u16, 1, 2].map(|k| peek(s, pc.wrapping_add(k)).0)
}

/// The data address and pointer bytes of the instruction at `pc` (bytes `fetched`) by the
/// form in `e`'s bits 0-3, read from its own bytes and RAM as it is decoded.
fn operands(
    s: &Sys,
    pc: u16,
    fetched: [u8; 3],
    x: u8,
    y: u8,
    e: u16,
) -> (Option<u16>, Option<[u16; 2]>) {
    let mem = |addr: u16| {
        let k = addr.wrapping_sub(pc) as usize;
        if k < 3 {
            fetched[k]
        } else {
            s.img[RAM][addr as usize]
        }
    };
    let (b1, w) = (fetched[1], fetched[1] as u16 | (fetched[2] as u16) << 8);
    let word = |p: u8| mem(p as u16) as u16 | (mem(p.wrapping_add(1) as u16) as u16) << 8;
    let zp = |p: u8| Some([p as u16, p.wrapping_add(1) as u16]);
    match e & 0xF {
        A_ZP => (Some(b1 as u16), None),
        A_ZPX => (Some(b1.wrapping_add(x) as u16), None),
        A_ZPY => (Some(b1.wrapping_add(y) as u16), None),
        A_ABS => (Some(w), None),
        A_ABSX => (Some(w.wrapping_add(x as u16)), None),
        A_ABSY => (Some(w.wrapping_add(y as u16)), None),
        A_INDX => (Some(word(b1.wrapping_add(x))), zp(b1.wrapping_add(x))),
        A_INDY => (Some(word(b1).wrapping_add(y as u16)), zp(b1)),
        A_IND => (
            None,
            Some([w, w & 0xFF00 | (w as u8).wrapping_add(1) as u16]),
        ),
        _ => (None, None),
    }
}

/// The main context of an interrupt-driven phase, from its first decode: the port bits and the
/// RAM `$0314` and `$FFFE` vectors then, the main context's instructions (not the sentinel)
/// until the first that stores outside page 1 or touches I/O, and that one's frame.
pub struct Scan {
    table: [u16; 256],
    sentinel: u16,
    frame_cycles: i64,
    t0: i64,
    pub vector: Option<(u8, u16, u16)>,
    pub insns: u64,
    pub busy: Option<i64>,
}

impl Scan {
    /// A scan by `table` ([`T_READ`]: reads the form's address, [`T_WRITE`]: a CPU store).
    pub fn new(table: [u16; 256], sentinel: u16) -> Scan {
        Scan {
            table,
            sentinel,
            frame_cycles: 0,
            t0: -1,
            vector: None,
            insns: 0,
            busy: None,
        }
    }

    /// Start at the next decode, frames of `frame_cycles` from it.
    pub fn start(&mut self, frame_cycles: i64) {
        self.frame_cycles = frame_cycles;
    }

    /// The decode at `pc` at cycle `t`, `main` when no interrupt context is open.
    pub fn decode(&mut self, s: &Sys, pc: u16, x: u8, y: u8, main: bool, t: i64) {
        if self.frame_cycles == 0 || self.busy.is_some() {
            return;
        }
        if self.t0 < 0 {
            let ram = &s.img[RAM];
            let word = |a: usize| ram[a] as u16 | (ram[a + 1] as u16) << 8;
            (self.t0, self.vector) = (t, Some((s.bits as u8, word(0x314), word(0xFFFE))));
        }
        if !main || pc == self.sentinel {
            return;
        }
        self.insns += 1;
        let fetched = fetch(s, pc);
        let e = self.table[fetched[0] as usize];
        let (ea, ptrs) = operands(s, pc, fetched, x, y, e);
        let io = |a: u16| view(s.bits, a) == V_IO;
        let store = e & T_WRITE != 0;
        let data = ea.filter(|_| store || e & T_READ != 0);
        if store && ea.is_some_and(|a| a >> 8 != 1)
            || data.is_some_and(io)
            || ptrs.is_some_and(|p| p.into_iter().any(io))
        {
            self.busy = Some((t - self.t0) / self.frame_cycles);
        }
    }
}

pub struct Tracer {
    table: [u16; 256],
    sentinel: u16,
    lo: u16,
    hi: u32,
    regs: Box<[u8; 0x10000]>,
    base: Box<[u8; 0x10000]>,
    pub phase: u8,
    pub sp0: u8,
    starts: Box<[u64]>,
    banks: Box<[u8]>,
    heights: Box<[u64]>,
    succ: Box<[[u32; 2]]>,
    edges: HashSet<u32>,
    last: Option<u16>,
    writes: Box<[u64]>,
    page1: Box<[(u16, u16)]>,
    pend: Option<Stores>,
    log: Vec<(u16, u8, bool)>,
}

impl Tracer {
    /// A trace whose SID window is the 32-byte devices at `lo..hi` step `$20` (with the
    /// `$D400` device answering its `$D400-$D7FF` mirrors), registers from `image`; calls
    /// return to `sentinel`.
    pub fn new(
        table: [u16; 256],
        sentinel: u16,
        lo: u16,
        hi: u32,
        image: &[u8; 0x10000],
    ) -> Tracer {
        Tracer {
            table,
            sentinel,
            lo,
            hi,
            regs: Box::new(*image),
            base: Box::new(*image),
            phase: PH_NONE,
            sp0: 0,
            starts: vec![0; 0x10000 / 64].into_boxed_slice(),
            banks: vec![0; 0x10000].into_boxed_slice(),
            heights: vec![0; 0x10000 * HWORDS].into_boxed_slice(),
            succ: vec![[0; 2]; 0x10000].into_boxed_slice(),
            edges: HashSet::new(),
            last: None,
            writes: vec![0; 0x10000 * VWORDS].into_boxed_slice(),
            page1: vec![(u16::MAX, 0); 0x10000].into_boxed_slice(),
            pend: None,
            log: Vec::new(),
        }
    }

    fn device(&self, base: u16) -> bool {
        base >= self.lo && (base as u32) < self.hi && (base - self.lo) & 0x1F == 0
    }

    /// The register `addr` answers as, when a SID device of the window holds it.
    fn reg(&self, addr: u16) -> Option<u16> {
        let base = addr & 0xFFE0;
        if self.device(base) {
            Some(addr)
        } else if SID_MIRROR.contains(&addr) && self.device(*SID_MIRROR.start()) {
            Some(SID_MIRROR.start() | addr & 0x1F)
        } else {
            None
        }
    }

    fn access(&mut self, addr: u16, value: u8, write: bool) {
        if self.phase != PH_NONE {
            self.log.push((addr, value, write));
            if write && self.phase == PH_CALL {
                self.stored(addr, value);
            }
        }
    }

    /// A CPU read of the I/O byte `addr`, when the window answers it.
    pub fn sid_read(&mut self, addr: u16) -> Option<u8> {
        let r = self.reg(addr)?;
        let v = self.regs[r as usize];
        self.access(r, v, false);
        Some(v)
    }

    /// A write of the I/O byte `addr`; true when the window took it.
    pub fn sid_write(&mut self, addr: u16, value: u8) -> bool {
        let Some(r) = self.reg(addr) else {
            return false;
        };
        self.regs[r as usize] = value;
        self.access(r, value, true);
        true
    }

    fn stored(&mut self, addr: u16, value: u8) {
        set(
            &mut self.writes,
            addr as usize * VWORDS * 64 + value as usize,
        );
    }

    fn page1(&mut self, pc: u16, addr: u16) {
        if addr >> 8 == 1 {
            let r = &mut self.page1[pc as usize];
            *r = (r.0.min(addr), r.1.max(addr));
        }
    }

    fn edge(&mut self, pc: u16, next: u16) {
        let s = &mut self.succ[pc as usize];
        let n = next as u32 + 1;
        if s[0] == n || s[1] == n {
            return;
        }
        *s = [n, s[0]];
        self.edges.insert((pc as u32) << 16 | next as u32);
    }

    /// Value the previous instruction's stores from RAM after it: a byte the CPU does not see
    /// as I/O, else (outside the SID window) the register a plain store stored.
    fn settle(&mut self, s: &Sys) {
        let Some(st) = self.pend.take() else {
            return;
        };
        for &addr in &st.addrs[..st.n] {
            if addr < 2 || view(st.bits, addr) != V_IO {
                self.stored(addr, s.img[RAM][addr as usize]);
            } else if !(addr >= self.lo && (addr as u32) < self.hi) {
                if let Some(v) = st.stored {
                    self.stored(addr, v);
                }
            }
        }
    }

    /// The decode of the instruction at `pc` with registers `a`, `x`, `y`, `sp`; a call's
    /// instruction is traced unless it is the sentinel or the call `timed_out`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(&mut self, s: &Sys, pc: u16, a: u8, x: u8, y: u8, sp: u8, timed_out: bool) {
        self.settle(s);
        if self.phase != PH_CALL || pc == self.sentinel || timed_out {
            return;
        }
        let h = (sp as i32 - self.sp0 as i32) as usize & 0x1FF;
        set(&mut self.starts, pc as usize);
        set(&mut self.heights, pc as usize * HWORDS * 64 + h);
        self.banks[pc as usize] |= 1 << s.bits;
        if let Some(p) = self.last.replace(pc) {
            self.edge(p, pc);
        }
        let fetched = fetch(s, pc);
        let e = self.table[fetched[0] as usize];
        if e == 0 {
            return;
        }
        let (ea, ptrs) = operands(s, pc, fetched, x, y, e);
        for p in ptrs.into_iter().flatten() {
            self.page1(pc, p);
        }
        if let (true, Some(t)) = (e & T_READ != 0, ea) {
            self.page1(pc, t);
        }
        let pushes = (e >> 9 & 3) as usize;
        let mut st = Stores {
            bits: s.bits,
            n: pushes,
            addrs: [0; 3],
            stored: match e >> 6 & 7 {
                S_A => Some(a),
                S_X => Some(x),
                S_Y => Some(y),
                S_AX => Some(a & x),
                _ => None,
            },
        };
        if pushes > 0 {
            for (k, d) in st.addrs[..pushes].iter_mut().enumerate() {
                *d = 0x100 | sp.wrapping_sub(k as u8) as u16;
            }
        } else if let (true, Some(t)) = (e & T_WRITE != 0, ea) {
            self.page1(pc, t);
            (st.n, st.addrs[0]) = (1, t);
        }
        if st.n > 0 {
            self.pend = Some(st);
        }
    }

    /// Value the last instruction's stores once the run is over.
    pub fn finish(&mut self, s: &Sys) {
        self.settle(s);
    }

    /// Close the current phase: its record, edges restarting at the next call.
    pub fn snap(&mut self, s: &Sys) -> Record {
        self.last = None;
        let (ram, base) = (&s.img[RAM], &mut *self.base);
        let mut idx = Vec::new();
        let (words, olds) = (ram.as_chunks::<8>().0, base.as_chunks::<8>().0);
        for (c, (r, b)) in words.iter().zip(olds).enumerate() {
            if r != b {
                idx.extend((0..8).filter(|&k| r[k] != b[k]).map(|k| (c * 8 + k) as u16));
            }
        }
        base.copy_from_slice(&ram[..]);
        let vals = idx.iter().map(|&i| ram[i as usize]).collect();
        Record {
            log: std::mem::take(&mut self.log),
            idx,
            vals,
        }
    }

    pub fn facts(&self) -> Facts {
        let starts: Vec<u16> = ones(&self.starts).map(|p| p as u16).collect();
        let mut edges: Vec<u32> = self.edges.iter().copied().collect();
        edges.sort_unstable();
        let per = |p: u16| p as usize;
        let heights = starts
            .iter()
            .map(|&p| {
                let row = &self.heights[per(p) * HWORDS..][..HWORDS];
                (p, ones(row).map(|h| (h as i16) << 7 >> 7).collect())
            })
            .collect();
        let banks = starts
            .iter()
            .map(|&p| {
                let b = self.banks[per(p)];
                (p, (0..8).filter(|k| b >> k & 1 != 0).collect())
            })
            .collect();
        let writes = (0..0x10000)
            .filter_map(|a| {
                let row = &self.writes[a * VWORDS..][..VWORDS];
                row.iter()
                    .any(|&w| w != 0)
                    .then(|| (a as u16, ones(row).map(|v| v as u8).collect()))
            })
            .collect();
        let page1 = (0..0x10000)
            .filter_map(|p| {
                let (lo, hi) = self.page1[p];
                (lo <= hi).then_some((p as u16, lo, hi))
            })
            .collect();
        Facts {
            starts,
            edges,
            writes,
            page1,
            heights,
            banks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracer(lo: u16, hi: u32) -> Tracer {
        let mut image = Box::new([0u8; 0x10000]);
        image[0xD418] = 0x0F;
        Tracer::new([0; 256], 0xCFF0, lo, hi, &image)
    }

    #[test]
    fn the_window_answers_its_devices_and_the_d400_mirrors() {
        let mut t = tracer(0xD400, 0xD420);
        t.phase = PH_CALL;
        assert_eq!(t.sid_read(0xD418), Some(0x0F));
        assert!(t.sid_write(0xD5F8, 0x1F));
        assert_eq!(t.sid_read(0xD418), Some(0x1F));
        assert_eq!(t.sid_read(0xDE00), None);
        assert!(!t.sid_write(0xDE00, 1));
        assert_eq!(
            t.log,
            [
                (0xD418, 0x0F, false),
                (0xD418, 0x1F, true),
                (0xD418, 0x1F, false)
            ]
        );
        assert_eq!(t.facts().writes, [(0xD418, vec![0x1F])]);
        let unaligned = tracer(0xD401, 0xD800);
        assert_eq!(unaligned.reg(0xD418), None);
    }

    #[test]
    fn only_traced_phases_log() {
        let mut t = tracer(0xD400, 0xD800);
        t.sid_write(0xD400, 1);
        t.phase = PH_INIT;
        t.sid_write(0xD401, 2);
        assert_eq!(t.log, [(0xD401, 2, true)]);
        assert!(t.facts().writes.is_empty());
    }

    #[test]
    fn edges_dedupe_through_the_successor_cache() {
        let mut t = tracer(0xD400, 0xD800);
        for n in [0x1002, 0x1010, 0x1002, 0x1020, 0x1010] {
            t.edge(0x1000, n);
        }
        let mut e: Vec<_> = t.edges.iter().copied().collect();
        e.sort_unstable();
        assert_eq!(e, [0x1000_1002, 0x1000_1010, 0x1000_1020]);
    }

    #[test]
    fn heights_read_back_signed_in_unsigned_order() {
        let mut t = tracer(0xD400, 0xD800);
        set(&mut t.starts, 0x1000);
        for h in [-1i32, 2, -255, 0, 255] {
            set(&mut t.heights, 0x1000 * HWORDS * 64 + (h as usize & 0x1FF));
        }
        t.banks[0x1000] = 0b1000_0010;
        t.page1(0x1000, 0x01F0);
        t.page1(0x1000, 0x0105);
        t.page1(0x1000, 0x0200);
        let f = t.facts();
        assert_eq!(f.starts, [0x1000]);
        assert_eq!(f.heights, [(0x1000, vec![0, 2, 255, -255, -1])]);
        assert_eq!(f.banks, [(0x1000, vec![1, 7])]);
        assert_eq!(f.page1, [(0x1000, 0x0105, 0x01F0)]);
    }
}
