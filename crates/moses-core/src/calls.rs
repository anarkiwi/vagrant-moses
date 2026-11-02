//! A queue of calls run back to back from the SMC recorder's sentinel trap, with no host
//! callback per call: each returns to the sentinel, whose trap closes the call's record (when
//! the trace is armed) and starts the next.

use std::collections::BTreeSet;

use crate::frames::{Event, Intr, Pushed, K_BRK};
use crate::machine::Core;
use crate::trace::{PH_CALL, PH_NONE};

pub const HALT_RETURN: u8 = 0;
pub const HALT_CYCLES: u8 = 1;
pub const HALT_JAM: u8 = 2;

/// One record per closed phase: `meta` `[phase, halt, a, x, y, sp, p]`, the phase's logged I/O
/// accesses `(addr, value, write)` ending at `log_end`, and the RAM bytes it changed (against
/// the previous record, the first against `origin`) ending at `delta_end`.
#[derive(Default)]
pub struct Records {
    pub origin: Vec<u8>,
    pub meta: Vec<[u8; 7]>,
    pub log: Vec<(u16, u8, bool)>,
    pub log_end: Vec<u64>,
    pub idx: Vec<u16>,
    pub vals: Vec<u8>,
    pub delta_end: Vec<u64>,
    /// The interrupts taken inside the records, by record index.
    pub events: Vec<Event>,
}

/// `core[addr] = value` as the host's own write, which no instruction owns.
pub fn host_write(core: &mut Core, addr: u16, value: u8) {
    core.poke(addr, value);
    let bits = core.s.bits;
    if let Some(r) = core.s.rec.as_deref_mut() {
        r.host(addr, bits);
    }
}

impl Records {
    /// Close the open phase as a record with `meta`: its logged I/O accesses and the RAM it
    /// changed. False when no trace is armed.
    pub fn push(&mut self, core: &mut Core, meta: [u8; 7]) -> bool {
        let Some(mut tr) = core.s.rec.as_deref_mut().and_then(|r| r.tracer.take()) else {
            return false;
        };
        if self.origin.is_empty() {
            self.origin = tr.base().to_vec();
        }
        let r = tr.snap(&core.s);
        tr.phase = PH_NONE;
        if let Some(rec) = core.s.rec.as_deref_mut() {
            rec.tracer = Some(tr);
        }
        self.meta.push(meta);
        self.log.extend(r.log);
        self.log_end.push(self.log.len() as u64);
        self.idx.extend(r.idx);
        self.vals.extend(r.vals);
        self.delta_end.push(self.idx.len() as u64);
        true
    }
}

pub struct Calls {
    /// `((phase, pc, a), times)`: each entry runs `times` calls back to back.
    queue: Vec<((u8, u16, u8), u64)>,
    next: usize,
    started: u64,
    left: u64,
    ports: [Option<u8>; 2],
    sentinel: u16,
    ix: Intr,
    cur: u8,
    pub records: Records,
    /// `(pc, sp)` of each call's entry, the 2-byte return frame pushed.
    pub entry_sp: BTreeSet<(u16, u8)>,
    /// `(pc, port dir, port data)` of each call's entry.
    pub entry_port: BTreeSet<(u16, u8, u8)>,
}

impl Calls {
    /// `queue` of `((phase, pc, a), times)` (phase `PH_INIT` or `PH_CALL`), each entry run
    /// `times` calls back to back; `ports` `[init, call]` the `$01` value the host stores
    /// before such a call when it differs. The recorder's budget limits each call.
    pub fn new(
        mut queue: Vec<((u8, u16, u8), u64)>,
        ports: [Option<u8>; 2],
        sentinel: u16,
    ) -> Calls {
        queue.retain(|&(_, times)| times > 0);
        Calls {
            left: queue.iter().map(|&(_, times)| times).sum(),
            queue,
            next: 0,
            started: 0,
            ports,
            sentinel,
            ix: Intr::new(1, sentinel),
            cur: PH_NONE,
            records: Records::default(),
            entry_sp: BTreeSet::new(),
            entry_port: BTreeSet::new(),
        }
    }

    /// The calls not yet started.
    pub fn left(&self) -> u64 {
        self.left
    }

    /// Close the open phase's record with `halt`.
    pub fn snap(&mut self, core: &mut Core, halt: u8) {
        if self.cur == PH_NONE {
            return;
        }
        let c = &core.cpu;
        let meta = [self.cur, halt, c.a, c.x, c.y, c.sp, c.p()];
        self.cur = PH_NONE;
        let t = core.t;
        if let Some(r) = core.s.rec.as_deref_mut() {
            r.end_unit(t);
        }
        let rs = &mut self.records;
        if rs.push(core, meta) {
            self.ix.drain((rs.meta.len() - 1) as u32, &mut rs.events);
        }
        self.ix.clear();
    }

    /// An interrupt of `kind` was entered, the CPU at its handler: inside a call it is an
    /// event of the call's record and a context of its own.
    pub fn enter(&mut self, core: &mut Core, kind: u8) {
        if kind == K_BRK || !self.ix.open() {
            return;
        }
        let mut p = Pushed::read(core, kind);
        if self.ix.nested() && p.ret == self.sentinel {
            p.ret = self.ix.pop(core);
        }
        self.ix.nest(core, &p);
        self.entry_sp.insert((p.at, p.sp));
        self.entry_port
            .insert((p.at, core.s.port_dir, core.s.port_data));
    }

    /// The sentinel's trap: close the returning call's record and start the next call, or
    /// stop the run (false) when none is left.
    pub fn trap(&mut self, core: &mut Core) -> bool {
        if self.ix.nested() {
            core.cpu.pc = self.ix.pop(core);
            return true;
        }
        self.snap(core, HALT_RETURN);
        let Some(&((phase, pc, a), times)) = self.queue.get(self.next) else {
            core.stop = true;
            return false;
        };
        (self.started, self.left) = (self.started + 1, self.left - 1);
        if self.started == times {
            (self.next, self.started) = (self.next + 1, 0);
        }
        self.cur = phase;
        let sp0 = core.cpu.sp.wrapping_sub(2);
        if let Some(tr) = core
            .s
            .rec
            .as_deref_mut()
            .and_then(|r| r.tracer.as_deref_mut())
        {
            tr.phase = phase;
            if phase == PH_CALL {
                tr.sp0 = sp0;
            }
        }
        core.s.events.clear();
        if let Some(port) = self.ports[(phase == PH_CALL) as usize] {
            if port != core.s.port_data {
                host_write(core, 1, port);
            }
        }
        let bits = core.s.bits;
        if let Some(r) = core.s.rec.as_deref_mut() {
            if phase == PH_CALL {
                if r.call < 0 {
                    r.begin(bits);
                } else {
                    r.next_call();
                }
            }
        }
        let (sp, ret) = (core.cpu.sp, self.sentinel.wrapping_sub(1));
        host_write(core, 0x100 | sp as u16, (ret >> 8) as u8);
        host_write(core, 0x100 | sp.wrapping_sub(1) as u16, ret as u8);
        (core.cpu.sp, core.cpu.a, core.cpu.pc) = (sp0, a, pc);
        let t = core.t;
        let now = core.s.rec.as_deref_mut().map_or(0, |r| {
            r.begin_unit(t);
            r.insns
        });
        if phase == PH_CALL {
            self.ix.start(now, 0, sp0);
            self.entry_sp.insert((pc, sp0));
            self.entry_port
                .insert((pc, core.s.port_dir, core.s.port_data));
        }
        true
    }
}
