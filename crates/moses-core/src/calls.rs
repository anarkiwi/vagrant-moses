//! A queue of calls run back to back from the SMC recorder's sentinel trap, with no host
//! callback per call: each returns to the sentinel, whose trap closes the call's record (when
//! the trace is armed) and starts the next.

use std::collections::BTreeSet;

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
}

pub struct Calls {
    queue: Vec<(u8, u16, u8)>,
    next: usize,
    ports: [Option<u8>; 2],
    sentinel: u16,
    call_cycles: i64,
    cur: u8,
    pub records: Records,
    /// `(pc, sp)` of each call's entry, the 2-byte return frame pushed.
    pub entry_sp: BTreeSet<(u16, u8)>,
    /// `(pc, port dir, port data)` of each call's entry.
    pub entry_port: BTreeSet<(u16, u8, u8)>,
}

impl Calls {
    /// `queue` of `(phase, pc, a)` (phase `PH_INIT` or `PH_CALL`); `ports` `[init, call]` the
    /// `$01` value the host stores before such a call when it differs; `call_cycles` (0: none)
    /// the cycles a call may run before the recorder stops the run.
    pub fn new(
        queue: Vec<(u8, u16, u8)>,
        ports: [Option<u8>; 2],
        sentinel: u16,
        call_cycles: i64,
    ) -> Calls {
        Calls {
            queue,
            next: 0,
            ports,
            sentinel,
            call_cycles,
            cur: PH_NONE,
            records: Records::default(),
            entry_sp: BTreeSet::new(),
            entry_port: BTreeSet::new(),
        }
    }

    /// The calls not yet started.
    pub fn left(&self) -> usize {
        self.queue.len() - self.next
    }

    fn host_write(core: &mut Core, addr: u16, value: u8) {
        core.poke(addr, value);
        let bits = core.s.bits;
        if let Some(r) = core.s.rec.as_deref_mut() {
            r.host(addr, bits);
        }
    }

    /// Close the open phase's record with `halt`.
    pub fn snap(&mut self, core: &mut Core, halt: u8) {
        if self.cur == PH_NONE {
            return;
        }
        let c = &core.cpu;
        let meta = [self.cur, halt, c.a, c.x, c.y, c.sp, c.p()];
        self.cur = PH_NONE;
        let Some(mut tr) = core.s.rec.as_deref_mut().and_then(|r| r.tracer.take()) else {
            return;
        };
        let rs = &mut self.records;
        if rs.meta.is_empty() {
            rs.origin = tr.base().to_vec();
        }
        let r = tr.snap(&core.s);
        tr.phase = PH_NONE;
        if let Some(rec) = core.s.rec.as_deref_mut() {
            rec.tracer = Some(tr);
        }
        rs.meta.push(meta);
        rs.log.extend(r.log);
        rs.log_end.push(rs.log.len() as u64);
        rs.idx.extend(r.idx);
        rs.vals.extend(r.vals);
        rs.delta_end.push(rs.idx.len() as u64);
    }

    /// The sentinel's trap: close the returning call's record and start the next call, or
    /// stop the run (false) when none is left.
    pub fn trap(&mut self, core: &mut Core) -> bool {
        self.snap(core, HALT_RETURN);
        let Some(&(phase, pc, a)) = self.queue.get(self.next) else {
            core.stop = true;
            return false;
        };
        self.next += 1;
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
                Self::host_write(core, 1, port);
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
        Self::host_write(core, 0x100 | sp as u16, (ret >> 8) as u8);
        Self::host_write(core, 0x100 | sp.wrapping_sub(1) as u16, ret as u8);
        (core.cpu.sp, core.cpu.a, core.cpu.pc) = (sp0, a, pc);
        let deadline = if self.call_cycles > 0 {
            core.t + self.call_cycles
        } else {
            i64::MAX
        };
        if let Some(r) = core.s.rec.as_deref_mut() {
            (r.deadline, r.timed_out) = (deadline, false);
        }
        if phase == PH_CALL {
            self.entry_sp.insert((pc, sp0));
            self.entry_port
                .insert((pc, core.s.port_dir, core.s.port_data));
        }
        true
    }
}
