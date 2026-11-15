//! Driver programs: a register machine whose ops drive the keyboard matrix and run the core
//! to program counters, run inside [`Core::run`] with no host call per op.
//!
//! The ops are `vagrant_moses.program`'s (`docs/native.md`, Driver programs). A run op
//! watches its stop set through the core's trap: [`Prog`]'s host stops the run at the decode
//! that completes it, before the instruction executes, and counts arrivals at the marker.

use std::sync::Arc;

use crate::machine::{Core, Watch};
use crate::seg::Batch;
use crate::sys::{port_read, Host, Sys, IO};

/// An operand: a register or an immediate.
#[derive(Clone, Copy, Debug)]
pub enum Src {
    Reg(u8),
    Imm(u32),
}

#[derive(Clone, Copy, Debug)]
pub enum Alu {
    Mov,
    Add,
    Sub,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Add8,
    Sub8,
}

#[derive(Clone, Copy, Debug)]
pub enum Cond {
    Eq,
    Ne,
    Lt,
    Ge,
}

pub enum Op {
    Keys([u8; 8]),
    Key {
        row: u8,
        col: u8,
        down: bool,
    },
    /// The key `row * 8 + col` in the low 6 bits of a register.
    KeyR {
        r: u8,
        down: bool,
    },
    /// Run until `n` more decodes at a pc in `target` (`watch` is `target` and the marker);
    /// with `here`, none when parked at a pc in `target` already.
    Run {
        watch: Watch,
        target: Watch,
        n: Src,
        here: bool,
    },
    /// `r = mem[addr + idx]` in view `mode` ([`LD_CPU`], [`LD_RAM`], [`LD_IO`]).
    Ld {
        r: u8,
        addr: u16,
        idx: Option<u8>,
        mode: u8,
    },
    Alu {
        op: Alu,
        d: u8,
        s: Src,
    },
    Cyc {
        lo: u8,
        hi: u8,
    },
    Frames(u8),
    Br {
        cond: Cond,
        a: u8,
        b: Src,
        to: usize,
    },
    /// Branch when bit 7 of a register is `set`.
    Bit7 {
        r: u8,
        set: bool,
        to: usize,
    },
    Jmp(usize),
    Djnz {
        r: u8,
        to: usize,
    },
    Out(u8),
    OutMem {
        addr: u16,
        len: u16,
    },
    Halt(Src),
}

/// The CPU's view under the live banking ([`peek`]).
pub const LD_CPU: u8 = 0;
/// RAM, whatever the banking.
pub const LD_RAM: u8 = 1;
/// The I/O page, whatever the banking ([`peek_io`]).
pub const LD_IO: u8 = 2;

pub const ST_HALT: u8 = 0;
pub const ST_BUDGET: u8 = 1;
pub const ST_JAM: u8 = 2;
pub const ST_OPS: u8 = 3;
pub const ST_FAULT: u8 = 4;
pub const STATUSES: [&str; 5] = ["halt", "budget", "jam", "ops", "fault"];
pub const REGS: usize = 16;

/// A validated program: ops, the cycle budget, the frame marker and the op budget.
pub struct Prog {
    pub ops: Vec<Op>,
    pub budget: u64,
    pub marker: Option<u16>,
    pub max_ops: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub status: u8,
    pub code: u32,
    pub cycle: i64,
    pub frames: u32,
    pub out: Vec<u32>,
}

impl Prog {
    /// `None` unless every register and branch target is in range.
    pub fn new(ops: Vec<Op>, budget: u64, marker: Option<u16>, max_ops: u64) -> Option<Prog> {
        let n = ops.len();
        let reg = |r: u8| (r as usize) < REGS;
        let src = |s: Src| !matches!(s, Src::Reg(r) if !reg(r));
        let ok = ops.iter().all(|op| match *op {
            Op::KeyR { r, .. } | Op::Frames(r) | Op::Out(r) => reg(r),
            Op::Key { row, col, .. } => row < 8 && col < 8,
            Op::Run { n, .. } | Op::Halt(n) => src(n),
            Op::Ld { r, idx, mode, .. } => reg(r) && idx.is_none_or(reg) && mode <= LD_IO,
            Op::Alu { d, s, .. } => reg(d) && src(s),
            Op::Cyc { lo, hi } => reg(lo) && reg(hi),
            Op::Br { a, b, to, .. } => reg(a) && src(b) && to <= n,
            Op::Bit7 { r, to, .. } | Op::Djnz { r, to } => reg(r) && to <= n,
            Op::Jmp(to) => to <= n,
            Op::Keys(_) | Op::OutMem { .. } => true,
        });
        ok.then_some(Prog {
            ops,
            budget,
            marker,
            max_ops,
        })
    }
}

/// An I/O register as a CPU read now would return it, made on copies of the chip, the
/// scheduler and the lines; `None` outside the VIC-II, colour RAM and CIAs.
pub fn peek_io(s: &Sys, addr: u16) -> Option<u8> {
    let copies = || (s.sched.clone(), s.ln.clone());
    Some(match addr >> 8 {
        0xD0..=0xD3 => {
            let (mut sched, mut ln) = copies();
            s.vic.clone().read(addr, &mut sched, &mut ln)
        }
        0xD8..=0xDB => s.colour[(addr & 0x3FF) as usize],
        0xDC | 0xDD => {
            let (mut sched, mut ln) = copies();
            s.cia[(addr >> 8 != 0xDC) as usize]
                .clone()
                .read(addr, &mut sched, &mut ln)
        }
        _ => return None,
    })
}

/// `addr` as the CPU sees it, with no side effect ([`peek_io`] for the I/O page).
pub fn peek(s: &Sys, addr: u16) -> Option<u8> {
    match addr {
        0 => Some(s.port_dir),
        1 => Some(port_read(s.port_dir, s.port_data)),
        _ => match s.map[(addr >> 12) as usize] {
            IO => peek_io(s, addr),
            k => Some(s.image(k)[addr as usize]),
        },
    }
}

/// The program's host: its trap counts the marker and stops the run op; the rest is `h`'s.
struct Driver<'a, H: Host> {
    h: &'a mut H,
    target: &'a Watch,
    left: u32,
    marker: u32,
    frames: u32,
}

impl<H: Host> Host for Driver<'_, H> {
    fn sid_read(&mut self, dev: usize, cycle: i64, reg: u8) -> u8 {
        self.h.sid_read(dev, cycle, reg)
    }
    fn sid_write(&mut self, dev: usize, cycle: i64, reg: u8, value: u8) {
        self.h.sid_write(dev, cycle, reg, value)
    }
    fn event(&mut self, cycle: i64, name: u8) {
        self.h.event(cycle, name)
    }
    fn io(&mut self, addr: u16, value: u8, write: bool) {
        self.h.io(addr, value, write)
    }
    fn failed(&self) -> bool {
        self.h.failed()
    }
    fn trap(&mut self, core: &mut Core) -> bool {
        let pc = core.cpu.pc;
        self.frames += (pc as u32 == self.marker) as u32;
        if self.left != 0 && self.target.contains(pc) {
            self.left -= 1;
            if self.left == 0 {
                core.stop = true;
                core.parked = true;
            }
        }
        false
    }
    fn decode(&mut self, _core: &mut Core) {}
    fn port_write(&mut self, _core: &mut Core) {}
    fn trap_writes(&self) -> bool {
        false
    }
}

/// Run `prog` on `core`: the trap, decode and port hooks and `arm_at` are set aside and
/// restored after. A host failure ends the program as halted; the caller reports it.
pub fn run<H: Host>(core: &mut Core, h: &mut H, prog: &Prog) -> Outcome {
    let saved = (
        std::mem::replace(&mut core.watch, Watch::none()),
        core.trap_set,
        core.decode_set,
        core.port_set,
        core.arm_pc.take(),
    );
    (core.trap_set, core.decode_set, core.port_set) = (true, false, false);
    let none = Watch::none();
    let mut d = Driver {
        h,
        target: &none,
        left: 0,
        marker: prog.marker.map_or(u32::MAX, u32::from),
        frames: 0,
    };
    let limit = core
        .t
        .saturating_add(prog.budget.min(i64::MAX as u64) as i64);
    let mut reg = [0u32; REGS];
    let mut out = Vec::new();
    let (mut pc, mut ops) = (0, 0u64);
    let (status, code) = loop {
        let Some(op) = prog.ops.get(pc) else {
            break (ST_HALT, 0);
        };
        if ops == prog.max_ops {
            break (ST_OPS, pc as u32);
        }
        ops += 1;
        pc += 1;
        let val = |s: Src, reg: &[u32; REGS]| match s {
            Src::Reg(r) => reg[r as usize],
            Src::Imm(v) => v,
        };
        match op {
            Op::Keys(rows) => core.s.cia[0].keys = *rows,
            &Op::Key { row, col, down } => key(core, row, col, down),
            &Op::KeyR { r, down } => {
                let k = reg[r as usize] as u8 & 63;
                key(core, k >> 3, k & 7, down);
            }
            Op::Run {
                watch,
                target,
                n,
                here,
            } => {
                let n = val(*n, &reg);
                if n == 0 || *here && core.parked && target.contains(core.cpu.pc) {
                    continue;
                }
                core.watch = watch.clone();
                (d.target, d.left) = (target, n);
                core.run(&mut d, limit);
                d.target = &none;
                if d.h.failed() {
                    break (ST_HALT, 0);
                }
                if core.jammed {
                    break (ST_JAM, pc as u32 - 1);
                }
                if d.left != 0 {
                    break (ST_BUDGET, pc as u32 - 1);
                }
            }
            &Op::Ld { r, addr, idx, mode } => {
                let a = addr.wrapping_add(idx.map_or(0, |i| reg[i as usize] as u16));
                let v = match mode {
                    LD_CPU => peek(&core.s, a),
                    LD_RAM => Some(core.s.ram[a as usize]),
                    _ => peek_io(&core.s, a),
                };
                match v {
                    Some(v) => reg[r as usize] = v as u32,
                    None => break (ST_FAULT, a as u32),
                }
            }
            &Op::Alu { op, d, s } => {
                let (x, y) = (reg[d as usize], val(s, &reg));
                reg[d as usize] = match op {
                    Alu::Mov => y,
                    Alu::Add => x.wrapping_add(y),
                    Alu::Sub => x.wrapping_sub(y),
                    Alu::And => x & y,
                    Alu::Or => x | y,
                    Alu::Xor => x ^ y,
                    Alu::Shl => x << (y & 31),
                    Alu::Shr => x >> (y & 31),
                    Alu::Add8 => x.wrapping_add(y) & 0xFF,
                    Alu::Sub8 => x.wrapping_sub(y) & 0xFF,
                };
            }
            &Op::Cyc { lo, hi } => {
                reg[hi as usize] = (core.t >> 32) as u32;
                reg[lo as usize] = core.t as u32;
            }
            &Op::Frames(r) => reg[r as usize] = d.frames,
            &Op::Br { cond, a, b, to } => {
                let (x, y) = (reg[a as usize], val(b, &reg));
                if match cond {
                    Cond::Eq => x == y,
                    Cond::Ne => x != y,
                    Cond::Lt => x < y,
                    Cond::Ge => x >= y,
                } {
                    pc = to;
                }
            }
            &Op::Bit7 { r, set, to } => {
                if (reg[r as usize] & 0x80 != 0) == set {
                    pc = to;
                }
            }
            &Op::Jmp(to) => pc = to,
            &Op::Djnz { r, to } => {
                let v = &mut reg[r as usize];
                *v = v.wrapping_sub(1);
                if *v != 0 {
                    pc = to;
                }
            }
            &Op::Out(r) => out.push(reg[r as usize]),
            &Op::OutMem { addr, len } => {
                let fault = (0..len)
                    .map(|i| addr.wrapping_add(i))
                    .find(|&a| peek(&core.s, a).map(|b| out.push(b as u32)).is_none());
                if let Some(a) = fault {
                    break (ST_FAULT, a as u32);
                }
            }
            &Op::Halt(s) => break (ST_HALT, val(s, &reg)),
        }
    };
    let frames = d.frames;
    (
        core.watch,
        core.trap_set,
        core.decode_set,
        core.port_set,
        core.arm_pc,
    ) = saved;
    Outcome {
        status,
        code,
        cycle: core.t,
        frames,
        out,
    }
}

fn key(core: &mut Core, row: u8, col: u8, down: bool) {
    let keys = &mut core.s.cia[0].keys[row as usize];
    *keys = *keys & !(1 << col) | (down as u8) << col;
}

/// Each program on its own fork of `base`, on `threads` threads, in program order.
/// `host(i)` is the host fork `i` runs with; `keep` returns each fork after its program.
/// A segment profiler armed on `base` takes in the forks' segments as one batch; a kept
/// fork's profiler starts afresh.
pub fn run_forks<H: Host + Send, F: Fn(usize) -> H + Sync>(
    base: &mut Core,
    progs: &[Arc<Prog>],
    threads: usize,
    keep: bool,
    host: F,
) -> Vec<(Outcome, H, Option<Core>)> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let batch = base.s.seg.as_ref().map(|g| g.next_batch());
    let shared: &Core = base;
    let work = || {
        let (mut done, mut part) = (Vec::new(), Batch::default());
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(prog) = progs.get(i) else {
                return (done, part);
            };
            let mut core = shared
                .fork()
                .expect("forks are refused while the recorder is armed");
            core.bare();
            if let (Some(g), Some(b)) = (core.s.seg.as_deref_mut(), batch) {
                g.src = b << 32 | i as u64;
            }
            let mut h = host(i);
            let o = run(&mut core, &mut h, prog);
            if let Some(g) = core.s.seg.take() {
                if keep {
                    core.s.seg = Some(Box::new(g.spawn()));
                }
                part.fold(*g);
            }
            done.push((i, o, h, keep.then_some(core)));
        }
    };
    let (mut all, total) = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.max(1)).map(|_| s.spawn(work)).collect();
        workers
            .into_iter()
            .fold((Vec::new(), Batch::default()), |(mut all, mut total), w| {
                let (done, part) = w.join().expect("a fork's thread panicked");
                all.extend(done);
                total.join(part);
                (all, total)
            })
    });
    if let Some(g) = base.s.seg.as_deref_mut() {
        g.absorb(total);
    }
    all.sort_unstable_by_key(|r| r.0);
    all.into_iter().map(|(_, o, h, c)| (o, h, c)).collect()
}
