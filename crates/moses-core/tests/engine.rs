//! The core on hand-assembled programs, with the cycle counts the 6510 documents.

use moses_core::lines::{CIA, EV_ENTRY};
use moses_core::machine::{BRK_LAYOUT, CLI, DP, F, POPSR, R, RTI, SEI, W};
use std::sync::Arc;

use moses_core::trace::{Tracer, PH_CALL};
use moses_core::{smc, Config, Core, Host, Layout, Recorder, SidDev, SidShared};

#[derive(Default)]
struct Log {
    sid: Vec<(i64, u8, u8)>,
    reads: Vec<u8>,
    io: Vec<(u16, u8, bool)>,
}

impl Host for Log {
    fn sid_read(&mut self, _dev: usize, _cycle: i64, reg: u8) -> u8 {
        self.reads.push(reg);
        0
    }
    fn sid_write(&mut self, _dev: usize, cycle: i64, reg: u8, value: u8) {
        self.sid.push((cycle, reg, value));
    }
    fn event(&mut self, _cycle: i64, _name: u8) {}
    fn trap(&mut self, _core: &mut Core) -> bool {
        false
    }
    fn decode(&mut self, _core: &mut Core) {}
    fn port_write(&mut self, _core: &mut Core) {}
    fn io(&mut self, addr: u16, value: u8, write: bool) {
        self.io.push((addr, value, write));
    }
}

fn image() -> Box<[u8; 0x10000]> {
    Box::new([0; 0x10000])
}

/// The opcodes the programs use, as `Machine.LAYOUTS` lays them out.
fn layouts() -> Vec<Layout> {
    let mut table = vec![Layout::default(); BRK_LAYOUT];
    let ops: [(usize, &[u8], i32, u8); 8] = [
        (0xA9, &[R, F], 1, 0),
        (0x8D, &[R, R, W, F], 2, 0),
        (0x4C, &[R, R, F], 2, 0),
        (0xEA, &[DP, F], 1, 0),
        (0x58, &[DP, CLI], 1, 0),
        (0x78, &[DP, SEI], 1, 0),
        (0x40, &[DP, R, POPSR, R, R, RTI], 2, 0),
        (0xAD, &[R, R, R, F], 3, 0),
    ];
    for (op, ents, exec_at, kind) in ops {
        table[op] = Layout {
            ents: ents.to_vec(),
            exec_at,
            rmw: false,
            mode: kind,
            base: (op as i32) << 3,
            branch: None,
        };
    }
    table
}

fn core(program: &[(u16, &[u8])]) -> Core {
    core_with(program, SidDev::Host(0), None)
}

fn core_with(program: &[(u16, &[u8])], sid: SidDev, reu: Option<u32>) -> Core {
    let mut ram = image();
    let mut kernal = image();
    kernal[0xFFFE] = 0x00;
    kernal[0xFFFF] = 0x11;
    for (org, bytes) in program {
        ram[*org as usize..*org as usize + bytes.len()].copy_from_slice(bytes);
    }
    Core::new(Config {
        ntsc: false,
        cia_new: false,
        ram,
        roms: [image(), image(), kernal],
        layouts: layouts(),
        sids: vec![(0xD400, sid)],
        custom_sink: false,
        lines: 312,
        cycles_per_line: 63,
        reu,
    })
}

#[test]
fn a_loop_writes_the_sid_every_nine_cycles() {
    // LDA #$55 / STA $D400 / JMP $1000: 2 + 4 + 3 cycles.
    let mut c = core(&[(0x1000, &[0xA9, 0x55, 0x8D, 0x00, 0xD4, 0x4C, 0x00, 0x10])]);
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    c.run(&mut host, 2200);
    assert!(host.sid.len() > 15);
    assert!(host.sid.iter().all(|&(_, reg, v)| reg == 0 && v == 0x55));
    assert!(host.sid.windows(2).all(|w| w[1].0 - w[0].0 == 9));
    assert_eq!(c.t, 2200);
}

#[test]
fn a_cia_timer_raises_its_irq_every_latch_plus_one_cycles() {
    let setup = [
        0x78, // SEI
        0xA9, 0x7F, 0x8D, 0x0D, 0xDC, // mask every source
        0xA9, 0x34, 0x8D, 0x04, 0xDC, // latch low
        0xA9, 0x12, 0x8D, 0x05, 0xDC, // latch high
        0xA9, 0x81, 0x8D, 0x0D, 0xDC, // enable the timer A interrupt
        0xA9, 0x11, 0x8D, 0x0E, 0xDC, // start, force load
        0x58, // CLI
        0x4C, 0x1B, 0x10, // idle
    ];
    let handler = [0xAD, 0x0D, 0xDC, 0x40]; // LDA $DC0D / RTI
    let mut c = core(&[(0x1000, &setup), (0x1100, &handler)]);
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    c.run(&mut host, 2000 + 0x1235 * 6);
    let raises: Vec<i64> =
        c.s.events
            .iter()
            .filter(|e| e.1 == 2)
            .map(|e| e.0)
            .collect();
    assert!(raises.len() >= 5);
    assert!(raises.windows(2).all(|w| w[1] - w[0] == 0x1235));
    let entries =
        c.s.events
            .iter()
            .filter(|e| e.1 == EV_ENTRY + CIA as u8)
            .count();
    assert_eq!(entries, raises.len() - usize::from(c.s.ln.pin));
}

#[test]
fn rdy_stalls_the_cpu_on_a_badline() {
    // DEN with the raster in the display window: bad lines steal cycles from LDA/STA loops.
    let program = [
        0xA9, 0x1B, 0x8D, 0x11, 0xD0, // $D011 = $1B
        0xEA, 0x4C, 0x05, 0x10,
    ];
    let mut quiet = core(&[(0x1000, &[0xEA, 0x4C, 0x00, 0x10])]);
    let mut busy = core(&[(0x1000, &program)]);
    let mut host = Log::default();
    for c in [&mut quiet, &mut busy] {
        c.reset(&mut host, 0x1000, 2000);
        c.run(&mut host, 2000 + 312 * 63 * 2);
    }
    // The same cycle count retires fewer instructions when BA is pulled low.
    assert!(busy.cpu.cycles < quiet.cpu.cycles - 400);
}

#[test]
fn a_logged_sid_calls_the_host_only_for_readable_registers() {
    // LDA #$55 / STA $D400 / LDA $D41B / LDA $D400 / JMP $1000
    let program = [
        0xA9, 0x55, 0x8D, 0x00, 0xD4, 0xAD, 0x1B, 0xD4, 0xAD, 0x00, 0xD4, 0x4C, 0x00, 0x10,
    ];
    let shared = Arc::new(SidShared::new(false));
    let mut c = core_with(
        &[(0x1000, &program)],
        SidDev::Logged(Arc::clone(&shared), 0),
        None,
    );
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    c.run(&mut host, 2200);
    assert!(host.sid.is_empty());
    assert!(!host.reads.is_empty() && host.reads.iter().all(|&r| r == 0x1B));
    // SAFETY: the run has returned.
    let state = unsafe { shared.state() };
    assert!(state.writes.len() > 5);
    assert!(state
        .writes
        .iter()
        .all(|&(_, reg, v)| reg == 0 && v == 0x55));
    assert_eq!(state.rows.len(), 1);
}

#[test]
fn an_reu_copies_c64_memory_while_the_cpu_halts() {
    let program = [
        0xA9, 0x10, 0x8D, 0x03, 0xDF, // C64 base $1000
        0xA9, 0x05, 0x8D, 0x07, 0xDF, // five bytes
        0xA9, 0x00, 0x8D, 0x08, 0xDF, 0xA9, 0x90, 0x8D, 0x01, 0xDF, // execute C64 -> REU now
        0xAD, 0x00, 0xDF, // status
        0x8D, 0x00, 0x20, // store it at $2000
        0x4C, 0x1A, 0x10,
    ];
    let mut c = core_with(&[(0x1000, &program)], SidDev::Host(0), Some(128));
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    c.run(&mut host, 2100);
    let reu = c.s.reu.as_ref().unwrap();
    assert_eq!(&reu.ram[..5], &program[..5]);
    assert_eq!(c.s.ram[0x2000], 0x40);
    assert_eq!(
        (reu.peek(2), reu.peek(3), reu.peek(4), reu.peek(7)),
        (5, 0x10, 5, 1)
    );
}

#[test]
fn the_recorder_reports_a_store_into_a_later_operand_once() {
    // LDA #$42 / STA $1006 / LDA #$00 / JMP $1007: the store patches the second LDA's operand.
    let code = [0xA9, 0x42, 0x8D, 0x06, 0x10, 0xA9, 0x00, 0x4C, 0x07, 0x10];
    let mut c = core(&[(0x1000, &code)]);
    let mut table = [1u8; 256];
    (table[0xA9], table[0x8D], table[0x4C]) = (2, 3, 3);
    let mut base = image();
    base[0x1000..0x100A].copy_from_slice(&code);
    let mut rec = Recorder::new(base, table, true);
    rec.begin(c.s.bits);
    c.s.rec = Some(Box::new(rec));
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    c.run(&mut host, 2100);
    let rec = c.s.rec.as_ref().unwrap();
    let got: Vec<_> = rec.events.iter().map(|e| (e.fields(), e.first)).collect();
    assert_eq!(
        got,
        [(
            (
                smc::W_CPU,
                Some(0x8D),
                smc::R_OPERAND,
                smc::T_SAME_CALL,
                smc::M_FLAT
            ),
            0
        )]
    );
    assert_eq!(rec.unclassified, 0);
}

#[test]
fn the_trace_observes_the_io_values_the_machine_answers() {
    let program = [
        0xA9, 0x55, 0x8D, 0x00, 0xD4, // write the SID bus latch
        0xAD, 0x05, 0xD4, 0xAD, 0x1B, 0xD4, // an unreadable register, then OSC3
        0xAD, 0x12, 0xD0, 0x8D, 0x00, 0x20, // raster
        0xAD, 0x04, 0xDC, 0x8D, 0x01, 0x20, // CIA 1 timer A
        0xAD, 0x00, 0xDE, 0x8D, 0x02, 0x20, // open I/O 1
        0x4C, 0x1D, 0x10,
    ];
    let late = [0xAD, 0x05, 0xD4, 0x4C, 0x03, 0x11];
    let shared = Arc::new(SidShared::new(false));
    let mut c = core_with(
        &[(0x1000, &program), (0x1100, &late)],
        SidDev::Logged(Arc::clone(&shared), 0),
        None,
    );
    let mut rec = Recorder::new(image(), [1; 256], true);
    rec.begin(c.s.bits);
    let mut tr = Tracer::new([0; 256], 0xFFFF, 0xD000, 0xE000, &c.s.ram);
    tr.phase = PH_CALL;
    rec.tracer = Some(Box::new(tr));
    c.s.rec = Some(Box::new(rec));
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    c.run(&mut host, 2200);
    c.reset(&mut host, 0x1100, 2200 + 0x1D00);
    c.run(&mut host, 2200 + 0x1D00 + 20);
    let seen = host.io.clone();
    for addr in [0xD012, 0xD405, 0xDC04] {
        c.peek(&mut host, addr);
    }
    let mut tr = c.s.rec.as_mut().unwrap().tracer.take().unwrap();
    let log = tr.snap(&c.s).log;
    let ram = &c.s.ram;
    assert_eq!(
        log,
        [
            (0xD400, 0x55, true),
            (0xD405, 0x55, false),
            (0xD41B, 0, false),
            (0xD012, ram[0x2000], false),
            (0xDC04, ram[0x2001], false),
            (0xDE00, ram[0x2002], false),
            (0xD405, 0, false),
        ]
    );
    assert_eq!(host.reads, [0x1B]);
    assert_eq!(host.io, seen);
    assert_eq!(log, seen);
}

#[test]
fn a_fork_runs_cycle_identically_and_independently() {
    // A CIA timer IRQ whose handler writes a logged SID, with an REU attached.
    let setup = [
        0x78, 0xA9, 0x7F, 0x8D, 0x0D, 0xDC, 0xA9, 0x34, 0x8D, 0x04, 0xDC, 0xA9, 0x02, 0x8D, 0x05,
        0xDC, 0xA9, 0x81, 0x8D, 0x0D, 0xDC, 0xA9, 0x11, 0x8D, 0x0E, 0xDC, 0x58, 0x4C, 0x1B, 0x10,
    ];
    let handler = [0x8D, 0x00, 0xD4, 0xAD, 0x0D, 0xDC, 0x40]; // STA $D400 / LDA $DC0D / RTI
    let shared = Arc::new(SidShared::new(false));
    let mut a = core_with(
        &[(0x1000, &setup), (0x1100, &handler)],
        SidDev::Logged(Arc::clone(&shared), 0),
        Some(128),
    );
    let mut host = Log::default();
    a.reset(&mut host, 0x1000, 2000);
    a.run(&mut host, 4321);
    let mut b = a.fork().unwrap();
    // SAFETY: neither machine is running.
    let logged = unsafe { shared.state() }.writes.len();
    a.run(&mut host, 40_000);
    b.run(&mut host, 40_000);
    assert_eq!((a.t, &a.cpu, &a.s.events), (b.t, &b.cpu, &b.s.events));
    assert_eq!(a.s.ram, b.s.ram);
    assert!(a.s.events.len() > 10);
    let SidDev::Logged(fork, _) = &b.s.sids[0].1 else {
        panic!("the fork's SID is not logged");
    };
    // SAFETY: as above.
    let (ours, theirs) = unsafe { (shared.state(), fork.state()) };
    assert_eq!(ours.writes[logged..], theirs.writes[..]);
    b.poke(0x2000, 0x99);
    assert_ne!(a.s.ram[0x2000], 0x99);
    a.s.rec = Some(Box::new(Recorder::new(image(), [1; 256], true)));
    assert!(a.fork().is_none());
}

#[test]
fn a_program_runs_to_pcs_counts_the_marker_and_forks_agree() {
    use moses_core::prog::{self, Alu, Cond, Op, Prog, Src, ST_BUDGET, ST_HALT};
    use moses_core::Watch;
    // $1000: LDA #$55 / STA $2000 / JMP $1000
    let mut c = core(&[(0x1000, &[0xA9, 0x55, 0x8D, 0x00, 0x20, 0x4C, 0x00, 0x10])]);
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    let at = |pcs: &[u16]| Watch::Set(Arc::new(pcs.iter().copied().collect()));
    let ops = vec![
        Op::Run {
            watch: at(&[0x1000, 0x1005]),
            target: at(&[0x1005]),
            n: Src::Imm(3),
            here: false,
        },
        Op::Frames(1),
        Op::Alu {
            op: Alu::Add,
            d: 1,
            s: Src::Imm(100),
        },
        Op::Br {
            cond: Cond::Eq,
            a: 1,
            b: Src::Imm(103),
            to: 5,
        },
        Op::Halt(Src::Imm(9)),
        Op::Out(1),
        Op::Halt(Src::Reg(1)),
    ];
    let p = Arc::new(Prog::new(ops, 1000, Some(0x1000), 100).unwrap());
    let o = prog::run(&mut c, &mut host, &p);
    assert_eq!(
        (o.status, o.code, o.frames, &o.out[..]),
        (ST_HALT, 103, 3, &[103][..])
    );
    assert!(c.parked && c.cpu.pc == 0x1005 && c.s.ram[0x2000] == 0x55);
    let spin = Arc::new(
        Prog::new(
            vec![Op::Run {
                watch: Watch::All,
                target: Watch::All,
                n: Src::Imm(1 << 20),
                here: false,
            }],
            500,
            None,
            100,
        )
        .unwrap(),
    );
    let t = c.t;
    assert_eq!(
        prog::run(&mut c.fork().unwrap(), &mut host, &spin).status,
        ST_BUDGET
    );
    let progs: Vec<_> = (0..12)
        .map(|i| if i % 3 == 0 { &spin } else { &p })
        .cloned()
        .collect();
    let one = prog::run_forks(&mut c, &progs, 1, false, |_| Log::default());
    let many = prog::run_forks(&mut c, &progs, 5, true, |_| Log::default());
    assert_eq!(c.t, t);
    for ((a, _, _), (b, _, f)) in one.iter().zip(&many) {
        assert_eq!(a, b);
        assert_eq!(f.as_ref().unwrap().t, a.cycle);
    }
    assert!(Prog::new(vec![Op::Jmp(2)], 1, None, 1).is_none());
}

#[test]
fn segments_key_on_what_they_read_and_forks_fold_into_the_base() {
    use moses_core::prog::{self, Op, Prog, Src};
    use moses_core::seg::{Segs, END_IO, END_PC};
    use moses_core::Watch;
    // $1000: LDA $2000 / STA $2001 / LDA $D020 / JMP $1000
    let code = [
        0xAD, 0x00, 0x20, 0x8D, 0x01, 0x20, 0xAD, 0x20, 0xD0, 0x4C, 0x00, 0x10,
    ];
    let keys = |byte: u8, poke: u16| {
        let mut c = core(&[(0x1000, &code), (0x2000, &[byte])]);
        let mut host = Log::default();
        c.reset(&mut host, 0x1000, 2000);
        c.poke(poke, 0x77);
        c.s.seg = Some(Box::new(Segs::new(64, None, false)));
        c.run(&mut host, 4000);
        let g = c.s.seg.as_mut().unwrap();
        let mut k: Vec<_> = g.table().keys().copied().collect();
        k.sort_unstable_by_key(|k| (k.entry, k.reads, k.n));
        (k, g.summary(4))
    };
    let (base, s) = keys(1, 0x3000);
    assert_eq!(s.tot.segs[END_IO], s.tot.segs.iter().sum::<u64>() - 1);
    assert_eq!(s.tot.cycles[END_IO], 11 * s.tot.segs[END_IO] - 3);
    assert_eq!(s.tot.conflicts, 0);
    assert_eq!(keys(1, 0x2001).0, base);
    assert_ne!(keys(2, 0x3000).0, base);

    let mut c = core(&[(0x1000, &code)]);
    let mut host = Log::default();
    c.reset(&mut host, 0x1000, 2000);
    let pcs = Arc::new([0x1000u16].into_iter().collect());
    c.s.seg = Some(Box::new(Segs::new(64, Some(pcs), false)));
    let run = Op::Run {
        watch: Watch::All,
        target: Watch::All,
        n: Src::Imm(40),
        here: false,
    };
    let p = Arc::new(Prog::new(vec![run], 1000, None, 10).unwrap());
    prog::run_forks(&mut c, &[Arc::clone(&p), p], 2, false, |_| Log::default());
    let s = c.s.seg.as_mut().unwrap().summary(4);
    assert_eq!((s.batches, s.tot.forks), (1, 2));
    assert_eq!(s.tot.segs[END_PC], 2 * 9);
    // Per fork: the first [LDA, STA] from reset, nine more and nine [JMP].
    let own = 8 * 8 + 8 * 3;
    assert_eq!(s.tot.saved, [2 * own, 2 * own + 8 + 8 + 3]);
    assert_eq!((s.global, s.distinct), (s.tot.saved[1], 3));
    assert_eq!(s.top[0].first, 1 << 32);
}
