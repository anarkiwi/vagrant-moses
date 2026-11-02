//! The core on hand-assembled programs, with the cycle counts the 6510 documents.

use moses_core::lines::{CIA, EV_ENTRY};
use moses_core::machine::{BRK_LAYOUT, CLI, F, POPSR, R, RTI, SEI, W};
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
        (0xEA, &[R, F], 1, 0),
        (0x58, &[R, CLI], 1, 0),
        (0x78, &[R, SEI], 1, 0),
        (0x40, &[R, R, POPSR, R, R, RTI], 2, 0),
        (0xAD, &[R, R, R, F], 3, 0),
    ];
    for (op, ents, exec_at, kind) in ops {
        table[op] = Layout {
            ents: ents.to_vec(),
            exec_at,
            kind,
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
    assert_eq!(c.s.img[0][0x2000], 0x40);
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
    let mut tr = Tracer::new([0; 256], 0xFFFF, 0xD000, 0xE000, &c.s.img[0]);
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
    let ram = &c.s.img[0];
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
