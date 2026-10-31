//! The core on hand-assembled programs, with the cycle counts the 6510 documents.

use moses_core::machine::{BRK_LAYOUT, CLI, F, POPSR, R, RTI, SEI, W};
use moses_core::{Config, Core, Host, Layout};

#[derive(Default)]
struct Log {
    sid: Vec<(i64, u8, u8)>,
}

impl Host for Log {
    fn sid_read(&mut self, _dev: usize, _cycle: i64, _reg: u8) -> u8 {
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
        sids: vec![(0xD400, 0)],
        custom_sink: false,
        lines: 312,
        cycles_per_line: 63,
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
    let entries = c.s.events.iter().filter(|e| e.1 == 7).count();
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
