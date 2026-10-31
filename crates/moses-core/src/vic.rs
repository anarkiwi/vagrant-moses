//! The VIC-II as libsidplayfp's MOS656X models it: raster IRQ, badline and sprite BA.
//!
//! A port of `mos656x.cpp` and `sprites.h`, so raster interrupts and the cycles the VIC
//! steals from the CPU land on the same cycles as in the sidtrace oracle. The light pen
//! is not modelled.

use crate::lines::{Lines, VIDEO};
use crate::sched::{Sched, E_BADLINE, E_EDGE, E_VIC, PHI1};

const FIRST_DMA_LINE: i32 = 0x30;
const LAST_DMA_LINE: i32 = 0xF7;
const FETCH_CYCLE: i32 = 11;
const TEXT_COLS: i32 = 40;
const IRQ_RASTER: u8 = 1;

/// Sprite DMA state (`sprites.h`): which sprites fetch data this line.
#[derive(Clone, Default)]
pub struct Sprites {
    pub exp_flop: u8,
    pub dma: u8,
    pub mc_base: [u8; 8],
    pub mc: [u8; 8],
}

impl Sprites {
    fn reset(&mut self) {
        *self = Sprites {
            exp_flop: 0xFF,
            ..Sprites::default()
        };
    }

    fn update_mc(&mut self) {
        for i in 0..8 {
            if self.dma >> i & 1 != 0 {
                self.mc[i] = (self.mc[i] + 3) & 0x3F;
            }
        }
    }

    fn update_mc_base(&mut self) {
        for i in 0..8 {
            if self.exp_flop >> i & 1 != 0 {
                self.mc_base[i] = self.mc[i];
                if self.mc_base[i] == 0x3F {
                    self.dma &= !(1 << i);
                }
            }
        }
    }

    fn check_exp(&mut self, regs: &[u8; 0x40]) {
        self.exp_flop ^= self.dma & regs[0x17];
    }

    fn check_display(&mut self) {
        self.mc = self.mc_base;
    }

    fn check_dma(&mut self, regs: &[u8; 0x40], raster_y: i32) {
        let y = (raster_y & 0xFF) as u8;
        for i in 0..8 {
            let mask = 1u8 << i;
            if regs[0x15] & mask != 0 && y == regs[2 * i + 1] && self.dma & mask == 0 {
                self.dma |= mask;
                self.mc_base[i] = 0;
                self.exp_flop |= mask;
            }
        }
    }

    fn line_crunch(&mut self, data: u8, line_cycle: i32) {
        for i in 0..8 {
            let mask = 1u8 << i;
            if data & mask == 0 && self.exp_flop & mask == 0 {
                if line_cycle == 14 {
                    let (mc, base) = (self.mc[i], self.mc_base[i]);
                    self.mc[i] = (0x2A & (base & mc)) | (0x15 & (base | mc));
                }
                self.exp_flop |= mask;
            }
        }
    }
}

/// MOS6569 (PAL) or MOS6567R8 (NTSC) raster, interrupt and BA timing.
#[derive(Clone)]
pub struct Vic {
    ntsc: bool,
    max_rasters: i32,
    cycles_per_line: i32,
    pub regs: [u8; 0x40],
    pub sprites: Sprites,
    pub irq_flags: u8,
    pub irq_mask: u8,
    yscroll: i32,
    pub raster_y: i32,
    pub line_cycle: i32,
    raster_clk: i64,
    bad_lines_enabled: bool,
    bad_line: bool,
    raster_cond: bool,
    vblanking: bool,
}

impl Vic {
    pub fn new(ntsc: bool, lines: i32, cycles_per_line: i32) -> Self {
        Vic {
            ntsc,
            max_rasters: lines,
            cycles_per_line,
            regs: [0; 0x40],
            sprites: Sprites::default(),
            irq_flags: 0,
            irq_mask: 0,
            yscroll: 0,
            raster_y: 0,
            line_cycle: 0,
            raster_clk: 0,
            bad_lines_enabled: false,
            bad_line: false,
            raster_cond: false,
            vblanking: false,
        }
    }

    /// Power-on state, the beam on the last line, its first event due now.
    pub fn reset(&mut self, s: &mut Sched) {
        self.irq_flags = 0;
        self.irq_mask = 0;
        self.yscroll = 0;
        self.raster_y = self.max_rasters - 1;
        self.line_cycle = 0;
        self.raster_clk = 0;
        self.bad_lines_enabled = false;
        self.bad_line = false;
        self.raster_cond = false;
        self.vblanking = false;
        self.regs = [0; 0x40];
        self.sprites.reset();
        s.cancel(E_VIC);
        s.schedule(E_VIC, 0, PHI1);
    }

    pub fn fire(&mut self, ev: u8, s: &mut Sched, ln: &mut Lines) {
        match ev {
            E_VIC => self.event(s, ln),
            E_BADLINE => ln.set_ba(!self.bad_line),
            _ => self.edge(s, ln),
        }
    }

    fn compare(&self) -> i32 {
        self.regs[0x12] as i32 | (self.regs[0x11] as i32 & 0x80) << 1
    }

    fn edge(&mut self, s: &mut Sched, ln: &mut Lines) {
        let old = self.raster_cond;
        self.raster_cond = self.raster_y == self.compare();
        if self.raster_cond && !old {
            self.irq_flags |= IRQ_RASTER;
            self.irq_state(s, ln);
        }
    }

    fn irq_state(&mut self, s: &mut Sched, ln: &mut Lines) {
        if self.irq_flags & self.irq_mask & 0x0F != 0 {
            if self.irq_flags & 0x80 == 0 {
                ln.irq(s, true, VIDEO);
                self.irq_flags |= 0x80;
            }
        } else if self.irq_flags & 0x80 != 0 {
            ln.irq(s, false, VIDEO);
            self.irq_flags &= 0x7F;
        }
    }

    fn den(&self) -> bool {
        self.regs[0x11] & 0x10 != 0
    }

    fn is_bad_line(&self) -> bool {
        let y = self.raster_y;
        self.bad_lines_enabled
            && (FIRST_DMA_LINE..=LAST_DMA_LINE).contains(&y)
            && (y & 7) == self.yscroll
    }

    fn check_vblank(&mut self, s: &mut Sched, ln: &mut Lines) {
        if self.raster_y == self.max_rasters - 1 {
            self.vblanking = true;
        }
        if self.raster_y == FIRST_DMA_LINE && !self.bad_lines_enabled && self.den() {
            self.bad_lines_enabled = true;
        }
        if self.raster_y == LAST_DMA_LINE {
            self.bad_lines_enabled = false;
        }
        self.bad_line = false;
        if !self.vblanking {
            self.raster_y += 1;
            self.edge(s, ln);
            if self.raster_y == FIRST_DMA_LINE && !self.bad_lines_enabled {
                self.bad_lines_enabled = self.den();
            }
        }
        if self.is_bad_line() {
            self.bad_line = true;
        }
    }

    fn vblank(&mut self, s: &mut Sched, ln: &mut Lines) {
        if self.vblanking {
            self.vblanking = false;
            self.raster_y = 0;
            self.edge(s, ln);
        }
    }

    fn start_dma(&self, n: i32, ln: &mut Lines) {
        if n == 0 {
            ln.set_ba(self.sprites.dma & 1 == 0);
        } else if self.sprites.dma >> n & 1 != 0 {
            ln.set_ba(false);
        }
    }

    fn end_dma(&self, n: i32, ln: &mut Lines) {
        if n == 7 || (self.sprites.dma as u32) & (6u32 << n) == 0 {
            ln.set_ba(true);
        }
    }

    fn event(&mut self, s: &mut Sched, ln: &mut Lines) {
        let phase = s.now & 1;
        let cycles = s.time(phase) - self.raster_clk;
        let delay = if cycles != 0 {
            self.raster_clk += cycles;
            self.line_cycle =
                ((self.line_cycle as i64 + cycles) % self.cycles_per_line as i64) as i32;
            if self.ntsc {
                self.clock_ntsc(s, ln)
            } else {
                self.clock_pal(s, ln)
            }
        } else {
            1
        };
        s.schedule(E_VIC, delay - phase, PHI1);
    }

    /// Advance the beam to the current cycle (a CPU access).
    fn sync(&mut self, s: &mut Sched, ln: &mut Lines) {
        s.cancel(E_VIC);
        self.event(s, ln);
    }

    /// Line cycles 11-15, shared by PAL and NTSC.
    fn clock_common(&mut self, lc: i32, ln: &mut Lines) -> i64 {
        match lc {
            11 => {
                if self.bad_line {
                    ln.set_ba(false);
                }
                3
            }
            12 => 2,
            13 => 1,
            14 => {
                self.sprites.update_mc();
                1
            }
            _ => {
                self.sprites.update_mc_base();
                39
            }
        }
    }

    fn clock_pal(&mut self, s: &mut Sched, ln: &mut Lines) -> i64 {
        let lc = self.line_cycle;
        match lc {
            0 => {
                self.check_vblank(s, ln);
                self.end_dma(2, ln);
            }
            1 => {
                self.vblank(s, ln);
                self.start_dma(5, ln);
                if self.sprites.dma & 0xF8 == 0 {
                    return 10;
                }
            }
            2 | 4 => self.end_dma(lc / 2 + 2, ln),
            3 | 5 => self.start_dma(lc / 2 + 5, ln),
            6 => {
                self.end_dma(5, ln);
                return if self.sprites.dma & 0xC0 != 0 { 2 } else { 5 };
            }
            8 => {
                self.end_dma(6, ln);
                return 2;
            }
            10 => self.end_dma(7, ln),
            11..=15 => return self.clock_common(lc, ln),
            54 => {
                self.sprites.check_dma(&self.regs, self.raster_y);
                self.start_dma(0, ln);
            }
            55 => {
                self.sprites.check_dma(&self.regs, self.raster_y);
                self.sprites.check_exp(&self.regs);
                self.start_dma(0, ln);
            }
            57 => {
                self.sprites.check_display();
                if self.sprites.dma & 0x1F == 0 {
                    return 6;
                }
            }
            56 | 58 | 60 | 62 => self.start_dma((lc - 54) >> 1, ln),
            59 | 61 => self.end_dma((lc - 59) >> 1, ln),
            7 | 9 => {}
            _ => return (54 - lc) as i64,
        }
        1
    }

    fn clock_ntsc(&mut self, s: &mut Sched, ln: &mut Lines) -> i64 {
        let lc = self.line_cycle;
        match lc {
            0 => {
                self.check_vblank(s, ln);
                self.start_dma(5, ln);
            }
            1 => {
                self.vblank(s, ln);
                self.end_dma(3, ln);
                if self.sprites.dma & 0xF0 == 0 {
                    return 10;
                }
            }
            2 | 4 => self.start_dma(lc / 2 + 5, ln),
            3 => self.end_dma(4, ln),
            5 => {
                self.end_dma(5, ln);
                return if self.sprites.dma & 0xC0 != 0 { 2 } else { 6 };
            }
            7 => {
                self.end_dma(6, ln);
                return 2;
            }
            9 => {
                self.end_dma(7, ln);
                return 2;
            }
            11..=15 => return self.clock_common(lc, ln),
            54 => ln.set_ba(true),
            55 | 56 => {
                self.sprites.check_dma(&self.regs, self.raster_y);
                if lc == 55 {
                    self.sprites.check_exp(&self.regs);
                }
                self.start_dma(0, ln);
            }
            58 => {
                self.sprites.check_display();
                if self.sprites.dma & 0x1F == 0 {
                    return 7;
                }
            }
            57 | 59 | 61 | 63 => self.start_dma((lc - 55) >> 1, ln),
            60 | 62 | 64 => self.end_dma((lc - 60) >> 1, ln),
            6 | 8 | 10 => {}
            _ => return (54 - lc) as i64,
        }
        1
    }

    /// A CPU read of register `addr` (mirrored every 64 bytes).
    pub fn read(&mut self, addr: u16, s: &mut Sched, ln: &mut Lines) -> u8 {
        let addr = (addr & 0x3F) as usize;
        self.sync(s, ln);
        match addr {
            0x11 => (self.regs[0x11] & 0x7F) | ((self.raster_y & 0x100) >> 1) as u8,
            0x12 => self.raster_y as u8,
            0x19 => self.irq_flags | 0x70,
            0x1A => self.irq_mask | 0xF0,
            0x13 | 0x14 => 0,
            0..=0x1F => self.regs[addr],
            0x20..=0x2E => self.regs[addr] | 0xF0,
            _ => 0xFF,
        }
    }

    /// A CPU write of `data` to register `addr`.
    pub fn write(&mut self, addr: u16, data: u8, s: &mut Sched, ln: &mut Lines) {
        let addr = (addr & 0x3F) as usize;
        self.regs[addr] = data;
        self.sync(s, ln);
        if addr == 0x11 {
            self.write_control(data, s);
        }
        match addr {
            0x11 | 0x12 => s.schedule(E_EDGE, 0, PHI1),
            0x17 => self.sprites.line_crunch(data, self.line_cycle),
            0x19 => {
                self.irq_flags &= (!data & 0x0F) | 0x80;
                self.irq_state(s, ln);
            }
            0x1A => {
                self.irq_mask = data & 0x0F;
                self.irq_state(s, ln);
            }
            _ => {}
        }
    }

    fn write_control(&mut self, data: u8, s: &mut Sched) {
        let old_yscroll = self.yscroll;
        self.yscroll = (data & 7) as i32;
        let was_enabled = self.bad_lines_enabled;
        let y = self.raster_y;
        if y == FIRST_DMA_LINE && self.line_cycle == 0 {
            self.bad_lines_enabled = self.den();
        }
        let old_y = (if y > 0 { y } else { self.max_rasters }) - 1;
        if old_y == FIRST_DMA_LINE && self.den() {
            self.bad_lines_enabled = true;
        }
        let changed = old_yscroll != self.yscroll || self.bad_lines_enabled != was_enabled;
        if !changed || !(FIRST_DMA_LINE..=LAST_DMA_LINE).contains(&y) {
            return;
        }
        let was_bad = was_enabled && old_yscroll == (y & 7);
        let now_bad = self.bad_lines_enabled && self.yscroll == (y & 7);
        if now_bad == was_bad {
            return;
        }
        let old_bad = self.bad_line;
        if was_bad {
            if self.line_cycle < FETCH_CYCLE {
                self.bad_line = false;
            }
        } else if self.line_cycle <= FETCH_CYCLE + TEXT_COLS + 6 {
            self.bad_line = true;
        }
        if self.bad_line != old_bad {
            s.schedule(E_BADLINE, 0, PHI1);
        }
    }
}
