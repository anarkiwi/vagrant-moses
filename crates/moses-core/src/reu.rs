//! The Commodore 17xx REU (`vagrant_moses.reu.Reu`): registers, RAM and the DMA engine.

pub const SIZES: [u32; 8] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];
pub const TO_REU: u8 = 0;
pub const FROM_REU: u8 = 1;
pub const SWAP: u8 = 2;
pub const VERIFY: u8 = 3;

const IRQ_PENDING: u8 = 0x80;
const END_OF_BLOCK: u8 = 0x40;
const VERIFY_ERROR: u8 = 0x20;
const CHIPS_256K: u8 = 0x10;
const EXECUTE: u8 = 0x80;
const AUTOLOAD: u8 = 0x20;
const FF00_DISABLED: u8 = 0x10;
const INT_ENABLE: u8 = 0x80;
const INT_EOB: u8 = 0x40;
const INT_VERIFY: u8 = 0x20;
const FIX_C64: u8 = 0x80;
const FIX_REU: u8 = 0x40;
pub const REGS: u8 = 0x0B;

/// The C64 side of a transfer, as `$01` maps it.
pub trait Port {
    fn dma_read(&mut self, addr: u16) -> u8;
    fn dma_write(&mut self, addr: u16, value: u8);
}

/// `(wrap, dram_wrap, store_mask, bank_unused, status)` of an REU of `size_kb` KiB.
pub fn geometry(size_kb: u32) -> Option<(u32, u32, u32, u8, u8)> {
    if !SIZES.contains(&size_kb) {
        return None;
    }
    Some(match size_kb {
        128 => (0x20000, 0x20000, 0x7FFFF, 0xF8, 0),
        256 | 512 => (0x80000, 0x80000, 0x7FFFF, 0xF8, CHIPS_256K),
        _ => {
            let size = size_kb << 10;
            (0x80000, size, size - 1, 0, CHIPS_256K)
        }
    })
}

fn set_byte(word: u16, high: bool, value: u8) -> u16 {
    if high {
        word & 0x00FF | (value as u16) << 8
    } else {
        word & 0xFF00 | value as u16
    }
}

fn all(reg: u8, bit: u8) -> bool {
    reg & (INT_ENABLE | bit) == INT_ENABLE | bit
}

pub struct Reu {
    pub size_kb: u32,
    pub ram: Vec<u8>,
    wrap: u32,
    dram_wrap: u32,
    store_mask: u32,
    bank_unused: u8,
    preset: u8,
    pub latch: u8,
    pub armed: bool,
    pub busy: bool,
    /// The IRQ line as the REU drives it.
    pub line: bool,
    /// The IRQ line as last signalled to the machine.
    pub signalled: bool,
    status: u8,
    command: u8,
    c64: u16,
    c64_shadow: u16,
    base: u16,
    base_shadow: u16,
    bank: u8,
    bank_shadow: u8,
    length: u16,
    length_shadow: u16,
    int_mask: u8,
    control: u8,
    kind: u8,
    left: u32,
    host: u16,
    raddr: u32,
    hstep: u16,
    rstep: u32,
    phase: bool,
    after_write: bool,
    delay: u8,
    last: bool,
    over: bool,
    tail: bool,
    phantom: u8,
    mask: u8,
    value: u8,
    rv: u8,
}

impl Reu {
    pub fn new(size_kb: u32) -> Option<Reu> {
        let (wrap, dram_wrap, store_mask, bank_unused, preset) = geometry(size_kb)?;
        let mut r = Reu {
            size_kb,
            ram: vec![0; (size_kb as usize) << 10],
            wrap,
            dram_wrap,
            store_mask,
            bank_unused,
            preset,
            latch: 0xFF,
            armed: false,
            busy: false,
            line: false,
            signalled: false,
            status: 0,
            command: 0,
            c64: 0,
            c64_shadow: 0,
            base: 0,
            base_shadow: 0,
            bank: 0,
            bank_shadow: 0,
            length: 0,
            length_shadow: 0,
            int_mask: 0,
            control: 0,
            kind: 0,
            left: 0,
            host: 0,
            raddr: 0,
            hstep: 0,
            rstep: 0,
            phase: false,
            after_write: false,
            delay: 0,
            last: false,
            over: false,
            tail: false,
            phantom: 0,
            mask: 0,
            value: 0,
            rv: 0,
        };
        r.reset();
        Some(r)
    }

    /// The registers as a reset leaves them (RAM and the bus latch survive).
    pub fn reset(&mut self) {
        self.status = self.preset;
        self.command = FF00_DISABLED;
        (self.c64, self.c64_shadow, self.base, self.base_shadow) = (0, 0, 0, 0);
        (self.bank, self.bank_shadow) = (self.bank_unused, self.bank_unused);
        (self.length, self.length_shadow) = (0xFFFF, 0xFFFF);
        self.int_mask = 0x1F;
        self.control = 0x3F;
        self.line = false;
    }

    /// Register `reg` without a read's side effects.
    pub fn peek(&self, reg: u8) -> u8 {
        match reg & 0x1F {
            0 => self.status,
            1 => self.command,
            2 => self.c64 as u8,
            3 => (self.c64 >> 8) as u8,
            4 => self.base as u8,
            5 => (self.base >> 8) as u8,
            6 => self.bank | self.bank_unused,
            7 => self.length as u8,
            8 => (self.length >> 8) as u8,
            9 => self.int_mask,
            10 => self.control,
            _ => 0xFF,
        }
    }

    /// A CPU read of `$DF00-$DFFF`; `None` while the DMA drives the bus.
    pub fn read(&mut self, addr: u16) -> Option<u8> {
        if self.busy {
            return None;
        }
        let reg = (addr & 0x1F) as u8;
        let mut value = self.peek(reg);
        if reg == 0 {
            self.status &= !(VERIFY_ERROR | END_OF_BLOCK | IRQ_PENDING);
            self.line = false;
        } else if reg == 6 {
            value |= 0xF8;
        }
        Some(value)
    }

    /// A CPU write of `$DF00-$DFFF`; true when it starts a DMA now.
    pub fn write(&mut self, addr: u16, value: u8) -> bool {
        let reg = (addr & 0x1F) as u8;
        if self.busy || reg >= REGS {
            return false;
        }
        match reg {
            1 => {
                self.command = value;
                if value & EXECUTE != 0 {
                    self.armed = value & FF00_DISABLED == 0;
                    return !self.armed;
                }
            }
            2 | 3 => {
                self.c64 = set_byte(self.c64_shadow, reg == 3, value);
                self.c64_shadow = self.c64;
            }
            4 | 5 => {
                self.base = set_byte(self.base_shadow, reg == 5, value);
                self.base_shadow = self.base;
            }
            6 => {
                self.bank = value & !self.bank_unused;
                self.bank_shadow = self.bank;
            }
            7 | 8 => {
                self.length = set_byte(self.length_shadow, reg == 8, value);
                self.length_shadow = self.length;
            }
            9 => {
                self.int_mask = value | 0x1F;
                self.check_irq(self.status);
            }
            _ => self.control = value | 0x3F,
        }
        false
    }

    fn check_irq(&mut self, bits: u8) {
        for (flag, enable) in [(END_OF_BLOCK, INT_EOB), (VERIFY_ERROR, INT_VERIFY)] {
            if bits & flag != 0 && all(self.int_mask, enable) {
                self.status |= IRQ_PENDING;
                self.line = true;
            }
        }
    }

    /// A write to `$FF00`: true when it starts the armed DMA.
    pub fn ff00(&mut self) -> bool {
        std::mem::take(&mut self.armed)
    }

    fn load(&self, addr: u32) -> u8 {
        let a = (addr & (self.dram_wrap - 1)) as usize;
        self.ram.get(a).copied().unwrap_or(self.latch)
    }

    fn store(&mut self, addr: u32, value: u8) {
        let a = (addr & (self.dram_wrap - 1)) as usize;
        if let Some(b) = self.ram.get_mut(a) {
            *b = value;
        }
    }

    fn advance(&mut self) {
        self.host = self.host.wrapping_add(self.hstep);
        let nxt = (self.raddr & 0x7FFFF) + self.rstep;
        self.raddr = self.raddr & 0xF80000 | if nxt == self.wrap { 0 } else { nxt };
        self.left -= 1;
    }

    fn begin(&mut self) {
        self.busy = true;
        self.kind = self.command & 3;
        self.host = self.c64;
        self.raddr = self.base as u32 | (self.bank as u32) << 16;
        self.left = if self.length == 0 {
            0x10000
        } else {
            self.length as u32
        };
        self.hstep = (self.control & FIX_C64 == 0) as u16;
        self.rstep = (self.control & FIX_REU == 0) as u32;
        (self.phase, self.delay, self.phantom, self.mask) = (false, 0, 0, 0);
        (self.after_write, self.last, self.over, self.tail) = (false, false, false, false);
    }

    /// One DMA cycle with the VIC's BA `ba`; true when the CPU runs this cycle.
    pub fn step<P: Port>(&mut self, ba: bool, port: &mut P) -> bool {
        if !self.busy {
            self.begin();
        }
        if self.after_write {
            if ba {
                self.delay += 1;
                if self.delay > 1 {
                    (self.delay, self.last, self.after_write) = (0, true, false);
                    return false;
                }
            } else {
                self.delay = 0;
            }
            self.last = false;
        } else if ba {
            return false;
        }
        self.after_write = false;
        self.slot(port)
    }

    fn slot<P: Port>(&mut self, port: &mut P) -> bool {
        if self.phantom != 0 {
            self.phantom -= 1;
            return false;
        }
        if self.left == 0 || self.over {
            if self.last && !self.tail && (self.kind == FROM_REU || self.kind == SWAP) {
                self.tail = true;
                return false;
            }
            self.finish(port);
            return true;
        }
        match self.kind {
            TO_REU => {
                self.value = port.dma_read(self.host);
                self.store(self.raddr, self.value);
                self.advance();
            }
            FROM_REU => {
                let value = self.load(self.raddr);
                self.latch = value;
                port.dma_write(self.host, value);
                self.advance();
                self.after_write = true;
            }
            SWAP if !self.phase => {
                self.rv = self.load(self.raddr);
                self.value = port.dma_read(self.host);
                self.phase = true;
            }
            SWAP => {
                self.store(self.raddr, self.value);
                port.dma_write(self.host, self.rv);
                self.advance();
                self.phase = false;
                self.after_write = true;
            }
            _ => {
                let differ = self.load(self.raddr) != port.dma_read(self.host);
                self.advance();
                if differ {
                    (self.mask, self.over) = (VERIFY_ERROR, true);
                    self.phantom = (self.left != 0) as u8;
                }
            }
        }
        false
    }

    fn finish<P: Port>(&mut self, port: &mut P) {
        let mut left = self.left;
        if self.kind != VERIFY || left == 0 {
            self.mask |= END_OF_BLOCK;
            left = 1;
        } else if left == 1 && self.load(self.raddr) == port.dma_read(self.host) {
            self.mask |= END_OF_BLOCK;
        }
        let raddr = self.raddr & self.store_mask;
        self.status |= self.mask;
        if self.command & AUTOLOAD != 0 {
            (self.c64, self.base) = (self.c64_shadow, self.base_shadow);
            (self.bank, self.length) = (self.bank_shadow, self.length_shadow);
        } else {
            if self.control & FIX_C64 == 0 {
                self.c64 = self.host;
            }
            if self.control & FIX_REU == 0 {
                self.base = raddr as u16;
                self.bank = (raddr >> 16) as u8;
            }
            self.length = left as u16;
        }
        self.check_irq(self.mask);
        if self.kind == TO_REU {
            self.latch = self.value;
        } else if self.kind == FROM_REU {
            self.latch = self.load(self.raddr);
        }
        self.busy = false;
        self.command = self.command & !EXECUTE | FF00_DISABLED;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flat([u8; 0x10000]);

    impl Port for Flat {
        fn dma_read(&mut self, addr: u16) -> u8 {
            self.0[addr as usize]
        }
        fn dma_write(&mut self, addr: u16, value: u8) {
            self.0[addr as usize] = value;
        }
    }

    fn program(r: &mut Reu, regs: &[(u16, u8)]) -> bool {
        regs.iter()
            .fold(false, |go, &(a, v)| r.write(0xDF00 | a, v) || go)
    }

    #[test]
    fn geometry_per_size() {
        assert!(geometry(64).is_none());
        assert_eq!(geometry(128).unwrap().0, 0x20000);
        assert_eq!(geometry(2048).unwrap().1, 0x200000);
        assert_eq!(Reu::new(128).unwrap().peek(0), 0);
        assert_eq!(Reu::new(256).unwrap().peek(0), CHIPS_256K);
    }

    #[test]
    fn registers_and_status_read() {
        let mut r = Reu::new(512).unwrap();
        assert_eq!(r.read(0xDF06), Some(0xF8));
        assert_eq!(r.read(0xDF29), Some(0x1F));
        assert_eq!(r.read(0xDF0B), Some(0xFF));
        r.write(0xDF02, 0x34);
        r.write(0xDF03, 0x12);
        assert_eq!((r.peek(2), r.peek(0x23)), (0x34, 0x12));
        r.status |= END_OF_BLOCK;
        r.write(0xDF09, 0xC0);
        assert!(r.line);
        assert_eq!(r.read(0xDF00), Some(0xD0));
        assert!(!r.line);
        assert_eq!(r.read(0xDF00), Some(CHIPS_256K));
    }

    #[test]
    fn to_reu_and_ff00_trigger() {
        let mut r = Reu::new(128).unwrap();
        let mut c64 = Flat([0; 0x10000]);
        c64.0[0x1000..0x1004].copy_from_slice(&[1, 2, 3, 4]);
        let regs = [
            (2, 0),
            (3, 0x10),
            (4, 0xFE),
            (5, 0xFF),
            (6, 1),
            (7, 4),
            (8, 0),
        ];
        assert!(!program(&mut r, &regs));
        assert!(!r.write(0xDF01, 0x80));
        assert!(r.armed && r.ff00());
        let cycles = (0..).position(|_| r.step(false, &mut c64)).unwrap();
        assert_eq!(cycles, 4);
        assert_eq!((&r.ram[0x1FFFE..], &r.ram[..2]), (&[1, 2][..], &[3, 4][..]));
        assert_eq!(r.peek(0) & END_OF_BLOCK, END_OF_BLOCK);
        assert_eq!(
            (r.peek(4), r.peek(5), r.peek(6) & 7, r.peek(7)),
            (2, 0, 0, 1)
        );
    }
}
