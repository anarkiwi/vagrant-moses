"""The C64 memory map: chip register windows, CPU port, vectors and video timing.

Documented hardware facts only. The SID appears here as an I/O window (where its
registers decode); what those registers mean is the SID host's concern.
"""

from __future__ import annotations

from typing import NamedTuple

# --- 6510 CPU port ----------------------------------------------------------
CPU_DDR, CPU_PORT = 0x0000, 0x0001
DDR_DEFAULT, BANK_DEFAULT = 0x2F, 0x37  # BASIC + KERNAL + I/O in, CHARGEN out
LORAM, HIRAM, CHAREN = 0x01, 0x02, 0x04

# --- ROM and I/O windows ----------------------------------------------------
BASIC_BASE, CHARGEN_BASE, KERNAL_BASE = 0xA000, 0xD000, 0xE000
IO_BASE, IO_LAST = 0xD000, 0xDFFF
VIC_BASE, VIC_LAST = 0xD000, 0xD02E  # mirrored every $40 to $D3FF
SID_BASE, SID_LAST, SID_STRIDE = 0xD400, 0xD7FF, 0x20  # mirrored every $20
COLOUR_RAM, COLOUR_LAST = 0xD800, 0xDBFF  # 4-bit nybbles
CIA1_BASE, CIA1_LAST = 0xDC00, 0xDC0F  # mirrored every $10 to $DCFF
CIA2_BASE, CIA2_LAST = 0xDD00, 0xDD0F
IO1_BASE, IO2_BASE = 0xDE00, 0xDF00  # expansion-port I/O pages
SCREEN_RAM = 0x0400  # KERNAL default text screen

# --- VIC-II registers ---------------------------------------------------------
VIC_CONTROL_1 = 0xD011  # bit 7: raster bit 8 (compare on write, beam on read)
VIC_RASTER = 0xD012
VIC_IRQ_STATUS = 0xD019
VIC_IRQ_ENABLE = 0xD01A

# --- CIA registers: offsets, then the absolute addresses of both chips --------
TA_LO, TA_HI, TB_LO, TB_HI, ICR, CRA, CRB = 4, 5, 6, 7, 0xD, 0xE, 0xF
CIA1_TIMER_A_LO, CIA1_TIMER_A_HI = CIA1_BASE + TA_LO, CIA1_BASE + TA_HI
CIA1_TIMER_B_LO, CIA1_TIMER_B_HI = CIA1_BASE + TB_LO, CIA1_BASE + TB_HI
CIA1_ICR, CIA1_CRA, CIA1_CRB = CIA1_BASE + ICR, CIA1_BASE + CRA, CIA1_BASE + CRB
CIA2_TIMER_A_LO, CIA2_TIMER_A_HI = CIA2_BASE + TA_LO, CIA2_BASE + TA_HI
CIA2_TIMER_B_LO, CIA2_TIMER_B_HI = CIA2_BASE + TB_LO, CIA2_BASE + TB_HI
CIA2_ICR, CIA2_CRA, CIA2_CRB = CIA2_BASE + ICR, CIA2_BASE + CRA, CIA2_BASE + CRB

# --- interrupt vectors --------------------------------------------------------
RAM_IRQ_VECTOR_LO, RAM_IRQ_VECTOR_HI = 0x0314, 0x0315  # CINV
RAM_BRK_VECTOR_LO, RAM_BRK_VECTOR_HI = 0x0316, 0x0317  # CBINV
RAM_NMI_VECTOR_LO, RAM_NMI_VECTOR_HI = 0x0318, 0x0319  # NMINV
CPU_NMI_VECTOR, CPU_RESET_VECTOR, CPU_IRQ_VECTOR = 0xFFFA, 0xFFFC, 0xFFFE
VECTORS = (
    RAM_IRQ_VECTOR_LO,
    RAM_IRQ_VECTOR_HI,
    RAM_NMI_VECTOR_LO,
    RAM_NMI_VECTOR_HI,
    CPU_NMI_VECTOR,
    CPU_NMI_VECTOR + 1,
    CPU_IRQ_VECTOR,
    CPU_IRQ_VECTOR + 1,
)


class Video(NamedTuple):
    """One video standard's CPU timing.

    ``kernal_latch`` is the CIA #1 Timer A latch the KERNAL leaves for its
    60 Hz IRQ on this standard.
    """

    clock_hz: int
    lines: int
    cycles_per_line: int
    kernal_latch: int

    @property
    def cycles_per_frame(self) -> int:
        """CPU cycles in one video frame."""
        return self.lines * self.cycles_per_line

    @property
    def geometry(self):
        """``(lines, cycles_per_line)``, the :class:`~vagrant_moses.vicii.Vic` form."""
        return (self.lines, self.cycles_per_line)


PAL = Video(985248, 312, 63, 0x4025)
NTSC = Video(1022727, 263, 65, 0x4295)


def video(ntsc: bool) -> Video:
    """The :data:`NTSC` or :data:`PAL` timing."""
    return NTSC if ntsc else PAL


def raster_compare(d011: int, d012: int) -> int:
    """The 9-bit raster compare line written as ``$D011`` bit 7 and ``$D012``."""
    return (d011 & 0x80) << 1 | d012


def is_vic_reg(addr: int) -> bool:
    """``addr`` is a VIC-II register (unmirrored window)."""
    return VIC_BASE <= addr <= VIC_LAST


def is_cia1_reg(addr: int) -> bool:
    """``addr`` is a CIA #1 register (unmirrored window)."""
    return CIA1_BASE <= addr <= CIA1_LAST


def is_cia2_reg(addr: int) -> bool:
    """``addr`` is a CIA #2 register (unmirrored window)."""
    return CIA2_BASE <= addr <= CIA2_LAST


def is_cia_timer(addr: int) -> bool:
    """``addr`` is a timer latch/counter byte of either CIA."""
    return (is_cia1_reg(addr) or is_cia2_reg(addr)) and TA_LO <= addr & 0xF <= TB_HI


def is_irq_vector(addr: int) -> bool:
    """``addr`` is a byte of the RAM IRQ vector CINV."""
    return addr in (RAM_IRQ_VECTOR_LO, RAM_IRQ_VECTOR_HI)


def is_nmi_vector(addr: int) -> bool:
    """``addr`` is a byte of the RAM NMI vector NMINV."""
    return addr in (RAM_NMI_VECTOR_LO, RAM_NMI_VECTOR_HI)


def is_cpu_vector(addr: int) -> bool:
    """``addr`` is a byte of the 6510 vector table ``$FFFA-$FFFF``."""
    return CPU_NMI_VECTOR <= addr <= 0xFFFF
