"""vagrant_moses -- a standalone C64 machine model.

The memory map, the $01-banked bus and ROMs, the VIC-II and CIAs over a two-phase
scheduler, a cycle-exact machine, KERNAL traps, the IRQ dispatch helpers and native
release-media extraction. The SID audio model and all SID-file knowledge are
deliberately not here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import (
    bus,
    c64,
    cia,
    cpu,
    eventsched,
    kernal,
    machine,
    media,
    memmap,
    reu,
    roms,
    smc,
)
from . import vicii
from .bus import Bus
from .c64 import (
    IRQ_VEC,
    NMI_VEC,
    HW_IRQ_VEC,
    HW_NMI_VEC,
    IRQ_DISPATCH,
    IRQ_RETURN,
    KERNAL_IRQ,
    poweron_ram,
    read_vector,
    installed_vector,
    installed_handler,
    install_kernal_irq_stubs,
    irq_stubs,
    install_irq_entry,
    dispatch_vector,
)
from .cia import Cia, InterruptSource, Timer
from .media import MediaError, MediaFile, iter_prgs, read_disk, read_t64
from .eventsched import Event, PHI1, PHI2, Scheduler
from .kernal import KernalHost, KernalHostError, sys_addr
from .machine import Machine, NativeSid, boot_cycle
from .memmap import NTSC, PAL, Video
from .reu import Reu
from .roms import RomFetchError, kernal_stub
from .vicii import NTSC_GEOMETRY, PAL_GEOMETRY, Sprites, Vic

try:
    __version__ = version("vagrant-moses")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "bus",
    "c64",
    "cia",
    "cpu",
    "eventsched",
    "kernal",
    "machine",
    "memmap",
    "reu",
    "Reu",
    "roms",
    "vicii",
    "media",
    "MediaError",
    "MediaFile",
    "iter_prgs",
    "read_disk",
    "read_t64",
    "Bus",
    "dispatch_vector",
    "KernalHost",
    "KernalHostError",
    "sys_addr",
    "Machine",
    "NativeSid",
    "boot_cycle",
    "PAL",
    "NTSC",
    "Video",
    "RomFetchError",
    "kernal_stub",
    "IRQ_VEC",
    "NMI_VEC",
    "HW_IRQ_VEC",
    "HW_NMI_VEC",
    "IRQ_DISPATCH",
    "IRQ_RETURN",
    "KERNAL_IRQ",
    "poweron_ram",
    "read_vector",
    "installed_vector",
    "installed_handler",
    "install_kernal_irq_stubs",
    "irq_stubs",
    "install_irq_entry",
    "Cia",
    "Timer",
    "InterruptSource",
    "Event",
    "Scheduler",
    "PHI1",
    "PHI2",
    "Vic",
    "Sprites",
    "PAL_GEOMETRY",
    "NTSC_GEOMETRY",
    "__version__",
]
