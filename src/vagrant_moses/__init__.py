"""vagrant_moses -- a standalone C64 machine model.

VIC-II and CIA volatile I/O models, the two-phase event scheduler they share, and
the power-on RAM, interrupt-vector discovery, ROM-free IRQ dispatch path, and native
release-media extraction. The SID audio model and all SID-file knowledge are
deliberately not here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import c64, cia, eventsched, media, vicii
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
)
from .cia import Cia, InterruptSource, Timer
from .media import MediaError, MediaFile, iter_prgs, read_disk, read_t64
from .eventsched import Event, PHI1, PHI2, Scheduler
from .vicii import NTSC_GEOMETRY, PAL_GEOMETRY, Sprites, Vic

try:
    __version__ = version("vagrant-moses")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "c64",
    "cia",
    "eventsched",
    "vicii",
    "media",
    "MediaError",
    "MediaFile",
    "iter_prgs",
    "read_disk",
    "read_t64",
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
