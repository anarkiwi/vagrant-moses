"""vagrant_moses -- a standalone C64 machine model.

Power-on RAM, interrupt-vector discovery, and the ROM-free IRQ dispatch path a
host needs to enter a tune's playroutine faithfully. See :mod:`vagrant_moses.c64`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import c64
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

try:
    __version__ = version("vagrant-moses")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "c64",
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
    "__version__",
]
