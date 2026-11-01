"""The native machine's self-modifying-code recorder (:mod:`vagrant_moses._core` only).

Armed with a lift-time image, the machine keeps each byte's last writer and turns every fetch or
control read that departs from the image into a deduplicated :class:`Event`, with no Python call
per instruction; ``docs/native.md`` defines the writers, roles, timings and mappings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, NamedTuple, Optional, Tuple

from . import machine as _machine

__all__ = ["AVAILABLE", "Event", "Recorder", "opcode_table"]

AVAILABLE = _machine.NATIVE and hasattr(_machine._core.Machine, "smc_arm")
_PTR = {"ind": 1, "indy": 2, "indx": 3}
_RET = {"RTS": 1, "RTI": 2}


class Event(NamedTuple):
    """A distinct event: who wrote the byte, how it was read, when and through which map."""

    writer: str
    writer_op: Optional[int]
    role: str
    timing: str
    mapping: str
    first: int


@lru_cache(maxsize=None)
def opcode_table() -> bytes:
    """Per opcode: length (bits 0-1), pointer form (bits 2-3: ``ind``, ``indy``, ``indx``)
    and return pull (bits 4-5: ``RTS``, ``RTI``), from jennings' opcode table."""
    from jennings.opcodes import (  # pylint: disable=import-outside-toplevel
        MODE_LEN,
        OPCODES,
    )

    out = bytearray(256)
    for op in range(256):
        mn, mode = OPCODES[op]
        out[op] = MODE_LEN[mode] | _PTR.get(mode, 0) << 2 | _RET.get(mn, 0) << 4
    return bytes(out)


class Recorder:
    """The recorder of the native ``machine``, armed over ``image`` (``banked``: classify RAM
    written under ROM or the I/O window by its mapping)."""

    def __init__(self, machine, image, banked: bool = True):
        if not AVAILABLE:
            raise ImportError("the SMC recorder needs vagrant_moses._core with smc_arm")
        self.m = machine
        machine.smc_arm(bytes(image), opcode_table(), banked)

    def begin(self) -> None:
        """End the pre-lift phase: call 0 starts under the bank bits now."""
        self.m.smc_begin()

    def next_call(self) -> None:
        self.m.smc_next_call()

    def frame_clock(self, frame_cycles: int) -> None:
        """Index events by ``(cycle - t0) // frame_cycles`` from the next decode (at ``t0``)."""
        self.m.smc_frame_clock(frame_cycles)

    @property
    def call(self) -> int:
        """The call index: -1 before :meth:`begin`."""
        return self.m.smc_call

    def host_write(self, addr: int, value: int) -> None:
        """``machine[addr] = value`` that no instruction owns (a call frame, the port)."""
        self.m.smc_host_write(addr, value)

    @property
    def deadline(self) -> Optional[int]:
        """The first decode after this cycle stops the run and sets :attr:`timed_out`."""
        return self.m.smc_deadline

    @deadline.setter
    def deadline(self, cycle: Optional[int]) -> None:
        self.m.smc_deadline = cycle

    @property
    def timed_out(self) -> bool:
        return self.m.smc_timed_out

    def finish(self) -> None:
        """Attribute the last instruction's writes once the run is over."""
        self.m.smc_finish()

    def events(self) -> List[Event]:
        """The distinct events, in the order first seen."""
        return [Event(*e) for e in self.m.smc_events()]

    def unclassified(self) -> Tuple[int, list]:
        """``(count, sample)``: bytes that changed with no writer (``unowned``, with the role
        and pc of the read) or writes no decode accounts for (``unattributed``)."""
        return self.m.smc_unclassified()

    def disarm(self) -> None:
        self.m.smc_disarm()
