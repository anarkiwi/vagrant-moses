"""The native machine's self-modifying-code recorder (:mod:`vagrant_moses._core` only).

Armed with a lift-time image, the machine keeps each byte's last writer and turns every fetch or
control read that departs from the image into a deduplicated :class:`Event`, with no Python call
per instruction; ``docs/native.md`` defines the writers, roles, timings and mappings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, NamedTuple, Optional, Tuple

from . import machine as _machine

__all__ = [
    "AVAILABLE",
    "TRACE",
    "Event",
    "Facts",
    "Recorder",
    "Scan",
    "Tracer",
    "opcode_table",
]

AVAILABLE = _machine.NATIVE and hasattr(_machine._core.Machine, "smc_arm")
TRACE = AVAILABLE and hasattr(_machine._core.Machine, "smc_trace")
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


class Facts(NamedTuple):
    """A trace's lift facts, ascending: instruction starts, ``(pc, next)`` edges, the values
    stored per address, ``(pc, lo, hi)`` page-1 data ranges, and stack heights (``sp`` less the
    call's entry ``sp``) and bank bits per start."""

    starts: List[int]
    edges: List[Tuple[int, int]]
    writes: List[Tuple[int, List[int]]]
    page1: List[Tuple[int, int, int]]
    heights: List[Tuple[int, List[int]]]
    banks: List[Tuple[int, List[int]]]


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


class Tracer:
    """The armed ``recorder``'s trace: lift facts of the call phase's instructions and one
    record per phase. ``table`` holds 256 little-endian ``u16`` per opcode (``docs/native.md``);
    calls return to ``sentinel``; the SID window ``sid`` (``(lo, hi)``) is answered natively,
    registers from ``image``."""

    NONE, INIT, CALL = range(3)

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, recorder, table, sentinel, sid, image):
        if not TRACE:
            raise ImportError("the trace needs vagrant_moses._core with smc_trace")
        self.m = recorder.m
        self.m.smc_trace(bytes(table), sentinel, sid[0], sid[1], bytes(image))

    def phase(self, phase: int, sp0: int = 0) -> None:
        """Enter ``phase`` (:attr:`NONE`, :attr:`INIT`, :attr:`CALL`), a call entered at ``sp0``."""
        self.m.smc_trace_phase(phase, sp0)

    def snap(self) -> Tuple[list, bytes, bytes]:
        """Close the phase: ``(log, idx, vals)``, its ``(addr, value, write)`` SID accesses in
        order and the RAM bytes changed since the previous snap (``idx`` little-endian
        ``u16``)."""
        return self.m.smc_trace_snap()

    def facts(self) -> Facts:
        starts, edges, writes, page1, heights, banks = self.m.smc_trace_facts()
        return Facts(
            starts,
            edges,
            [(a, list(v)) for a, v in writes],
            page1,
            heights,
            [(p, list(v)) for p, v in banks],
        )


class Scan:
    """The armed ``recorder``'s scan of an interrupt-driven phase's main context. ``table``
    holds 256 little-endian ``u16`` per opcode: the address form (as :class:`Tracer`'s), bit 4
    a read and bit 5 a CPU store of it; the ``sentinel`` is not counted."""

    def __init__(self, recorder, table, sentinel):
        if not TRACE:
            raise ImportError("the scan needs vagrant_moses._core with smc_scan")
        self.m = recorder.m
        self.m.smc_scan(bytes(table), sentinel)

    def start(self, frame_cycles: int) -> None:
        """Start at the next decode, frames of ``frame_cycles`` cycles from it."""
        self.m.smc_scan_start(frame_cycles)

    def result(self) -> Tuple[Optional[Tuple[int, int, int]], int, Optional[int]]:
        """``(vector, insns, busy)``: ``(port bits, RAM $0314 word, RAM $FFFE word)`` at the
        first decode (None before it), the main context's instructions until the first that
        stores outside page 1 or touches I/O, and that one's frame (None: none did)."""
        return self.m.smc_scan_result()
