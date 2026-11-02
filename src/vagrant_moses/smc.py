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
    "CALLS",
    "FRAMES",
    "Budget",
    "Calls",
    "Chunk",
    "ENTRY_FIELDS",
    "EVENT_FIELDS",
    "Frames",
    "Records",
    "SOURCES",
    "STOPS",
    "Event",
    "Facts",
    "Recorder",
    "Scan",
    "Tracer",
    "opcode_table",
]

AVAILABLE = _machine.NATIVE and hasattr(_machine._core.Machine, "smc_arm")
TRACE = AVAILABLE and hasattr(_machine._core.Machine, "smc_trace")
CALLS = AVAILABLE and hasattr(_machine._core.Machine, "smc_calls")
FRAMES = AVAILABLE and hasattr(_machine._core.Machine, "smc_frames")
#: The interrupt sources (an event's or entry's ``kind`` indexes it) and the reasons a run
#: stops (:attr:`Calls.outcome` and :attr:`Frames.outcome` index it).
SOURCES = getattr(_machine._core, "SOURCES", ()) if AVAILABLE else ()
STOPS = getattr(_machine._core, "STOPS", ()) if AVAILABLE else ()
#: ``(name, little-endian numpy type)`` of the fields of an event (21 bytes) and of a frame's
#: entry (18 bytes), as ``numpy.dtype`` takes them.
EVENT_FIELDS = [("record", "<u4"), ("icount", "<u4"), ("pos", "<u4"), ("at", "<u2")] + [
    (r, "u1") for r in ("a", "x", "y", "sp", "p", "depth", "kind")
]
ENTRY_FIELDS = (
    [("at", "<u2"), ("pc", "<u2")]
    + [(r, "u1") for r in ("a", "x", "y", "sp", "p", "kind")]
    + [("insns", "<u4"), ("cycles", "<u4")]
)
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
    calls return to ``sentinel``; the CPU and DMA I/O accesses at ``window`` (``(lo, hi)``)
    are logged as the machine answers them. ``facts`` false keeps only the records."""

    NONE, INIT, CALL = range(3)

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, recorder, table, sentinel, window, facts=True):
        if not TRACE:
            raise ImportError("the trace needs vagrant_moses._core with smc_trace")
        self.m = recorder.m
        self.m.smc_trace(bytes(table), sentinel, window[0], window[1], facts)

    def phase(self, phase: int, sp0: int = 0) -> None:
        """Enter ``phase`` (:attr:`NONE`, :attr:`INIT`, :attr:`CALL`), a call entered at ``sp0``."""
        self.m.smc_trace_phase(phase, sp0)

    def snap(self) -> Tuple[list, bytes, bytes]:
        """Close the phase: ``(log, idx, vals)``, its ``(addr, value, write)`` I/O accesses in
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


class Budget(NamedTuple):
    """The cycles and instructions a unit (a call or a frame) and a song (all its units) may
    run, 0 for no limit. A run that hits one stops, and :attr:`Calls.outcome` or
    :attr:`Frames.outcome` says which (:data:`STOPS`); the open unit then closes as halted
    (``cycles``)."""

    unit_cycles: int = 0
    unit_insns: int = 0
    song_cycles: int = 0
    song_insns: int = 0


class Records(NamedTuple):
    """A :class:`Calls` run's records, one per closed phase, as little-endian bytes: ``meta``
    7 per record (``phase, halt, a, x, y, sp, p``; halt 0 return, 1 cycles, 2 jam), ``log`` 4
    per logged I/O access (``addr`` u16, ``value``, ``write``) ending at the u64 ``log_end``,
    and the RAM changed since the previous record (the first: since ``origin``) as u16
    ``idx`` and ``vals`` ending at the u64 ``delta_end``. ``events`` holds the interrupts
    taken inside the records, :data:`EVENT_FIELDS` per event, ``record`` the index of the
    record in these."""

    origin: bytes
    meta: bytes
    log: bytes
    log_end: bytes
    idx: bytes
    vals: bytes
    delta_end: bytes
    events: bytes = b""


class Chunk(NamedTuple):
    """The frames :meth:`Frames.take` returns: their ``records``, the ``entries`` (one
    :data:`ENTRY_FIELDS` row per record) and the ``events`` (:data:`EVENT_FIELDS`)."""

    records: Records
    entries: bytes
    events: bytes


class Calls:
    """A queue of calls the armed ``recorder``'s machine runs from ``sentinel`` with no Python
    call per call: each pushes a return to ``sentinel - 1``, and the sentinel's trap closes the
    call's :class:`Tracer` record (when one is armed) and starts the next. ``queue`` holds
    ``(phase, pc, a)`` (phase :attr:`Tracer.INIT` or :attr:`Tracer.CALL`), or those packed as 4
    bytes per call (``phase, pc lo, pc hi, a``); ``ports`` maps
    ``"init"``/``"call"`` to the ``$01`` value stored before such a call when it differs;
    ``call_cycles`` (0: none) is each call's cycle limit (``budget.unit_cycles`` when that is
    set); ``times`` (default: once each) how many calls back to back each queued call makes.
    ``budget`` (:class:`Budget`) limits each call and the whole queue.

    An interrupt taken inside a call joins the call's record: its handler returns through the
    sentinel and the interrupt is an event of the record (:attr:`Records.events`)."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        recorder,
        queue,
        sentinel,
        ports=None,
        call_cycles=0,
        times=None,
        budget=None,
    ):
        if not CALLS:
            raise ImportError("the call queue needs vagrant_moses._core with smc_calls")
        self.m = recorder.m
        if not isinstance(queue, (bytes, bytearray, memoryview)):
            queue = bytes(
                b for ph, pc, a in queue for b in (ph, pc & 0xFF, pc >> 8, a & 0xFF)
            )
        ports = ports or {}
        self.m.smc_calls(
            queue,
            sentinel,
            ports.get("init"),
            ports.get("call"),
            call_cycles,
            None if times is None else list(times),
            None if budget is None else tuple(budget),
        )

    @property
    def left(self) -> int:
        """The queued calls not yet started."""
        return self.m.smc_calls_left

    @property
    def outcome(self) -> int:
        """Why the run stopped: an index of :data:`STOPS` (0: no limit was hit)."""
        return self.m.smc_outcome

    def close(self) -> None:
        """Close the open record once the run is over (halt ``jam`` or ``cycles``)."""
        self.m.smc_calls_close()

    def records(self) -> Records:
        return Records(*self.m.smc_calls_records(), self.m.smc_calls_events())

    def take(self) -> Records:
        """:meth:`records`, emptied: the next record's changes are against the ``origin`` of
        the next take, the RAM the last record taken left."""
        return Records(*self.m.smc_calls_take_all())

    def entries(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """``(entry_sp, entry_port)``: each call's ``(pc, sp)`` after its return frame and
        ``(pc, port dir, port data)``, ascending; interrupt handlers entered inside a call
        count as calls."""
        return self.m.smc_calls_entries()


class Frames:
    """The interrupt frames of the armed ``recorder``'s machine (which needs a :class:`Tracer`
    over ``sentinel``): each NMI or IRQ from its entry to its RTI is a record, recorded with no
    Python call. A handler's pushed return is pointed at ``sentinel``, whose trap resumes the
    interrupted code. An interrupt taken inside a frame is an event of its record, with the
    instruction count of the interrupted context at the boundary it was taken at; BRK is no
    frame. ``budget`` (:class:`Budget`) limits each frame and the run.

    Run the machine in windows, :meth:`take` the closed frames after each, and :meth:`close`
    at the end."""

    def __init__(self, recorder, sentinel, budget=None):
        if not FRAMES:
            raise ImportError("the frames need vagrant_moses._core with smc_frames")
        self.m = recorder.m
        self._tail = self._state = None
        self.m.smc_frames(sentinel, None if budget is None else tuple(budget))

    def take(self) -> Chunk:
        """The frames closed since the last take, emptied: the next frame's changes are against
        the RAM the last one left."""
        if self._state is not None:
            out, self._tail = self._tail or Chunk(Records(*[b""] * 7), b"", b""), None
            return out
        *rec, entries, events = self.m.smc_frames_take()
        return Chunk(Records(*rec), entries, events)

    def _get(self):
        if self._state is None:
            return self.m.smc_frames_state()
        return self._state

    @property
    def image(self) -> Optional[bytes]:
        """The RAM at the first entry (None before it)."""
        return self._get()[0]

    @property
    def dirty(self) -> bool:
        """RAM outside page 1 changed between two frames: the interrupted code does real work."""
        return self._get()[1]

    @property
    def foreign(self) -> int:
        """The kinds taken outside any frame, bit ``k`` for kind ``k`` (4: BRK)."""
        return self._get()[2]

    @property
    def open(self) -> bool:
        """Whether a frame is open: a run that ends now cuts it."""
        return self._get()[3]

    @property
    def closed(self) -> int:
        """The frames closed so far."""
        return self._get()[4]

    @property
    def outcome(self) -> int:
        """Why the run stopped: an index of :data:`STOPS` (0: no limit was hit)."""
        return self.m.smc_outcome

    def entries(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """``(entry_sp, entry_port)``: each handler entry's ``(pc, sp)`` after the interrupt
        frame it pushed and ``(pc, port dir, port data)``, ascending."""
        return self.m.smc_frames_entries()

    def close(self) -> None:
        """Stop recording once the run is over; the next :meth:`take` returns the last frames.
        A frame still open is closed as halted (``jam``, else ``cycles``) when the CPU jammed
        or a limit stopped the run, else dropped, as a run's end cuts it."""
        self._state = self.m.smc_frames_state()
        *rec, entries, events = self.m.smc_frames_close()
        self._tail = Chunk(Records(*rec), entries, events)
