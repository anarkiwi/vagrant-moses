"""The SID write order and hard restart a canonical document can leave to a player: what
reaches a voice's envelope, the event pattern of a frame's control, AD and SR writes, the
per-voice orders of the kept writes, the exact minimum cover of orders that keep every frame's
envelope, and the hard restart's settled state.

Input is a write stream as :meth:`NativeSid.drain_writes` gives it: absolute
``(cycle, reg, value)``, split by :func:`split_frames` into the init writes and per-frame
``(start, writes)``. The envelope is judged by pylibresidfp (the ``sidorder`` extra), imported
on first use. See docs/sidorder.md.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

REGS = 25
GATE, TEST = 0x01, 0x08
SAW = 0x20
RATE = 8000.0
ENV = (
    "envelope state next_state gate lfsr rate exponential_counter exponential_counter_period "
    "counter_enabled state_pipeline envelope_pipeline exponential_pipeline reset_lfsr "
    "attack decay sustain release"
).split()
PAL_CLOCK = 985248
WRAP = 0x7FFF
PICK = {
    "ctrl": (4, -1),
    "ctrl:2": (4, -2),
    "ad": (5, -1),
    "sr": (6, -1),
    "sr:2": (6, -2),
}
PAIRS = (("ctrl:2", "ctrl"), ("sr:2", "sr"))
ORDERS = [
    o
    for extra in ((), ("ctrl:2",), ("sr:2",), ("ctrl:2", "sr:2"))
    for o in permutations(("ad", "sr", "ctrl", *extra))
    if all(o.index(a) < o.index(b) for a, b in PAIRS if a in o)
]
CANONICAL = (
    ("sr:2", "sr", "ad", "ctrl:2", "ctrl"),
    ("ad", "sr:2", "sr", "ctrl:2", "ctrl"),
    ("ctrl:2", "ctrl", "ad", "sr:2", "sr"),
)


def events(start: tuple, writes: list) -> tuple:
    """A voice's event pattern from its ``start`` ``(ctrl, ad, sr)`` through ``writes``
    ``(register offset within the voice, value)`` in order: every write that changes what its
    envelope generator uses, or resets its oscillator, in order, as ``(offset, value)`` (a control
    write by its gate and test bits). With the gate on, AD and SR set the rate and sustain in
    effect at once; with it off only SR's release is in effect, so an AD write waits for the next
    gate-on and is placed just before it (or at the end), its last value only. Writes of the value
    already held, and timing, leave no trace."""
    held = dict(zip((4, 5, 6), start))
    out, latent = [], None
    for reg, value in writes:
        if reg == 4:
            edge = (held[4] ^ value) & (GATE | TEST)
            if edge & GATE and value & GATE and latent is not None:
                out.append((5, latent))
                latent = None
            if edge:
                out.append((4, value & (GATE | TEST)))
        elif held[reg] != value:
            if reg == 5 and not held[4] & GATE:
                latent = value
            else:
                out.append((reg, value))
        held[reg] = value
    return tuple(out + [(5, latent)] * (latent is not None))


def kept(writes: list, order: tuple) -> list:
    """The writes ``order`` keeps of a voice's ``writes`` ``(offset, value)``, in its order: the
    last control, AD and SR writes, and the control or SR write before the last where it names
    it (``ctrl:2``, ``sr:2``)."""
    return pick(lasts(writes), order)


def lasts(writes: list) -> dict:
    """The control, AD and SR values of ``writes`` ``(offset, value)`` by register offset."""
    return {reg: [v for r, v in writes if r == reg] for reg in (4, 5, 6)}


def pick(last: dict, order: tuple) -> list:
    """:func:`kept` from :func:`lasts`."""
    out = []
    for item in order:
        reg, k = PICK[item]
        if len(last[reg]) >= -k:
            out.append((reg, last[reg][k]))
    return out


@lru_cache(maxsize=1 << 16)
def keeps_of(ctrl: tuple, ad: tuple, sr: tuple) -> list:
    """What each order of :data:`ORDERS` keeps, from the last two control, last AD and last two
    SR values written."""
    last = {4: ctrl, 5: ad, 6: sr}
    return [tuple(pick(last, order)) for order in ORDERS]


def all_keeps(last: dict) -> list:
    """What each order of :data:`ORDERS` keeps of the writes whose :func:`lasts` is ``last``."""
    return keeps_of(tuple(last[4][-2:]), tuple(last[5][-1:]), tuple(last[6][-2:]))


def text(pattern: tuple) -> str:
    """A pattern readably: control writes as their gate and test bits (``g+t-``), AD and SR as
    ``a0A`` and ``s88``."""
    out = []
    for reg, value in pattern:
        if reg == 4:
            out.append(
                "".join(
                    n + "-+"[bool(value & b)] for b, n in ((GATE, "g"), (TEST, "t"))
                )
            )
        else:
            out.append(f"{'as'[reg - 5]}{value:02X}")
    return " ".join(out) or "-"


def split_frames(writes, starts, end: int | None = None) -> tuple:
    """``(init, [(start, frame writes)])`` of absolute ``(cycle, reg, value)`` ``writes``:
    the writes before ``starts[0]`` and, per start, those up to the next (the last up to
    ``end``, or all). Registers past :data:`REGS` are dropped."""
    w = np.asarray(writes, np.int64).reshape(-1, 3)
    w = w[w[:, 1] < REGS]
    cut = np.searchsorted(w[:, 0], [*starts, np.inf if end is None else end])
    return w[: cut[0]], [(int(s), w[a:b]) for s, a, b in zip(starts, cut[:-1], cut[1:])]


def voice_writes(w: np.ndarray, voice: int) -> np.ndarray:
    """``voice``'s control, AD and SR writes of ``w`` with registers relative to the voice."""
    out = w[(w[:, 1] >= 7 * voice + 4) & (w[:, 1] <= 7 * voice + 6)].copy()
    out[:, 1] -= 7 * voice
    return out


def absolute(rel: np.ndarray, voice: int) -> np.ndarray:
    """Voice-relative writes ``rel`` at ``voice``'s registers."""
    out = rel.copy()
    out[:, 1] += 7 * voice
    return out


def arrange(w: np.ndarray, keep: list) -> np.ndarray:
    """The ``keep`` writes ``(offset, value)`` in the earliest cycle slots of the voice-relative
    writes ``w`` that carry them."""
    pairs = set(keep)
    slots = sorted(int(c) for c, r, v in w.tolist() if (r, v) in pairs)[: len(keep)]
    return np.array([(c, r, v) for c, (r, v) in zip(slots, keep)], np.int64).reshape(
        -1, 3
    )


def ordered(w: np.ndarray, order: tuple) -> np.ndarray:
    """Voice-relative writes ``w`` as ``order`` keeps and arranges them, in the native cycle
    slots."""
    return arrange(w, kept([(int(r), int(v)) for _, r, v in w.tolist()], order))


def chip(clock: float = PAL_CLOCK, rate: float = RATE, method: str = "decimate"):
    """A pylibresidfp chip; the default decimates to :data:`RATE`, enough for envelope work."""
    from pylibresidfp import SID  # pylint: disable=import-outside-toplevel

    return SID(clock_hz=clock, sampling_hz=rate, method=method)


def env_states(sid) -> list:
    """Each voice's whole envelope state of ``sid`` as bytes: every field of
    :data:`ENV`, so equal states with equal later writes have equal futures."""
    v = sid.voices()
    m = np.stack([v[k] for k in ENV])
    return [m[:, i].tobytes() for i in range(3)]


def same(a, b) -> bool:
    """Whether two envelope traces are identical."""
    return a.tobytes() == b.tobytes()


def levels(trace, start: int, until: int) -> np.ndarray:
    """The envelope level each cycle from ``start`` to ``until`` of an envelope trace."""
    out = np.empty(until - start, np.int64)
    edges = np.append(np.asarray(trace.cycle) - start, until - start)
    for (a, b), level in zip(zip(edges[:-1], edges[1:]), trace.envelope):
        out[max(a, 0) : b] = level
    return out


def compare(a: np.ndarray, b: np.ndarray, clock: float = PAL_CLOCK) -> dict:
    """How two envelope level series differ: for how long (ms), the largest difference against
    full scale (dBFS), and the largest level ratio (dB, a level of 0 counted as 1)."""
    gap = np.abs(a - b)
    ratio = np.abs(20 * np.log10(np.maximum(a, 1) / np.maximum(b, 1)))
    return {
        "ms": 1000.0 * int((gap > 0).sum()) / clock,
        "dbfs": 20 * np.log10(gap.max() / 255) if gap.any() else -np.inf,
        "db": float(ratio.max()) if len(ratio) else 0.0,
    }


def settle_cycles(clock: float = PAL_CLOCK) -> int:
    """The most cycles a hard restart (gate off, release rate 0) takes to leave a voice in the
    same state whatever came before: a full wrap of the rate counter, during which the envelope
    does not step, then the release from the highest level to 0 at rate 0."""
    sid = chip(clock)
    sid.trace([(1, 5, 0x00), (2, 6, 0xF0), (3, 4, SAW | GATE)], 2 * 19656)
    start = sid.cycle
    trace = sid.envelope_trace(0, [(start, 6, 0x00), (start, 4, SAW)], start + 19656)
    silent = trace.cycle[trace.envelope == 0]
    return WRAP + int(silent[0] - start)


def neutral(rows: np.ndarray, since: int) -> np.ndarray:
    """Voice-relative ``rows`` with every AD write from cycle ``since`` on set to 0 and every SR
    write from then on to its release nibble alone."""
    out = rows.copy()
    late = out[:, 0] >= since
    out[late & (out[:, 1] == 5), 2] = 0
    sr = late & (out[:, 1] == 6)
    out[sr, 2] &= 0x0F
    return out


def registers(init: np.ndarray) -> np.ndarray:
    """The register file ``init`` writes leave."""
    regs = np.zeros(REGS, np.int64)
    regs[init[:, 1]] = init[:, 2]
    return regs


class Timeline:
    """A write stream's chip state at every frame start, for judging what a change to one
    voice's writes in one frame does to its envelope.

    One chip plays the whole stream once; its state is kept per frame and restored into scratch
    chips on demand, and a change is judged frame by frame from its frame until its envelope
    state equals the native one at a frame end, when the same later writes give the same
    future."""

    def __init__(self, init, frames: list, clock: float = PAL_CLOCK, **chip_args):
        self.init, self.frames, self.clock = np.asarray(init), frames, clock
        self.audio = bool(chip_args)
        self.starts = [s for s, _ in frames]
        self.scratch = [chip(clock, **chip_args) for _ in range(3)]
        sid = chip(clock, **chip_args)
        sid.trace(
            self.envelope_writes(self.init[self.init[:, 0] >= sid.cycle]),
            self.starts[0],
        )
        self.snaps, self.envs, self._rel, self._clean = [], [], {}, None
        for (_, w), end in zip(frames, [*self.starts[1:], None]):
            self.snaps.append(sid.save_state())
            self.envs.append(env_states(sid))
            if end is not None:
                sid.trace(self.envelope_writes(w), end)

    def envelope_writes(self, w: np.ndarray) -> np.ndarray:
        """The control, AD and SR writes of ``w`` of all voices, a control write by its gate and
        test bits alone (the oscillators then cost nothing to clock), unless audio is wanted.
        """
        if self.audio:
            return w
        out = w[(w[:, 1] < 21) & (w[:, 1] % 7 >= 4)].copy()
        ctrl = out[:, 1] % 7 == 4
        out[ctrl, 2] &= GATE | TEST
        return out

    def rel(self, f: int, v: int) -> np.ndarray:
        """Voice ``v``'s control, AD and SR writes of frame ``f``, relative to the voice."""
        if (f, v) not in self._rel:
            self._rel[f, v] = voice_writes(self.frames[f][1], v)
        return self._rel[f, v]

    def at(self, slot: int, g: int):
        """Scratch chip ``slot`` restored to the start of frame ``g``."""
        sid = self.scratch[slot]
        if slot or self._clean != g:
            sid.restore_state(self.snaps[g])
            self._clean = g if not slot else self._clean
        return sid

    def agree(self, v: int, writes_a: list, writes_b: list, lo: int, stop: int) -> bool:
        """Whether voice ``v``'s envelope is identical through frames ``lo`` on, playing
        ``writes_a`` and ``writes_b`` (voice-relative writes per frame) from the state at the
        start of ``lo``; the same later writes from a frame end at or after ``stop`` on equal
        states decide it."""
        chips_ = [self.at(1, lo), self.at(2, lo)]
        for g, (wa, wb) in enumerate(zip(writes_a, writes_b), lo):
            end, ws = self.starts[g + 1], (absolute(wa, v), absolute(wb, v))
            if not same(*(c.envelope_trace(v, w, end) for c, w in zip(chips_, ws))):
                return False
            for c, w in zip(chips_, ws):
                c.trace(w, end)
            if g >= stop and env_states(chips_[0])[v] == env_states(chips_[1])[v]:
                return True
        return True

    def keeps(self, f: int, v: int, news: dict, horizon: int) -> dict:
        """Which of the voice-relative frame-``f`` writes ``news`` keep voice ``v``'s envelope
        identical to the native writes' over ``horizon`` frames, as ``{key: (verdict, whether
        the later frames' writes were needed)}``."""
        until, native, out = self.starts[f + 1], self.rel(f, v), {}
        clean = ref = None
        for key, new in news.items():
            if new is native:
                out[key] = True, False
                continue
            if clean is None:
                clean = self.at(0, f)
                ref = clean.envelope_trace(v, absolute(native, v), until)
            if not same(ref, clean.envelope_trace(v, absolute(new, v), until)):
                out[key] = False, False
                continue
            sid = self.scratch[1]
            sid.restore_state(self.snaps[f])
            sid.trace(absolute(new, v), until)
            if env_states(sid)[v] == self.envs[f + 1][v]:
                out[key] = True, False
            else:
                out[key] = self._continue(sid, v, f + 1, horizon - 1), True
        return out

    def _continue(self, sid, v: int, g: int, frames: int) -> bool:
        """Whether ``sid`` (a candidate, at the start of frame ``g``) keeps voice ``v``'s
        envelope identical to the native one through ``frames`` frames or until their states
        are equal."""
        native = self.scratch[2]
        for h in range(g, g + frames):
            native.restore_state(self.snaps[h])
            w, end = absolute(self.rel(h, v), v), self.starts[h + 1]
            if not same(
                native.envelope_trace(v, w, end), sid.envelope_trace(v, w, end)
            ):
                return False
            sid.trace(w, end)
            if env_states(sid)[v] == self.envs[h + 1][v]:
                return True
        return True


def bits(mask: int) -> np.ndarray:
    """``mask`` as a boolean array over :data:`ORDERS`."""
    return np.array([c == "1" for c in format(mask, f"0{len(ORDERS)}b")])


def mask_of(*orders: int) -> int:
    """The mask with the :data:`ORDERS` indices ``orders`` set."""
    return sum(1 << (len(ORDERS) - 1 - i) for i in set(orders))


def covers(masks, most: int = 4) -> tuple:
    """``(size, [cover])``: every smallest set of orders (at most ``most``) such that each mask
    with any order in it has one of them."""
    needed = np.array([bits(m) for m in masks if m])
    for size in range(1, most + 1):
        found = [
            c
            for c in combinations(range(len(ORDERS)), size)
            if needed[:, c].any(axis=1).all()
        ]
        if found:
            return size, found
    return None, []


def pattern_changes(init: np.ndarray, frames: list) -> tuple:
    """``(changed, pairs)``: per order of :data:`ORDERS`, the (frame, voice) cases whose event
    pattern it changes, and a count of each distinct ``(native, ordered, order index)``.
    """
    regs, changed, pairs = registers(init), np.zeros(len(ORDERS), np.int64), Counter()
    for _, w in frames:
        for v in range(3):
            mine = [
                (r - 7 * v, x) for _, r, x in w.tolist() if 7 * v + 4 <= r <= 7 * v + 6
            ]
            if mine:
                start, last, memo = (
                    tuple(int(x) for x in regs[7 * v + 4 : 7 * v + 7]),
                    lasts(mine),
                    {},
                )
                native = events(start, mine)
                for i, keep in enumerate(all_keeps(last)):
                    if keep not in memo:
                        memo[keep] = events(start, list(keep))
                    if memo[keep] != native:
                        changed[i] += 1
                        pairs[(text(native), text(memo[keep]), i)] += 1
        regs[w[:, 1]] = w[:, 2]
    return changed, pairs


def order_masks(tl: Timeline, horizon: int = 10) -> dict:
    """``{mask: (count, example native pattern)}``: per (frame, voice) with an envelope event,
    the set of :data:`ORDERS` that keeps its envelope identical over ``horizon`` frames, as a
    mask (bit ``len(ORDERS) - 1 - i`` for order ``i``). Cases with the same envelope state and
    writes (and, where the later frames decided it, the same later writes) are judged once.
    """
    regs, found, examples, cache = registers(tl.init), Counter(), {}, {}
    for f in range(len(tl.frames) - horizon):
        for v in range(3):
            mine = tl.rel(f, v)
            start = tuple(int(x) for x in regs[7 * v + 4 : 7 * v + 7])
            native = events(start, [(r, x) for _, r, x in mine.tolist()])
            if native:
                mask = _mask(tl, f, v, mine, horizon, cache)
                found[mask] += 1
                examples.setdefault(mask, text(native))
        regs[tl.frames[f][1][:, 1]] = tl.frames[f][1][:, 2]
    return {m: (n, examples[m]) for m, n in found.items()}


def _mask(
    tl: Timeline, f: int, v: int, mine: np.ndarray, horizon: int, cache: dict
) -> int:
    """:func:`order_masks`' mask of one case, through ``cache``."""
    shift = np.array([tl.starts[f], 0, 0], np.int64)
    key = (tl.envs[f][v], (mine - shift).tobytes())
    if key in cache and not cache[key][1]:
        return cache[key][0]
    tail = b"".join((tl.rel(g, v) - shift).tobytes() for g in range(f + 1, f + horizon))
    if (key, tail) in cache:
        return cache[key, tail][0]
    last, news, index, seen = (
        lasts([(int(r), int(x)) for _, r, x in mine.tolist()]),
        {},
        [],
        {},
    )
    for keep in all_keeps(last):
        if keep not in seen:
            new = arrange(mine, list(keep))
            seen[keep] = (new - shift).tobytes()
            news[seen[keep]] = mine if np.array_equal(new, mine) else new
        index.append(seen[keep])
    verdicts = tl.keeps(f, v, news, horizon)
    tailed = any(t for _, t in verdicts.values())
    mask = int("".join("1" if verdicts[k][0] else "0" for k in index), 2)
    cache[(key, tail) if tailed else key] = mask, tailed
    return mask


def order_cases(tl: Timeline, order: tuple, horizon: int = 10):
    """Each (frame, voice) whose event pattern ``order`` changes, as ``(case, run)``: ``case``
    its frame, voice, native and ordered patterns and how the envelopes differ over that
    frame and the ``horizon - 1`` after it (:func:`compare`); ``run`` the ``(state, native
    writes, ordered writes, until)`` of the case, the state a chip's ``restore_state`` takes at
    the frame's start."""
    regs = registers(tl.init)
    for f in range(len(tl.frames) - horizon):
        for v in range(3):
            mine = tl.rel(f, v)
            pairs = [(r, x) for _, r, x in mine.tolist()]
            start = tuple(int(x) for x in regs[7 * v + 4 : 7 * v + 7])
            native, new = events(start, pairs), events(start, kept(pairs, order))
            if new != native:
                tail = [tl.rel(g, v) for g in range(f + 1, f + horizon)]
                until = tl.starts[f + horizon]
                a_w = absolute(np.concatenate([mine, *tail]), v)
                b_w = absolute(np.concatenate([ordered(mine, order), *tail]), v)
                sid = tl.at(0, f)
                a, b = (sid.envelope_trace(v, w, until) for w in (a_w, b_w))
                case = {
                    "frame": f,
                    "voice": v,
                    "native": text(native),
                    "ordered": text(new),
                }
                begin = tl.starts[f]
                case |= compare(
                    levels(a, begin, until), levels(b, begin, until), tl.clock
                )
                yield case, (tl.snaps[f], a_w, b_w, until)
        regs[tl.frames[f][1][:, 1]] = tl.frames[f][1][:, 2]


def hard_restarts(  # pylint: disable=too-many-locals
    tl: Timeline, settle: int, horizon: int = 10
) -> Counter:
    """Each note-on (a gate-on from gate off) as ``(settled, whole frames of hard restart,
    neutral)`` with a count: the frames before its frame's first write that the gate had been
    off with SR's release nibble 0, whether that is at least ``settle`` cycles, and whether the
    AD and SR values written meanwhile are irrelevant (:func:`neutral` gives an identical
    envelope over ``horizon`` frames; ``None`` where the hard restart began too long ago to
    check)."""
    regs, since, found = registers(tl.init), [None] * 3, Counter()
    for f in range(len(tl.frames) - horizon):
        for v in range(3):
            mine = tl.rel(f, v)
            ctrl, sr = int(regs[7 * v + 4]), int(regs[7 * v + 6])
            ons = mine[(mine[:, 1] == 4) & (mine[:, 2] & GATE != 0)]
            if len(ons) and not ctrl & GATE:
                first = int(mine[0, 0])
                held = first - since[v] if since[v] is not None else 0
                begun = (
                    int(np.searchsorted(tl.starts, since[v], side="right")) - 1
                    if held
                    else f
                )
                neutral_values = None
                if 0 <= begun and f - begun < 5:
                    cut = since[v] if held else first
                    neutral_values = _neutral(tl, f, v, begun, cut, horizon)
                found[(held >= settle, held // 19656, neutral_values)] += 1
            for cycle, reg, value in mine.tolist():
                ctrl, sr = (
                    (value, sr) if reg == 4 else (ctrl, value if reg == 6 else sr)
                )
                quiet = not ctrl & GATE and not sr & 0x0F
                since[v] = (
                    (since[v] if since[v] is not None else cycle) if quiet else None
                )
        regs[tl.frames[f][1][:, 1]] = tl.frames[f][1][:, 2]
    return found


def _neutral(tl: Timeline, f: int, v: int, begun: int, cut: int, horizon: int) -> bool:
    """Whether voice ``v``'s frame-``f`` note-on plays the same envelope with the AD and SR
    written from cycle ``cut`` of frame ``begun`` on neutral (:func:`neutral`). The hard restart
    leaves the gate off and the release nibble 0, so those writes reach the envelope only as the
    attack, decay and sustain they latch; the note frame starts from the native state with the
    neutral latches written first."""
    late = np.concatenate(
        [np.empty((0, 3), np.int64), *(tl.rel(g, v) for g in range(begun, f))]
    )
    late = late[late[:, 0] >= cut]
    ad, sr = late[late[:, 1] == 5], late[late[:, 1] == 6]
    patch = []
    if (ad[:, 2] != 0).any():
        patch.append((tl.starts[f], 5, 0))
    if (sr[:, 2] & 0xF0).any():
        patch.append((tl.starts[f], 6, int(sr[-1, 2]) & 0x0F))
    if not patch:
        return True
    rows = [tl.rel(g, v) for g in range(f, f + horizon)]
    trial = [np.concatenate([np.array(patch, np.int64), rows[0]]), *rows[1:]]
    return tl.agree(v, rows, trial, f, f)
