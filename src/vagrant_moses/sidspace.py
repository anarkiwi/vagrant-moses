"""The SID envelope generator's response to the order of one frame's control, AD and SR writes,
tested exhaustively from synthesised generator states with pylibresidfp's ``envelope_grid``.

An AD or SR write only stores its nibbles and, for the state in force, the rate; the generator
reads them when its rate counter matches and at state changes. So the claim tested here is that
without a rate-counter match inside the write window every order of a frame's writes (each
register's own writes kept in order) leaves the envelope and the generator's state identical, and
that with matches the orders differ by envelope steps taken at the rate in force between writes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations

import numpy as np
from tqdm import tqdm

ATTACK, DECAY_SUSTAIN, RELEASE = range(3)
PERIODS = (
    (0xFF, 1),
    (0x5D, 2),
    (0x36, 4),
    (0x1A, 8),
    (0x0E, 16),
    (0x06, 30),
    (0x00, 1),
)


def rates() -> np.ndarray:
    """The 16 rate-counter compare values, read from the generator itself."""
    from pylibresidfp import (
        ENVELOPE_STATE,
        SID,
    )  # pylint: disable=import-outside-toplevel

    out = []
    for r in range(16):
        sid = SID()
        sid.write(6, r)
        out.append(int(np.frombuffer(sid.envelope_state(0), ENVELOPE_STATE)["rate"][0]))
    return np.array(out, np.uint16)


def step(lfsr: np.ndarray) -> np.ndarray:
    """The rate counter's LFSR one cycle on."""
    return (lfsr >> 1) | (((lfsr << 14) ^ (lfsr << 13)) & 0x4000)


def back(lfsr: np.ndarray, times: int = 1) -> np.ndarray:
    """The rate counter's LFSR ``times`` cycles before."""
    lfsr = np.asarray(lfsr, np.uint16)
    for _ in range(times):
        lfsr = ((lfsr << 1) & 0x7FFF) | (((lfsr >> 14) ^ lfsr) & 1)
    return lfsr


def far(cycles: int, table: np.ndarray) -> int:
    """An LFSR state that matches none of ``table`` within ``cycles`` cycles."""
    x = np.arange(1, 0x8000, dtype=np.uint16)
    ok = np.ones(len(x), bool)
    for _ in range(cycles + 1):
        ok &= ~np.isin(x, table)
        x = step(x)
    return int(np.flatnonzero(ok)[0] + 1)


def period(level: np.ndarray, falling: bool) -> np.ndarray:
    """The exponential counter period at ``level`` reached from above or below."""
    marks = np.array([m for m, _ in PERIODS])
    vals = np.array([p for _, p in PERIODS])
    if falling:
        idx = np.array([np.flatnonzero(marks >= lv)[-1] for lv in level])
    else:
        idx = np.array([np.flatnonzero(marks <= lv)[0] for lv in level])
    return vals[idx]


def starts(state: int, level, ad, sr, lfsr, table: np.ndarray) -> np.ndarray:
    """Settled generator states: ``state`` at each ``level`` with AD ``ad``, SR ``sr`` and rate
    counter ``lfsr`` (broadcast together)."""
    from pylibresidfp import ENVELOPE_STATE  # pylint: disable=import-outside-toplevel

    level, ad, sr, lfsr = np.broadcast_arrays(
        *(np.asarray(v) for v in (level, ad, sr, lfsr))
    )
    out = np.zeros(level.shape, ENVELOPE_STATE)
    a, d, s, r = ad >> 4, ad & 15, sr >> 4, sr & 15
    out["lfsr"], out["envelope_counter"], out["env3"] = lfsr, level, level
    out["attack"], out["decay"], out["sustain"], out["release"] = a, d, s * 0x11, r
    out["rate"] = table[(a, d, r)[state]]
    out["exponential_counter_period"] = period(level.ravel(), state != ATTACK).reshape(
        level.shape
    )
    out["state"] = out["next_state"] = state
    out["gate"] = state != RELEASE
    out["counter_enabled"] = (level != 0) | (state == ATTACK)
    return out


def programs(writes: list, gap: int) -> tuple:
    """Every order of ``writes`` ``(reg, value)`` (one per register), one write each ``gap``
    cycles from cycle 1, as a ``(P, K, 3)`` array, with the cycles that cover them."""
    cycles = 1 + gap * len(writes) + 8
    orders = list(permutations(writes))
    out = np.array(
        [[(1 + gap * i, r, v) for i, (r, v) in enumerate(o)] for o in orders]
    )
    return out.reshape(len(orders), len(writes), 3).astype(np.int64), cycles


def disagree(
    states: np.ndarray,
    progs: np.ndarray,
    cycles: int,
    levels: bool = True,
    jobs: int = 0,
) -> np.ndarray:
    """Per state, per group of ``progs`` sharing ``axis 0`` rows (shape ``(G, P, K, 3)``): whether
    the group's programs disagree, ``(len(states), G)``; with ``levels`` only in the envelope's
    level changes and end state."""
    from pylibresidfp import envelope_grid  # pylint: disable=import-outside-toplevel

    g, p = progs.shape[:2]
    flat = progs.reshape(g * p, *progs.shape[2:])
    chunk = max(1, (1 << 22) // (g * p))
    parts = [states[i : i + chunk] for i in range(0, len(states), chunk)]

    def one(part):
        d = envelope_grid(part, flat, cycles, levels).reshape(len(part), g, p, 2)
        return (d != d[:, :, :1]).any(axis=(2, 3))

    with ThreadPoolExecutor(jobs or None) as pool:
        return np.concatenate(
            list(tqdm(pool.map(one, parts), total=len(parts), leave=False))
        )


SAMPLE = (np.array([0x00, 0x8A, 0xF5, 0x4F]), np.array([0x00, 0xFF, 0x48, 0xA3]))


def no_match_orders(
    state: int, gap: int, full: str = "ad", values=None, levels=None
) -> tuple:
    """``(cases, disagreeing)``: from ``state`` at each of ``levels`` (default every one), with the
    rate counter matching nothing in the window, every held and new value of the ``full``
    register (``values``, default all 256) against :data:`SAMPLE` held and new values of the
    other, each gate and test change, and every order of the three writes ``gap`` cycles apart.
    ``disagreeing`` lists ``(start, (new AD, new SR, gate flip, test flip))`` where the orders'
    envelopes or end states differ."""
    table = rates()
    _, cycles = programs([(4, 0), (5, 0), (6, 0)], gap)
    values = np.arange(256) if values is None else np.asarray(values)
    levels = (
        np.arange(255 if state == ATTACK else 256)
        if levels is None
        else np.asarray(levels)
    )
    held = (values, SAMPLE[0]) if full == "ad" else (SAMPLE[0], values)
    new = (values, SAMPLE[1]) if full == "ad" else (SAMPLE[1], values)
    grid = np.meshgrid(levels, *held, indexing="ij")
    states = starts(state, *(g.ravel() for g in grid), far(cycles, table), table)
    gate = int(state != RELEASE)
    keys = [
        (int(a), int(s), g, t)
        for a in new[0]
        for s in new[1]
        for g, t in ((0, 0), (1, 0), (0, 1), (1, 1))
    ]
    groups = np.stack(
        [
            programs([(4, 0x40 | (gate ^ g) | t << 3), (5, a), (6, s)], gap)[0]
            for a, s, g, t in keys
        ]
    )
    bad = disagree(states, groups, cycles)
    return bad.size, [(states[i], keys[j]) for i, j in np.argwhere(bad)]


def main(argv=None) -> int:
    """Run :func:`no_match_orders` over the whole space (every state, AD and SR in turn, writes
    1 and 4 cycles apart) and print each pass; non-zero if any order mattered."""
    import argparse  # pylint: disable=import-outside-toplevel

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args(argv)
    failed = 0
    for gap in args.gaps:
        for full in ("ad", "sr"):
            for state in (ATTACK, DECAY_SUSTAIN, RELEASE):
                cases, bad = no_match_orders(state, gap, full)
                failed += len(bad)
                print(
                    f"state {state} gap {gap} every {full}: {len(bad)} of {cases} differ"
                )
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
