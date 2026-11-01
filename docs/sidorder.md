# SID write order and hard restart

`vagrant_moses.sidorder` finds which per-voice orders of a frame's SID register writes, and which
hard restart, keep what a player commands, with the SID's envelope as the judge. It works on write
streams only; the envelope model is libresidfp through pylibresidfp (the `sidorder` extra,
imported on first use). Capturing a stream from a SID file is the caller's job.

## Input

A stream is what `NativeSid.drain_writes()` gives, absolute `(cycle, reg, value)`, cut at the
frames' start cycles:

```python
from vagrant_moses import sidorder as S

init, frames = S.split_frames(writes, starts)      # init writes, [(start, writes[n, 3])]
tl = S.Timeline(init, frames)                      # one chip pass; a state per frame start
masks = S.order_masks(tl, horizon=10)              # {mask: (count, example pattern)}
size, covers = S.covers(masks)                     # the exact minimum cover of orders
notes = S.hard_restarts(tl, S.settle_cycles())     # {(settled, frames, neutral): count}
changed, pairs = S.pattern_changes(init, frames)   # event patterns only, no chip
```

## What reaches the envelope

A voice's envelope reacts to its control, AD and SR writes in the order they arrive:

- a gate edge switches between release and attack, and so which rate nibble is in use;
- an AD write changes the rate at once with the gate on (attack or decay) and is latent with the
  gate off (it takes effect at the next gate-on);
- an SR write changes the release rate at once with the gate off, and the sustain level with it
  on;
- the rate counter, a 15-bit LFSR, is reset by neither, so the order of gate, AD and SR writes
  decides which rate it runs against when the attack starts or switches. A player can lengthen or
  shorten the attack this way.

The test bit does not reach the envelope. The gate reaches the envelope's state within 2-3
cycles, less than one 6502 store, so a player's next write always meets the new state.
`tests/test_sidorder.py` pins each of these on synthetic streams.

## Event patterns

`events(start, writes)` reduces a frame's control, AD and SR writes to a voice, from the registers
it starts with, to its **event pattern**: every write that changes the gate or test bit, the AD or
the SR, in order; with the gate off an AD write is placed just before the next gate-on (or at the
frame's end), its last value only. Writes of the value already held and timing leave no trace.
`text(pattern)` prints `a0A s88 g+t+`: AD `$0A`, SR `$88`, then a gate-on with the test bit set.

## Orders

Every frame keeps each register's last write, and optionally the write before the last of the
control (`ctrl:2`, a gate or test pulse) and of SR (`sr:2`). A per-voice order arranges the kept
`ad`, `sr`, `ctrl` (and `ctrl:2`, `sr:2`, each before its last write) in the frame's own write
slots: `ORDERS`, 60 of them. `kept(writes, order)` is the writes an order keeps.

`order_masks` plays each (frame, voice) with an envelope event from the voice's real envelope
state under every order, over the frame and the following ones, and keeps the orders whose
envelope trace is identical to the native one. `covers` finds every smallest set of orders such
that each case is kept by one of them. On the corpora measured so far it is three, `CANONICAL`,
each needed by different player builds:

| order | for |
|---|---|
| `sr:2,sr,ad,ctrl:2,ctrl` | most players: the envelope before the gate, SR first |
| `ad,sr:2,sr,ctrl:2,ctrl` | builds writing AD before SR with the gate on |
| `ctrl:2,ctrl,ad,sr:2,sr` | builds writing the gate before the envelope (ascending register dumps) |

Every case measured is kept exactly by one of the three, and nearly every tune needs only one of
them for all its frames.

### How a case is judged

`Timeline` plays the whole stream once and keeps the chip state at every frame start. A case is
judged from the state at its frame start: the native and the reordered frame are traced in the
frame and compared, then the voice's whole envelope state (every field libresidfp reports for it:
level, state, LFSR, rate counter, exponential counter, pipelines, latched rates) at the frame end.
Equal states with the same later writes have the same future, so that decides the case; otherwise
it continues frame by frame up to `horizon`, stopping at the first frame end where the states are
equal. Cases with the same envelope state at the frame start, the same frame writes relative to
the start and the same candidate are judged once (and, where the later frames' writes were needed,
with those writes).

## Hard restart

A hard restart turns the gate off with SR's release nibble 0 some frames before a note. Once it
has lasted a full rate-counter wrap (32767 cycles, during which the envelope does not step) and the
release from the highest level to 0 at rate 0, the voice is in the same state whatever came before:
the envelope at 0 and the rate counter in rate 0's nine-state cycle. A PAL frame is 9 x 2184
cycles, so longer hard restarts change nothing more. `settle_cycles(clock)` measures that bound on
the chip (39566 for PAL), and `hard_restarts` classes each note-on by the whole frames of hard
restart before its frame and whether that reaches the bound.

Most notes in tunes have a shorter hard restart (or none), so their attack depends on the note
before; what a tune commands is the hard restart's length. The canonical hard restart is therefore
one parameter, its length k in frames (0, 1, 2, or settled). The AD and SR values written during it
do not matter (checked per note by `neutral`: the same writes with AD 0 and SR's sustain 0 give an
identical envelope), except in builds that write the gate before the envelope, where the gate-on
uses the AD and SR the hard restart left.

## Other entry points

- `order_cases(tl, order)`: each (frame, voice) an order changes, with how long and by how much the
  envelopes differ (`compare`), and the state and writes to render them. Build the `Timeline` with
  `rate=44100.0, method="resample"` to render audio from that state.
- `pattern_changes(init, frames)`: per order, how many cases change an event pattern; no chip.
