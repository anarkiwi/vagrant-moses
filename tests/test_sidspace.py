"""vagrant_moses.sidspace: synthesised envelope states and the order invariance of a frame's
writes when the rate counter does not match during them."""

import numpy as np
import pytest

pylibresidfp = pytest.importorskip("pylibresidfp")
if not hasattr(pylibresidfp, "envelope_grid"):
    pytest.skip("pylibresidfp has no envelope_grid", allow_module_level=True)

from vagrant_moses import sidspace as X  # pylint: disable=wrong-import-position

TABLE = X.rates()
NONE = np.zeros((0, 3), np.int64)


def test_lfsr_steps_invert_and_far_never_matches():
    x = np.arange(1, 0x8000, dtype=np.uint16)
    assert (X.back(X.step(x)) == x).all() and (X.step(X.back(x)) == x).all()
    lfsr = np.array([X.far(40, TABLE)], np.uint16)
    for _ in range(41):
        assert lfsr[0] not in TABLE
        lfsr = X.step(lfsr)


@pytest.mark.parametrize("seed", range(4))
def test_starts_are_the_chips_own_settled_states(seed):
    rng = np.random.default_rng(seed)
    seen = 0
    for _ in range(60):
        sid = pylibresidfp.SID()
        ad, sr = (int(v) for v in rng.integers(256, size=2))
        sid.write(4, 0x40)
        sid.run(NONE, 40000)
        sid.write(5, ad)
        sid.write(6, sr)
        sid.write(4, 0x41)
        sid.run(NONE, sid.cycle + int(rng.integers(50, 60000)))
        if rng.random() < 0.5:
            sid.write(4, 0x40)
            sid.run(NONE, sid.cycle + int(rng.integers(50, 60000)))
        e = np.frombuffer(sid.envelope_state(0), pylibresidfp.ENVELOPE_STATE)[0]
        busy = (
            "state_pipeline",
            "envelope_pipeline",
            "exponential_pipeline",
            "reset_lfsr",
        )
        if any(e[k] for k in busy) or e["new_exponential_counter_period"]:
            continue
        made = X.starts(
            int(e["state"]), int(e["envelope_counter"]), ad, sr, int(e["lfsr"]), TABLE
        )
        for k in ("exponential_counter", "exponential_counter_period") + (
            ("counter_enabled",) if e["envelope_counter"] == 0 else ()
        ):
            made[k] = e[k]
        assert made.tobytes() == e.tobytes()
        seen += 1
    assert seen > 30


@pytest.mark.parametrize("state", [X.ATTACK, X.DECAY_SUSTAIN, X.RELEASE])
@pytest.mark.parametrize("gap", [1, 4])
def test_no_match_no_order_matters(state, gap):
    values = np.arange(0, 256, 17)
    cases, bad = X.no_match_orders(state, gap, "ad", values, np.arange(0, 255, 23))
    assert cases and not bad
    cases, bad = X.no_match_orders(state, gap, "sr", values, np.arange(0, 255, 23))
    assert cases and not bad


def test_a_match_between_the_writes_makes_order_matter():
    _, cycles = X.programs([(5, 0), (6, 0)], 8)
    states = X.starts(X.ATTACK, 0x40, 0x00, 0x00, X.back(TABLE[:1], 4)[0], TABLE)
    progs = X.programs([(5, 0xF0), (6, 0xF0)], 8)[0][None]
    assert X.disagree(states.reshape(1), progs, cycles).all()


def test_moving_a_gate_edge_changes_only_when_the_state_changed():
    states = X.starts(X.RELEASE, 0x80, 0x00, 0x00, X.far(20, TABLE), TABLE).reshape(1)
    progs = X.programs([(4, 0x41), (5, 0x00)], 4)[0][None]
    assert X.disagree(states, progs, 20, levels=False).all()
    assert not X.disagree(states, progs, 20).any()


def test_main_runs_each_pass(monkeypatch, capsys):
    monkeypatch.setattr(X, "no_match_orders", lambda state, gap, full: (4, []))
    assert X.main(["--gaps", "2"]) == 0
    assert capsys.readouterr().out.count("0 of 4 differ") == 6
    monkeypatch.setattr(X, "no_match_orders", lambda state, gap, full: (4, [1]))
    assert X.main(["--gaps", "2"]) == 1
