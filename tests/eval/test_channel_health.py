"""⊕ R343(b)(iii)/(iv) — the saturation rule and the degradation flag, with their planted breaks.

The defect each row is the ONLY witness to:

* a SATURATED rung reported as ordinary progress. The burst's screen read 79 % at step 2004 and
  the rule is expected to trip inside run6's first third, so a label that never fires would let
  strength claims keep answering to an instrument that has stopped discriminating;
* a degradation flag that fires on a PLATEAU. The self-play-cycling signature is a drop
  **while promotions continue**; without the conjunction the flag is a noise generator, and the
  row that pins it is the one where the same drop with promotions STOPPED must NOT flag;
* a running maximum that includes the reading being judged — which can never be exceeded by it,
  so the flag could only ever fire on the first drop. The row drives a recovery-then-drop
  sequence that a self-inclusive maximum gets wrong;
* a counter that resets when it should accumulate: two CONSECUTIVE flags are an architect read,
  so the streak is the operand of a decision and is pinned as such.
"""
from __future__ import annotations

import pytest

from mantis.eval.channel_health import RoundReading, assess


def _r(idx: int, wins: int, games: int = 32, promoted: bool = True,
       rung: str = "sealbot_d5") -> RoundReading:
    return RoundReading(round_idx=idx, games=games, wins=wins, promoted=promoted, rung=rung)


def test_no_history_is_no_data_never_a_zero() -> None:
    """A channel with no rounds has not measured 0.0 — it has measured nothing, and a
    dashboard that draws 0 % for it reports a collapse that did not happen."""
    h = assess([])
    assert h.pooled_wr is None and h.label == "NO-DATA" and not h.saturated


def test_a_mid_range_channel_is_measuring() -> None:
    h = assess([_r(i, 18) for i in range(1, 5)])
    assert h.pooled_wr == pytest.approx(18 / 32)
    assert not h.saturated and not h.degraded and h.label == "MEASURING"


def test_the_saturation_rule_fires_at_the_ruling_threshold() -> None:
    """(iii): pooled WR over the last four rounds >= 0.85."""
    h = assess([_r(i, 28) for i in range(1, 5)])  # 28/32 = 0.875
    assert h.pooled_wr >= 0.85 and h.saturated and h.label == "SATURATED"
    assert h.rounds_pooled == 4


def test_the_saturation_rule_does_not_fire_just_below_it() -> None:
    """The planted break for (iii): one win fewer per round and the label must go back."""
    h = assess([_r(i, 27) for i in range(1, 5)])  # 27/32 = 0.84375
    assert h.pooled_wr < 0.85 and not h.saturated and h.label == "MEASURING"


def test_only_the_last_window_is_pooled() -> None:
    """A rule that pooled the whole run would never notice a recent change."""
    history = [_r(i, 30) for i in range(1, 5)] + [_r(i, 10) for i in range(5, 9)]
    h = assess(history)
    assert h.rounds_pooled == 4 and h.pooled_wr == pytest.approx(10 / 32)
    assert not h.saturated, "the early high rounds must not keep the label saturated"


def test_a_drop_while_promotions_continue_is_flagged() -> None:
    """(iv): the self-play-cycling signature — weaker against a FIXED opponent while the
    internal gate keeps promoting, which is exactly what an internal-only bar cannot see."""
    history = [_r(i, 30, promoted=True) for i in range(1, 5)]
    history += [_r(i, 12, promoted=True) for i in range(5, 9)]
    h = assess(history)
    assert h.running_max_wr is not None and h.pooled_wr < h.running_max_wr
    assert h.degraded and h.label == "DEGRADED"
    assert h.consecutive_degradation_flags == 1


def test_the_same_drop_with_promotions_STOPPED_is_not_flagged() -> None:
    """THE PLANTED BREAK FOR (iv), and the row that makes the flag a signature rather than a
    plateau detector. Identical win counts; only `promoted` moves."""
    history = [_r(i, 30, promoted=True) for i in range(1, 5)]
    history += [_r(i, 12, promoted=False) for i in range(5, 9)]
    h = assess(history)
    assert h.pooled_wr == pytest.approx(12 / 32)
    assert not h.degraded, (
        "a drop with promotions stopped is an ordinary plateau; flagging it makes the counter "
        "that gates an architect read into a noise generator"
    )


def test_a_small_drop_inside_the_ci_is_not_flagged() -> None:
    """The threshold is 2xCI, not any decrease — a channel that wobbled must not flag."""
    history = [_r(i, 20, promoted=True) for i in range(1, 5)]
    history += [_r(i, 19, promoted=True) for i in range(5, 9)]
    h = assess(history)
    assert not h.degraded


def test_the_running_maximum_excludes_the_window_being_judged() -> None:
    """A self-inclusive maximum can never be exceeded by the reading it contains, so the drop
    would be measured against itself. Drive a recovery, then a drop: the maximum must remember
    the EARLIER peak, not the window in hand."""
    history = [_r(i, 31, promoted=True) for i in range(1, 5)]   # peak window
    history += [_r(i, 16, promoted=True) for i in range(5, 9)]  # recovery-ish
    history += [_r(i, 8, promoted=True) for i in range(9, 13)]  # the drop being judged
    h = assess(history)
    assert h.running_max_wr is not None and h.running_max_wr > h.pooled_wr
    assert h.degraded


def test_consecutive_flags_accumulate_and_a_clean_round_resets_them() -> None:
    """Two CONSECUTIVE flags are an architect read (R343(b)(iv)), so the streak is the operand
    of a decision — it must accumulate across calls and reset on a clean reading."""
    dropped = [_r(i, 30, promoted=True) for i in range(1, 5)]
    dropped += [_r(i, 12, promoted=True) for i in range(5, 9)]
    first = assess(dropped, previous_consecutive_flags=0)
    second = assess(dropped, previous_consecutive_flags=first.consecutive_degradation_flags)
    assert (first.consecutive_degradation_flags, second.consecutive_degradation_flags) == (1, 2)

    recovered = assess([_r(i, 30, promoted=True) for i in range(1, 5)],
                       previous_consecutive_flags=2)
    assert recovered.consecutive_degradation_flags == 0, "a clean reading clears the streak"


def test_an_impossible_reading_is_refused_not_pooled() -> None:
    """More wins than games would pool to a win rate above 1 and silently poison both rules."""
    with pytest.raises(ValueError, match="cannot have happened"):
        assess([RoundReading(round_idx=1, games=4, wins=9, promoted=False,
                             rung="sealbot_d5")])


def test_zero_game_rounds_are_no_data_not_a_loss() -> None:
    """A skipped rung (R139 grounds, an absent vendor) reports no games. Treating that as 0
    wins would manufacture a collapse out of an operator-authorized skip."""
    h = assess([_r(i, 0, games=0) for i in range(1, 5)])
    assert h.pooled_wr is None and h.label == "NO-DATA" and not h.degraded


def test_the_window_refuses_to_pool_across_a_rung_identity_change() -> None:
    """AUDIT-1 F-14 applied to a pooled window, and the reason it is not optional.

    `_first_sealbot_wr`'s own docstring: once `sealbot_d5` saturates it draws 0 games
    off-cadence and the reported number silently becomes `sealbot_d6`'s — so a trajectory rule
    over a mixed window compares two OPPONENTS and calls the difference a regression. Here a
    strong d5 history is followed by a weak d6 reading: the pooled WR must be d6's alone, and
    the degradation flag must NOT fire on the opponent having changed.
    """
    history = [_r(i, 30, rung="sealbot_d5") for i in range(1, 5)]
    history += [_r(5, 8, rung="sealbot_d6")]
    h = assess(history)
    assert h.rounds_pooled == 1, "the window must stop at the identity change"
    assert h.pooled_wr == pytest.approx(8 / 32)
    assert not h.degraded, (
        "a weaker reading against a HARDER opponent is not degradation; flagging it would "
        "make the counter that gates an architect read fire on the ladder working as designed"
    )


# ══ THE PRODUCER (LAW-07): the rules must reach the event stream, not just compute ═══════
def test_the_pipeline_emits_channel_health_from_a_real_round_result() -> None:
    """R4/LAW-07 — every monitor input cites a LIVE PRODUCER.

    THE DEFECT THIS CLOSES, and it was mine: `channel_health.assess` shipped first with 12
    green rows and NO caller anywhere in `src/`. The saturation label and the degradation flag
    were named as dashboard lines in an exit screen while nothing emitted them — a dashboard
    line with no producer is the phantom-gate shape LAW-07 exists to forbid, and naming one is
    the overclaim the curation protocol forbids. This row drives the real
    `EvalPipeline._assess_external_channel` over a real round-result mapping and asserts the
    event lands.
    """
    from types import SimpleNamespace

    from mantis.eval.pipeline import EvalPipeline

    emitted: list[dict] = []
    fake = SimpleNamespace(
        _sink=SimpleNamespace(emit=emitted.append),
        _eval_cfg=SimpleNamespace(ladder=SimpleNamespace(
            bootstrap_resamples=200, bootstrap_ci_level=0.95, bootstrap_seed=0)),
        _external_history=[], _degradation_flags=0, _round_counter=3,
    )
    result = {"wr_sealbot": 0.875, "wr_sealbot_games": 32,
              "wr_sealbot_rung": "sealbot_d5", "promoted": True}

    EvalPipeline._assess_external_channel(fake, result, round_id="r000004_4000", step=4000)

    assert len(emitted) == 1, "the assessment must publish exactly one reading per round"
    ev = emitted[0]
    assert ev["event"] == "eval_channel_health"
    assert ev["rung"] == "sealbot_d5" and ev["label"] == "SATURATED"
    assert ev["pooled_wr"] == pytest.approx(0.875) and ev["pooled_games"] == 32
    assert ev["consecutive_degradation_flags"] == 0
    assert fake._external_history and fake._external_history[-1].wins == 28


def test_a_round_the_instrument_did_not_play_is_absent_not_a_loss() -> None:
    """THE PLANTED BREAK for the producer. A skipped or off-cadence sealbot rung reports no
    games; recording that as 0 wins would manufacture a collapse out of an operator-authorized
    skip and trip the degradation flag on the ladder working as designed."""
    from types import SimpleNamespace

    from mantis.eval.pipeline import EvalPipeline

    emitted: list[dict] = []
    fake = SimpleNamespace(
        _sink=SimpleNamespace(emit=emitted.append),
        _eval_cfg=SimpleNamespace(ladder=SimpleNamespace(
            bootstrap_resamples=200, bootstrap_ci_level=0.95, bootstrap_seed=0)),
        _external_history=[], _degradation_flags=0, _round_counter=1,
    )
    EvalPipeline._assess_external_channel(
        fake, {"wr_sealbot": None, "wr_sealbot_games": 0, "wr_sealbot_rung": None,
               "promoted": False},
        round_id="r000002_2000", step=2000)

    assert emitted == [], "a round the instrument did not play publishes nothing"
    assert fake._external_history == [], "and it does not enter the series as a loss"
