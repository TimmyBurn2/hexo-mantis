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


def _r(idx: int, wins: int, games: int = 32, promoted: bool = True) -> RoundReading:
    return RoundReading(round_idx=idx, games=games, wins=wins, promoted=promoted)


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
        assess([RoundReading(round_idx=1, games=4, wins=9, promoted=False)])


def test_zero_game_rounds_are_no_data_not_a_loss() -> None:
    """A skipped rung (R139 grounds, an absent vendor) reports no games. Treating that as 0
    wins would manufacture a collapse out of an operator-authorized skip."""
    h = assess([_r(i, 0, games=0) for i in range(1, 5)])
    assert h.pooled_wr is None and h.label == "NO-DATA" and not h.degraded
