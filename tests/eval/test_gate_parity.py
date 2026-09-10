"""Pin the deploy-strength gate's ported arithmetic.

R8 >300 justify: one harness — record builders, gate_cfg, promotion-sequence spy — shared by every
oracle here, which would drift apart if duplicated across files.

  * draw-aware win rate `(wins + 0.5*draws) / n`, over the SCREEN games alone for `wr_screen`;
  * escalation on a SINGLE lower bound, with no upper band;
  * `wr_confirm` over the POOLED screen+confirm set, never confirm-only;
  * `promoted = wr_ok and ci_clean and not low_power`, `ci_clean` needing `ci_lo_boot > 0.0`;
  * distinct-game dedup by `(p1, p2, tuple(moves))` feeding the low-power warning.
"""
from __future__ import annotations

import pytest

from mantis.eval.aggregate import (  # noqa: F401 — RED-at-import anchor: mantis.eval does not exist yet
    aggregate_gate,
    gate_promotion_decision,
    should_escalate,
)

# Record shape: {"p1", "p2", "winner": "p1"|"p2"|"draw", "moves": [[q, r], ...]}, where `moves`
# drives the trajectory-hash dedupe.


def _records(n: int, *, wins: int, draws: int, losses: int, tag: str) -> list[dict]:
    """Build `n` paired "cand" vs "best" games, each with a distinct move list."""
    assert wins + draws + losses == n
    out: list[dict] = []
    i = 0
    for _ in range(wins):
        out.append({"p1": "cand", "p2": "best", "winner": "p1", "moves": [[0, i], [1, i]]})
        i += 1
    for _ in range(draws):
        out.append({"p1": "cand", "p2": "best", "winner": "draw", "moves": [[2, i], [3, i]]})
        i += 1
    for _ in range(losses):
        out.append({"p1": "cand", "p2": "best", "winner": "p2", "moves": [[4, i], [5, i]]})
        i += 1
    return out


def _gate_cfg(**overrides):
    from types import SimpleNamespace

    base = dict(
        promotion_winrate=0.55, screen_confirm_lo=0.44, bootstrap_resamples=1000,
        min_distinct_per_pair=10, seed_base=20260625,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# All 8 corners of (wr_ok, ci_clean, low_power): wr_ok is 0.60 vs 0.50 against a 0.55 bar,
# ci_clean is 5.0 vs None.
@pytest.mark.parametrize(
    "wr_confirm,ci_lo_boot,low_power,expected",
    [
        (0.60, 5.0, False, True),     # the only cell that promotes
        (0.60, 5.0, True, False),
        (0.60, None, False, False),
        (0.60, None, True, False),
        (0.50, 5.0, False, False),
        (0.50, 5.0, True, False),
        (0.50, None, False, False),
        (0.50, None, True, False),
    ],
)
def test_gate_truth_table_matches_run3(wr_confirm, ci_lo_boot, low_power, expected) -> None:
    from mantis.eval.aggregate import gate_promotion_decision

    assert gate_promotion_decision(wr_confirm, ci_lo_boot, low_power, 0.55) is expected


@pytest.mark.parametrize(
    "wr_confirm,ci_lo_boot,low_power,expected",
    [
        (0.55, 5.0, False, True),    # exactly at the bar: wr_ok is `>=`
        (0.60, -1.0, False, False),  # present but not > 0
        (0.60, 0.0, False, False),   # exactly zero is not `> 0.0`
    ],
)
def test_gate_truth_table_boundary_cases(wr_confirm, ci_lo_boot, low_power, expected) -> None:
    from mantis.eval.aggregate import gate_promotion_decision

    assert gate_promotion_decision(wr_confirm, ci_lo_boot, low_power, 0.55) is expected


def test_aggregate_gate_calls_the_pure_decision_function_not_a_reimplementation(monkeypatch) -> None:
    import mantis.eval.aggregate as agg_mod

    calls: list[tuple] = []
    real = agg_mod.gate_promotion_decision

    def spy(wr_confirm, ci_lo_boot, low_power, promotion_winrate):
        calls.append((wr_confirm, ci_lo_boot, low_power, promotion_winrate))
        return real(wr_confirm, ci_lo_boot, low_power, promotion_winrate)

    monkeypatch.setattr(agg_mod, "gate_promotion_decision", spy)
    screen = _records(80, wins=50, draws=10, losses=20, tag="s")
    confirm = _records(128, wins=80, draws=20, losses=28, tag="c")
    cfg = _gate_cfg()
    result = agg_mod.aggregate_gate(screen, confirm, cfg)
    assert calls, "aggregate_gate must call gate_promotion_decision — not reimplement the table"
    wr_c, ci_c, lp_c, bar_c = calls[-1]
    assert wr_c == pytest.approx(result.wr_confirm)
    assert ci_c == result.elo_ci_lower_boot
    assert lp_c == result.low_power
    assert bar_c == cfg.promotion_winrate
    assert result.promoted == real(wr_c, ci_c, lp_c, bar_c)


def test_wr_confirm_is_pooled_draw_aware_from_raw_records() -> None:
    from mantis.eval.aggregate import aggregate_gate

    # draw-aware WR = (24+15)/80 = 0.4875
    screen = _records(80, wins=24, draws=30, losses=26, tag="s")
    # draw-aware WR = (55+20)/128 = 0.5859375
    confirm = _records(128, wins=55, draws=40, losses=33, tag="c")
    cfg = _gate_cfg(promotion_winrate=0.55)

    result = aggregate_gate(screen, confirm, cfg)

    pooled_wr = (79 + 0.5 * 70) / 208             # 0.548077 — correct
    confirm_only_wr = 75 / 128                    # 0.585938 — the confirm-only bug
    draw_blind_pooled_wr = 79 / (79 + 59)         # 0.572464 — the draw-blind bug

    assert pooled_wr != pytest.approx(confirm_only_wr)
    assert pooled_wr != pytest.approx(draw_blind_pooled_wr)
    assert pooled_wr < 0.55 < confirm_only_wr
    assert pooled_wr < 0.55 < draw_blind_pooled_wr

    assert result.wr_confirm == pytest.approx(pooled_wr), (
        "wr_confirm must be the POOLED draw-aware WR (:522-524), not confirm-only or draw-blind "
        "— either bug would flip the promotion decision at the 0.55 bar"
    )
    assert result.wr_screen == pytest.approx(39 / 80), "wr_screen is draw-aware over screen alone (:494)"
    assert result.n_screen == 80 and result.n_confirm == 128 and result.n_pooled == 208


def test_bootstrap_and_low_power_guard_consume_the_pooled_set() -> None:
    from mantis.eval.aggregate import aggregate_gate

    cfg = _gate_cfg(min_distinct_per_pair=10)

    # 3 distinct sequences filling 80 games: screen alone is under the threshold of 10, so a
    # screen-only guard would flag low_power.
    distinct_screen_moves = [[[0, 0], [1, 1]], [[0, 1], [1, 0]], [[0, 2], [1, 2]]]
    screen = [
        {"p1": "cand", "p2": "best", "winner": "p1" if i % 2 == 0 else "p2",
         "moves": distinct_screen_moves[i % 3]}
        for i in range(80)
    ]
    # 128 new distinct sequences take the pooled count to 131, flipping low_power to False. The
    # MIXED outcome cycle is load-bearing: measured, a plain 0/1 alternation collides on one 2.5%
    # quantile across seeds (12 at n=11; still colliding at n=67 for the 20260625/999 pair).
    _confirm_outcome_cycle = ["p1", "p1", "draw", "p2", "p1", "p2", "draw", "p1", "p2", "p1"]
    distinct_confirm_moves = [[[9, k], [8, k]] for k in range(128)]
    confirm = [
        {"p1": "cand", "p2": "best",
         "winner": _confirm_outcome_cycle[i % len(_confirm_outcome_cycle)],
         "moves": distinct_confirm_moves[i]}
        for i in range(128)
    ]

    result = aggregate_gate(screen, confirm, cfg)

    screen_only_distinct = 3
    pooled_distinct = 3 + 128
    assert screen_only_distinct < cfg.min_distinct_per_pair, "fixture sanity: screen alone is low-power"
    assert pooled_distinct >= cfg.min_distinct_per_pair, "fixture sanity: pooled clears the floor"
    assert result.low_power is False, (
        "the low-power guard must consume the POOLED distinct-game count (131), not the "
        "screen-alone count (3) — a screen-only guard would wrongly block promotion here"
    )
    assert result.eff_n == pooled_distinct, "eff_n (LAW-04) must be the pooled distinct-game count"

    # Two identical calls with the same seed_base must produce an identical bootstrap CI.
    result2 = aggregate_gate(screen, confirm, cfg)
    assert result.elo_ci_lower_boot == result2.elo_ci_lower_boot, (
        "the bootstrap must be seeded from gate.seed_base — identical inputs/seed must "
        "reproduce an identical CI lower bound"
    )
    cfg_other_seed = _gate_cfg(min_distinct_per_pair=10, seed_base=999)
    result3 = aggregate_gate(screen, confirm, cfg_other_seed)
    assert result3.elo_ci_lower_boot != result.elo_ci_lower_boot or result.elo_ci_lower_boot is None, (
        "a different seed_base should (with overwhelming probability) move the bootstrap CI — "
        "if this ever spuriously collides, the seed is very likely not threaded at all"
    )


@pytest.mark.parametrize(
    "wr_screen,expected_escalate",
    [(0.30, False), (0.43, False), (0.44, True), (0.50, True), (0.99, True)],
)
def test_screen_escalates_iff_wr_screen_at_least_screen_confirm_lo(wr_screen, expected_escalate) -> None:
    from mantis.eval.aggregate import should_escalate

    assert should_escalate(wr_screen, screen_confirm_lo=0.44) is expected_escalate


class _SpyOrder:
    """Record the promotion call order, carrying no actor surface so a sync-shaped call raises."""

    def __init__(self) -> None:
        self.order: list[str] = []

    def guarded_load(self, model, state_dict) -> None:
        self.order.append("guarded_load")
        self._loaded_state_dict = state_dict

    def save_anchor(self, model, path, *, step, run_id, encoding) -> None:
        self.order.append("save_anchor")


def _hooks(spy: "_SpyOrder", tmp_path):
    from types import SimpleNamespace

    from mantis.eval.promote import DeployTagHooks

    # `best_model` must bear attributes: a bare `object()` has no `__dict__` and cannot take the
    # throwaway `.state_dict` the sabotage row assigns onto it.
    anchor_state = SimpleNamespace(best_model=SimpleNamespace(), best_model_step=None)
    return DeployTagHooks(
        anchor_state=anchor_state,
        best_model_path=tmp_path / "best_model.pt", run_id="run5", encoding="gnn_axis_v1",
        save_anchor=spy.save_anchor, guarded_load=spy.guarded_load,
    )


def _fake_snapshot(monkeypatch, state_dict: dict) -> None:
    import mantis.eval.snapshot as snap_mod

    monkeypatch.setattr(snap_mod, "load_model_snapshot", lambda path, device="cpu": state_dict)


def test_gate_pass_sequence_is_anchor_only(tmp_path, monkeypatch) -> None:
    """Prove a gate pass moves ONLY the deploy tag: `guarded_load -> save_anchor` and nothing
    else, by full-list equality so any sync-shaped call fails here."""
    from mantis.eval.promote import apply_gate_decision

    spy = _SpyOrder()
    hooks = _hooks(spy, tmp_path)
    _fake_snapshot(monkeypatch, {"w": 1})
    result = {"promoted": True, "eval_broken_reason": None, "step": 4200,
              "candidate_snapshot_path": str(tmp_path / "cand.pt")}
    apply_gate_decision(hooks, result)
    assert spy.order == ["guarded_load", "save_anchor"], spy.order


def test_promoted_weights_are_the_evaluated_snapshot_bytes(tmp_path, monkeypatch) -> None:
    from mantis.eval.promote import apply_gate_decision

    evaluated_state_dict = {"w": "EVALUATED_SNAPSHOT_BYTES"}
    live_module_state_dict = {"w": "MUTATED_AFTER_KICK_LIVE_MODULE"}  # must never be read

    spy = _SpyOrder()
    hooks = _hooks(spy, tmp_path)
    _fake_snapshot(monkeypatch, evaluated_state_dict)
    # Sabotage: the value promotion would read if it ever consulted the live module.
    hooks.anchor_state.best_model.state_dict = lambda: live_module_state_dict  # type: ignore[attr-defined]

    result = {"promoted": True, "eval_broken_reason": None, "step": 4200,
              "candidate_snapshot_path": str(tmp_path / "cand.pt")}
    apply_gate_decision(hooks, result)

    assert spy._loaded_state_dict == evaluated_state_dict, (
        "F-12/LAW-12: promotion must load the EVALUATED snapshot the worker actually played, "
        "never the live trainer module"
    )
    assert spy._loaded_state_dict != live_module_state_dict
