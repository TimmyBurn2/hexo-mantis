"""Pin the gate aggregate's arithmetic over the GSPRT's games, and promotion as an anchor-only move.

  * draw-aware win rate `(wins + 0.5*draws) / n` over EVERY game the GSPRT played;
  * the pair bootstrap seeded from `gate.seed_base`;
  * distinct-game dedup by `(p1, p2, tuple(moves))` feeding the low-power guard over the same set.
"""
from __future__ import annotations

import pytest

from mantis.eval.aggregate import aggregate_gate

#: A verdict the aggregate takes as given; its arithmetic is the subject here.
_PROMOTE = {"decision": "promote", "checks": 1}

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
        bootstrap_resamples=1000,
        min_distinct_per_pair=10, seed_base=20260625,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_gate_rate_is_draw_aware_over_every_game_the_gsprt_played() -> None:
    first = _records(80, wins=24, draws=30, losses=26, tag="s")
    later = _records(128, wins=55, draws=40, losses=33, tag="c")
    result = aggregate_gate(first + later, _gate_cfg(), _PROMOTE)

    pooled_wr = (79 + 0.5 * 70) / 208             # 0.548077 — correct
    later_only_wr = 75 / 128                      # 0.585938 — a last-batch-only bug
    draw_blind_pooled_wr = 79 / (79 + 59)         # 0.572464 — the draw-blind bug
    assert pooled_wr != pytest.approx(later_only_wr) and pooled_wr != pytest.approx(draw_blind_pooled_wr)
    assert result.wr_confirm == pytest.approx(pooled_wr) and result.wr_screen == result.wr_confirm
    assert result.n_pooled == 208 and result.n_confirm == 0


def test_bootstrap_and_low_power_guard_consume_the_pooled_set() -> None:
    cfg = _gate_cfg(min_distinct_per_pair=10)

    # 3 distinct sequences filling 80 games: the first batch alone is under the threshold of 10,
    # so a first-batch guard would flag low_power.
    distinct_first_moves = [[[0, 0], [1, 1]], [[0, 1], [1, 0]], [[0, 2], [1, 2]]]
    first = [
        {"p1": "cand", "p2": "best", "winner": "p1" if i % 2 == 0 else "p2",
         "moves": distinct_first_moves[i % 3]}
        for i in range(80)
    ]
    # 128 new distinct sequences take the pooled count to 131, flipping low_power to False. MIXED is
    # load-bearing: 0/1 alone collided on one 2.5% quantile (12 at n=11; still n=67, 20260625/999).
    _later_outcome_cycle = ["p1", "p1", "draw", "p2", "p1", "p2", "draw", "p1", "p2", "p1"]
    distinct_later_moves = [[[9, k], [8, k]] for k in range(128)]
    later = [
        {"p1": "cand", "p2": "best",
         "winner": _later_outcome_cycle[i % len(_later_outcome_cycle)],
         "moves": distinct_later_moves[i]}
        for i in range(128)
    ]

    result = aggregate_gate(first + later, cfg, _PROMOTE)

    first_only_distinct = 3
    pooled_distinct = 3 + 128
    assert first_only_distinct < cfg.min_distinct_per_pair, "fixture sanity: the first batch alone is low-power"
    assert pooled_distinct >= cfg.min_distinct_per_pair, "fixture sanity: pooled clears the floor"
    assert result.low_power is False, (
        "the low-power guard must consume every game's distinct count (131), not the first "
        "batch's (3) — a first-batch guard would wrongly block promotion here"
    )
    assert result.eff_n == pooled_distinct, "eff_n (LAW-04) must be the pooled distinct-game count"

    # Two identical calls with the same seed_base must produce an identical bootstrap CI.
    result2 = aggregate_gate(first + later, cfg, _PROMOTE)
    assert result.elo_ci_lower_boot == result2.elo_ci_lower_boot, (
        "the bootstrap must be seeded from gate.seed_base — identical inputs/seed must "
        "reproduce an identical CI lower bound"
    )
    cfg_other_seed = _gate_cfg(min_distinct_per_pair=10, seed_base=999)
    result3 = aggregate_gate(first + later, cfg_other_seed, _PROMOTE)
    assert result3.elo_ci_lower_boot != result.elo_ci_lower_boot or result.elo_ci_lower_boot is None, (
        "a different seed_base should (with overwhelming probability) move the bootstrap CI — "
        "if this ever spuriously collides, the seed is very likely not threaded at all"
    )


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
