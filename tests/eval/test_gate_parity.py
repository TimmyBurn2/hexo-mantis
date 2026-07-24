"""R8 >300 justify: one gate-parity harness (record builders + gate_cfg + promotion-sequence
spy) shared by the truth-table, pooled-arithmetic, escalation and sequence-order oracles —
splitting would duplicate the record/spy fixtures across files and let them drift apart.

⊕ WP11-A DESIGN §b/§c.2/§c.5 — the run3 deploy-strength gate, ported knob-for-knob.
Truth table + pooled draw-aware arithmetic frozen from
`hexo_rl/eval/deploy_strength_eval.py` (read in full this session):
  * `_wr_for_label` (:363-376): `wr = (wins + 0.5*draws) / n` — draw-aware.
  * screen (:488-494): `wr_screen` = draw-aware WR over the SCREEN games alone.
  * escalation (:500-514): `escalate = wr_screen >= screen_confirm_lo` — a SINGLE lower
    bound, no upper band (`screen_confirm_hi` is NOT ported — MUST-FIX 1).
  * confirm + pool (:516-524): `pooled = screen + confirm`; `wr_confirm` = draw-aware WR over
    the POOLED set (NEVER confirm-only).
  * promotion (:555-563): `wr_ok = wr_confirm >= promotion_winrate`;
    `ci_clean = ci_lo_boot is not None and ci_lo_boot > 0.0`;
    `low_power = guard["low_power_warning"]`; `promoted = wr_ok and ci_clean and not low_power`.
  * `effective_n_guard`/`distinct_per_pair` (round_robin.py:203-253): distinct-game dedup by
    `(p1, p2, tuple(moves))`; `low_power_warning = distinct_per_pair_min < min_distinct_per_pair`.

RED-at-import: `mantis.eval.aggregate` / `mantis.eval.promote` / `mantis.eval.snapshot` do not
exist yet.

ORACLE-CHOSEN SEAM (documented, not a design contradiction — the design leaves the internal
promotion-decision function unnamed): `mantis.eval.aggregate` exposes a small PURE function
    gate_promotion_decision(wr_confirm: float, ci_lo_boot: float | None, low_power: bool,
                             promotion_winrate: float) -> bool
implementing exactly the :560-563 truth table, called BY `aggregate_gate` positionally/by-
keyword with those 4 parameter names — the 8-corner truth table is pinned against this pure
function directly (cheap, exact), and a second test proves `aggregate_gate` actually CALLS it
(spied) rather than reimplementing the table ad hoc (a B-1 fix: aggregate CONSTRUCTION must be
real, not just the table).
"""
from __future__ import annotations

import pytest

from mantis.eval.aggregate import (  # noqa: F401 — RED-at-import anchor: mantis.eval does not exist yet
    aggregate_gate,
    gate_promotion_decision,
    should_escalate,
)

# ── shared record-construction helpers ──────────────────────────────────────────────────
# ORACLE-CHOSEN record shape (parity with hexo_rl's `_play_pair`/round_robin.py convention,
# reused verbatim per design's aggregate.py citation): {"p1", "p2", "winner": "p1"|"p2"|"draw",
# "moves": [[q, r], ...]}. `moves` drives trajectory-hash dedupe (LAW-04); distinct per test.


def _records(n: int, *, wins: int, draws: int, losses: int, tag: str) -> list[dict]:
    """`n` = wins+draws+losses paired games, "cand" vs "best"; draw-aware WR = (wins+0.5*draws)/n.
    Every record gets a DISTINCT move list (`tag` + index) unless the caller overrides via
    `_records_with_moves` — distinctness here is irrelevant to wr_confirm/wr_screen arithmetic
    (only to dedup/low_power, tested separately)."""
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


# ══ the pure truth-table function — all 8 boolean corners of (wr_ok, ci_clean, low_power) ═
# wr_ok: 0.60 -> True, 0.50 -> False (bar=0.55). ci_clean: 5.0 -> True, None -> False.
@pytest.mark.parametrize(
    "wr_confirm,ci_lo_boot,low_power,expected",
    [
        (0.60, 5.0, False, True),     # wr_ok, ci_clean, not low_power -> PROMOTE (only True cell)
        (0.60, 5.0, True, False),     # wr_ok, ci_clean, low_power -> blocked
        (0.60, None, False, False),   # wr_ok, not ci_clean, not low_power -> blocked
        (0.60, None, True, False),    # wr_ok, not ci_clean, low_power -> blocked
        (0.50, 5.0, False, False),    # not wr_ok, ci_clean, not low_power -> blocked
        (0.50, 5.0, True, False),     # not wr_ok, ci_clean, low_power -> blocked
        (0.50, None, False, False),   # not wr_ok, not ci_clean, not low_power -> blocked
        (0.50, None, True, False),    # not wr_ok, not ci_clean, low_power -> blocked
    ],
)
def test_gate_truth_table_matches_run3(wr_confirm, ci_lo_boot, low_power, expected) -> None:
    from mantis.eval.aggregate import gate_promotion_decision

    assert gate_promotion_decision(wr_confirm, ci_lo_boot, low_power, 0.55) is expected


@pytest.mark.parametrize(
    "wr_confirm,ci_lo_boot,low_power,expected",
    [
        (0.55, 5.0, False, True),    # exactly AT the bar -> wr_ok is `>=`, so PROMOTE
        (0.60, -1.0, False, False),  # a NEGATIVE ci_lo_boot is present but not > 0 -> not ci_clean
        (0.60, 0.0, False, False),   # exactly zero is NOT `> 0.0` -> not ci_clean (boundary)
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


# ══ MUST-FIX 2 — pooled draw-aware wr_confirm (screen+confirm), never confirm-only ════════
def test_wr_confirm_is_pooled_draw_aware_from_raw_records() -> None:
    from mantis.eval.aggregate import aggregate_gate

    # screen: n=80, wins=24, draws=30, losses=26 -> draw-aware WR = (24+15)/80 = 39/80 = 0.4875
    screen = _records(80, wins=24, draws=30, losses=26, tag="s")
    # confirm: n=128, wins=55, draws=40, losses=33 -> (55+20)/128 = 75/128 = 0.5859375
    confirm = _records(128, wins=55, draws=40, losses=33, tag="c")
    cfg = _gate_cfg(promotion_winrate=0.55)

    result = aggregate_gate(screen, confirm, cfg)

    pooled_wr = (79 + 0.5 * 70) / 208             # 114/208 = 0.5480769230769231 — CORRECT
    confirm_only_wr = 75 / 128                    # 0.5859375 — WRONG (confirm-only bug)
    draw_blind_pooled_wr = 79 / (79 + 59)         # 0.5724637681159420 — WRONG (draw-blind bug)

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


# ══ MUST-FIX 2 — bootstrap + low-power guard consume the SAME pooled set ═════════════════
def test_bootstrap_and_low_power_guard_consume_the_pooled_set() -> None:
    from mantis.eval.aggregate import aggregate_gate

    cfg = _gate_cfg(min_distinct_per_pair=10)

    # Screen: only 3 DISTINCT move sequences repeated to fill 80 games (screen-alone distinct
    # count = 3, well under threshold 10 -> a screen-only guard would flag low_power).
    distinct_screen_moves = [[[0, 0], [1, 1]], [[0, 1], [1, 0]], [[0, 2], [1, 2]]]
    screen = [
        {"p1": "cand", "p2": "best", "winner": "p1" if i % 2 == 0 else "p2",
         "moves": distinct_screen_moves[i % 3]}
        for i in range(80)
    ]
    # Confirm: 128 NEW distinct sequences (none matching the screen ones, none repeated among
    # themselves) -> pooled distinct count = 3 + 128 = 131 >= 10 -> pooling flips low_power
    # from True (screen-alone) to False.
    #
    # A wins/draws/losses-MIXED outcome cycle (not a plain p1/p2 alternation) is load-bearing
    # here: a LAW-04-compliant bootstrap resamples the DISTINCT-game outcome array (never the
    # raw 208 records), and a low-resolution outcome set (few distinct games, or a purely
    # binary 0/1 alternating pattern) lets the 2.5% empirical quantile collapse onto the SAME
    # discrete atom across many different seeds by construction (verified empirically in
    # REVIEW_IMPL — 12 seeds through the shipped aggregate_gate all landed on one value at
    # n=11 distinct games; re-verified in this fix pass that a plain 0/1-alternating pattern
    # still collides at n=67 for this test's specific seed pair, 20260625 vs 999). Mixing in a
    # draw value (0.5) and 128 distinct games gives the bootstrap resample-mean distribution
    # enough resolution that seed_base=20260625 and seed_base=999 land on two DIFFERENT 2.5%
    # quantiles (empirically verified for this exact fixture in this fix pass).
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

    # Determinism-under-same-seed proxy for "bootstrap seeded from gate.seed_base" (:526-528):
    # two identical calls with the same seed_base must produce an IDENTICAL bootstrap CI.
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


# ══ escalation — single lower bound, no upper band (MUST-FIX 1) ═══════════════════════════
@pytest.mark.parametrize(
    "wr_screen,expected_escalate",
    [(0.30, False), (0.43, False), (0.44, True), (0.50, True), (0.99, True)],
)
def test_screen_escalates_iff_wr_screen_at_least_screen_confirm_lo(wr_screen, expected_escalate) -> None:
    from mantis.eval.aggregate import should_escalate

    assert should_escalate(wr_screen, screen_confirm_lo=0.44) is expected_escalate


# ══ promotion sequence order + terminal no-sync + F-12 snapshot pin ═══════════════════════
class _SpyOrder:
    def __init__(self) -> None:
        self.order: list[str] = []

    def guarded_load(self, model, state_dict) -> None:
        self.order.append("guarded_load")
        self._loaded_state_dict = state_dict

    def save_anchor(self, model, path, *, step, run_id, encoding) -> None:
        self.order.append("save_anchor")

    def sync_inference_weights(self, state_dict) -> None:
        self.order.append("sync_inference_weights")

    def update_checkpoint_step(self, step) -> None:
        self.order.append("update_checkpoint_step")


def _hooks(spy: "_SpyOrder", tmp_path):
    from types import SimpleNamespace

    from mantis.eval.promote import PromotionHooks

    # `best_model` must be a proper attribute-bearing fixture (not a bare `object()`, which
    # has no `__dict__` and cannot take the sabotage attribute-set below) — a plain
    # SimpleNamespace is the minimal such fixture (test_promoted_weights_are_the_evaluated_
    # snapshot_bytes assigns a throwaway `.state_dict` onto it to prove it is never read).
    anchor_state = SimpleNamespace(best_model=SimpleNamespace(), best_model_step=None)
    return PromotionHooks(
        promotion_target=spy, anchor_state=anchor_state,
        best_model_path=tmp_path / "best_model.pt", run_id="run5", encoding="gnn_axis_v1",
        save_anchor=spy.save_anchor, guarded_load=spy.guarded_load,
    )


def _fake_snapshot(monkeypatch, state_dict: dict) -> None:
    import mantis.eval.snapshot as snap_mod

    monkeypatch.setattr(snap_mod, "load_model_snapshot", lambda path, device="cpu": state_dict)


def test_gate_pass_sequence_order_anchor_save_sync_step(tmp_path, monkeypatch) -> None:
    from mantis.eval.promote import apply_gate_decision

    spy = _SpyOrder()
    hooks = _hooks(spy, tmp_path)
    _fake_snapshot(monkeypatch, {"w": 1})
    result = {"promoted": True, "eval_broken": False, "step": 4200,
              "candidate_snapshot_path": str(tmp_path / "cand.pt")}
    apply_gate_decision(hooks, result, sync_inference=True)
    assert spy.order == ["guarded_load", "save_anchor", "sync_inference_weights",
                         "update_checkpoint_step"], spy.order


def test_terminal_promotion_does_not_sync_pool(tmp_path, monkeypatch) -> None:
    from mantis.eval.promote import apply_gate_decision

    spy = _SpyOrder()
    hooks = _hooks(spy, tmp_path)
    _fake_snapshot(monkeypatch, {"w": 1})
    result = {"promoted": True, "eval_broken": False, "step": 4200,
              "candidate_snapshot_path": str(tmp_path / "cand.pt")}
    apply_gate_decision(hooks, result, sync_inference=False)
    assert "sync_inference_weights" not in spy.order
    assert "update_checkpoint_step" not in spy.order
    assert spy.order == ["guarded_load", "save_anchor"]


def test_promoted_weights_are_the_evaluated_snapshot_bytes(tmp_path, monkeypatch) -> None:
    from mantis.eval.promote import apply_gate_decision

    evaluated_state_dict = {"w": "EVALUATED_SNAPSHOT_BYTES"}
    live_module_state_dict = {"w": "MUTATED_AFTER_KICK_LIVE_MODULE"}  # must never be read

    spy = _SpyOrder()
    hooks = _hooks(spy, tmp_path)
    _fake_snapshot(monkeypatch, evaluated_state_dict)
    # Sabotage: if apply_gate_decision ever reads the live module instead of the snapshot,
    # this would be the wrong value it should NEVER see.
    hooks.anchor_state.best_model.state_dict = lambda: live_module_state_dict  # type: ignore[attr-defined]

    result = {"promoted": True, "eval_broken": False, "step": 4200,
              "candidate_snapshot_path": str(tmp_path / "cand.pt")}
    apply_gate_decision(hooks, result, sync_inference=True)

    assert spy._loaded_state_dict == evaluated_state_dict, (
        "F-12/LAW-12: promotion must load the EVALUATED snapshot the worker actually played, "
        "never the live trainer module"
    )
    assert spy._loaded_state_dict != live_module_state_dict
