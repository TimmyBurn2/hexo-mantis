"""⊕ WP11-A DESIGN §b/§c — eval reads the SAME sims resolver seam self-play does (rule 3;
dispatch item 6). Retires two zero-consumer keys: `eval.random_model_sims` (already exists at
HEAD but has no live consumer — the random floor is the consumer this WP wires) and
`eval.sealbot_model_sims` (consumer = sealbot rungs).

RED-at-import (file-level): the top-level `from mantis.bots.resolve import resolve_bot` below
is the RED-at-import anchor — `mantis.bots` does not exist yet, so THIS WHOLE FILE fails
collection today (mirrors the house convention in tests/train/test_coordinator_gates.py's
top-level `check_sealbot_wr_hard_abort` anchor import). Per-test provenance, for the record
(all currently unreachable behind the same collection error, but documented so IMPL's
green-transition is auditable test-by-test):
  * the kraken/strix sub-cases of `test_unknown_opponent_and_none_value_raise_*` would be
    RED-by-assertion even without the anchor (against the EXISTING
    `mantis.config.resolve.nsims.resolve_eval_model_sims` — `_KNOWN_OPPONENTS` is
    `("random", "sealbot")` at HEAD, so kraken/strix currently, pre-IMPL, wrongly raise
    "unknown eval opponent" instead of resolving);
  * the random/sealbot sub-cases of `test_unknown_opponent_and_none_value_raise_pre_existing_
    green` exercise ALREADY-GREEN pre-existing behavior — kept for completeness, not a new pin;
  * every other test needs `mantis.bots.resolve` / spies on `mantis.config.resolve.nsims`
    together, and is RED-at-import via the same anchor.
"""
from __future__ import annotations

import pytest

from mantis.bots.protocol import RungUnresolvable
from mantis.bots.resolve import resolve_bot  # noqa: F401 — RED-at-import anchor: mantis.bots does not exist yet
from mantis.config.resolve.nsims import resolve_eval_model_sims


@pytest.mark.parametrize("opponent", ["random", "sealbot"])
def test_unknown_opponent_and_none_value_raise_pre_existing_green(opponent: str) -> None:
    """Pre-existing HEAD behavior (NOT new-RED) — kept for completeness of the contract pin."""
    assert resolve_eval_model_sims(opponent, 96) == 96
    with pytest.raises(ValueError):
        resolve_eval_model_sims(opponent, None)


@pytest.mark.parametrize("opponent", ["kraken", "strix"])
def test_kraken_and_strix_are_known_opponents_after_wiring(opponent: str) -> None:
    """RED today: `_KNOWN_OPPONENTS` at HEAD is `("random", "sealbot")` only — kraken/strix
    wrongly raise "unknown eval opponent" until IMPL extends the tuple (design §a.4:
    `_KNOWN_OPPONENTS -> ("random","sealbot","kraken","strix")`, nsims.py:11)."""
    assert resolve_eval_model_sims(opponent, 128) == 128


def test_unknown_opponent_still_raises_after_extension() -> None:
    with pytest.raises(ValueError):
        resolve_eval_model_sims("nnue", 96)


def test_none_value_still_raises_for_every_known_opponent() -> None:
    for opponent in ("random", "sealbot", "kraken", "strix"):
        with pytest.raises(ValueError):
            resolve_eval_model_sims(opponent, None)


# ── the actual eval-side call sites — RED-at-import (mantis.bots.resolve doesn't exist) ──
# NOTE (ORACLE-CHOSEN seam): `resolve_bot` is fed the ALREADY-RESOLVED config value as
# `opponent_sims` (a real int, never None — None is reserved for "this rung has no sims
# dimension at all", e.g. sealbot's fixed `depth`) and is required to route it THROUGH
# `resolve_eval_model_sims(kind, opponent_sims)` for every kind (including "random" and
# "sealbot", whose numeric result may be otherwise unused by the constructed bot) — this is
# precisely how dispatch item 6 retires the two zero-consumer keys: the resolver call itself
# is the consumer, independent of whether the bot constructor does anything with the int.
def test_sealbot_rung_model_sims_route_through_resolve_eval_model_sims(monkeypatch) -> None:
    """`mantis.bots.resolve.resolve_bot` (sealbot rungs) must call
    `resolve_eval_model_sims("sealbot", cfg.eval.sealbot_model_sims)` — spied via monkeypatch
    on the resolver module the bots package imports it from."""
    import mantis.config.resolve.nsims as nsims_mod
    from mantis.bots.resolve import resolve_bot

    calls: list[tuple[str, int | None]] = []
    real = nsims_mod.resolve_eval_model_sims

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        return real(opponent, cfg_value)

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)
    with pytest.raises(RungUnresolvable):
        resolve_bot("sealbot", depth=5, opponent_sims=128)
    assert ("sealbot", 128) in calls, (
        "resolve_bot('sealbot', ...) must route model_sims through resolve_eval_model_sims"
    )


@pytest.mark.parametrize("kind", ["kraken", "strix"])
def test_opponent_rung_model_sims_route_through_resolver(monkeypatch, kind: str) -> None:
    import mantis.config.resolve.nsims as nsims_mod
    from mantis.bots.resolve import resolve_bot

    calls: list[tuple[str, int | None]] = []
    real = nsims_mod.resolve_eval_model_sims

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        return real(opponent, cfg_value)

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)
    with pytest.raises(RungUnresolvable):
        resolve_bot(kind, depth=None, opponent_sims=128)
    assert (kind, 128) in calls, (
        f"resolve_bot({kind!r}, ...) must route model_sims through resolve_eval_model_sims"
    )


def test_random_floor_routes_through_resolver(monkeypatch) -> None:
    """The random floor (RandomBot) has no model_sims of its own (it is not a search bot), but
    the resolver contract is still the ONE authority named opponent -> sims for every
    resolver-routed opponent kind; `resolve_bot("random", ...)` must not bypass it silently —
    it must route `cfg.eval.random_model_sims` through `resolve_eval_model_sims` too (retiring
    the zero-consumer key `eval.random_model_sims` per dispatch item 6), even though the
    resulting int has no further effect on RandomBot's uniform-legal-move behavior."""
    import mantis.config.resolve.nsims as nsims_mod
    from mantis.bots.resolve import resolve_bot

    calls: list[tuple[str, int | None]] = []
    real = nsims_mod.resolve_eval_model_sims

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        return real(opponent, cfg_value)

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)
    resolve_bot("random", depth=None, opponent_sims=96)
    assert ("random", 96) in calls, (
        "resolve_bot('random', ...) must route through resolve_eval_model_sims too — "
        "eval.random_model_sims must have a live EXERCISED consumer"
    )
