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
class _RoutingReached(Exception):
    """Raised BY THE SPY, from inside `resolve_eval_model_sims`, and by nothing else.

    ⊕ WP12-R Phase A / G-A3. This sentinel is the instrument: it can only escape `resolve_bot`
    if the routing call executed BEFORE any `return` or `raise` on every path — which is the
    ordering itself, observed directly instead of inferred from an unrelated exception.
    """


def test_sealbot_rung_model_sims_route_through_resolve_eval_model_sims(monkeypatch) -> None:
    """`resolve_bot("sealbot", …)` must route sims through `resolve_eval_model_sims` FIRST.

    **⊕ WP12-R Phase A, G-A3: the MECHANISM is re-pointed; the SUBJECT is unchanged.**

    The subject — sims routing runs before the resolution outcome — is load-bearing: it is the
    R93 trap DESIGN_A §2.2(4) found. `eval.{kraken,strix}_model_sims` have exactly ONE live
    consumer each, and the only route to it is this call, so an ordering change would turn two
    consumer-registry citations false while `test_every_key_has_consumer.py` stayed green.

    **What was wrong with observing it through `pytest.raises(RungUnresolvable)`.** That form
    was an ACCIDENT of the rung being unresolvable. It made the row assert two things at once —
    the ordering, and that sealbot cannot resolve — and only the second is environment-stable.
    After the §2.2 rewrite, on a box where the vendored engine is fetched and BUILT, sealbot
    RESOLVES and the row failed `DID NOT RAISE` for a reason that has nothing to do with its
    subject. Measured both ways: `1 failed` with a built tree, `1 passed` without.

    Worse, the CI-green reading was hollow: the row passed **because an unrelated raise fired**,
    so CI had never once observed the ordering it is named for. ORACLE_NOTES_A F-A4 predicted
    this exact fragility before any production code existed and deliberately did not pre-empt
    it; G-A3 is the ruling that followed the prediction coming true.

    **The re-point.** The spy RAISES instead of returning. The sentinel reaches the caller only
    if the routing ran before `resolve_bot` could return a factory or raise its own refusal —
    so the assertion IS the ordering, and it holds identically whether or not the engine is
    built, because the sentinel escapes long before the sealbot arm is reached. Strictly more
    coverage than the sealed form, not less: the row now observes its own subject for the first
    time, in both environments.
    """
    import mantis.config.resolve.nsims as nsims_mod
    from mantis.bots.resolve import resolve_bot

    calls: list[tuple[str, int | None]] = []

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        raise _RoutingReached

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)

    with pytest.raises(_RoutingReached):
        resolve_bot("sealbot", depth=5, opponent_sims=128)

    assert calls == [("sealbot", 128)], (
        f"resolve_bot('sealbot', …) must route model_sims through resolve_eval_model_sims "
        f"exactly once, BEFORE it resolves or refuses; observed {calls}"
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


# ── ⊕ WP12-R Phase A / O-A6 — the routing survives the §2.2 resolver rewrite ────────────
# The three rows above pin the routing against HEAD's control flow, each hard-requiring the
# refusal shape HEAD produces. DESIGN_A §2.2 rewrites that control flow: `sealbot` gains a
# real adapter arm and the kraken/strix refusal becomes R139's grounds. The row below is the
# POST-REWRITE pin, and it is a different claim from the three above — it asserts the
# routing per kind while being AGNOSTIC about whether the kind resolves or raises, which is
# what makes it survive both CI (no vendor tree, sealbot raises) and a box (extension built,
# sealbot returns a factory). The three above are not rewritten: they are HEAD's pins and
# re-running them is Phase A's evidence, not a second authority.
#
# THE TRAP, and it is why this row exists at all (DESIGN_A §2.2(4)): `eval.kraken_model_sims`
# and `eval.strix_model_sims` have exactly ONE live consumer each —
# `test_every_key_has_consumer.py:52-53` cites `resolve_eval_model_sims`, and the only route
# to it for those kinds is `resolve.py:52-53`. Hoisting the grounds-bearing raise above the
# routing call would instantly turn two consumer-registry citations into the precise
# falsehood R93 exists to catch, and `test_every_key_has_consumer.py` would stay GREEN —
# that a LAW-08 bijection test cannot see this is the whole reason R93 demands mutation over
# grep. MUTATION (M-A4): hoist the raise above `:52-53`; the kraken and strix cells RED, and
# random/sealbot stay GREEN `[reached, invisible]` — which is why the row is parametrized and
# asserts PER KIND. A single aggregated "the spy was called" assertion would be green under
# M-A4 and the row would be unfalsifiable.
@pytest.mark.parametrize("kind", ["random", "sealbot", "kraken", "strix"])
def test_every_bot_kind_routes_its_sims_through_the_resolver_after_the_rewrite(
    monkeypatch, kind: str
) -> None:
    import mantis.config.resolve.nsims as nsims_mod
    from mantis.bots.resolve import resolve_bot

    calls: list[tuple[str, int | None]] = []
    real = nsims_mod.resolve_eval_model_sims

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        return real(opponent, cfg_value)

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)
    try:
        resolve_bot(kind, depth=5 if kind == "sealbot" else None, opponent_sims=128)
    except RungUnresolvable:
        pass  # the refusal is the EXPECTED outcome for kraken/strix and for sealbot in CI;
        # the routing must already have happened by the time it fires (SR-3 / DESIGN §2.2(4))

    assert (kind, 128) in calls, (
        f"resolve_bot({kind!r}, ...) did not reach resolve_eval_model_sims. For kraken/strix "
        f"that makes test_every_key_has_consumer.py:52-53 a FALSE citation (R93/LAW-08); the "
        f"routing must execute BEFORE the refusal, exactly as it does at HEAD. calls={calls}"
    )
