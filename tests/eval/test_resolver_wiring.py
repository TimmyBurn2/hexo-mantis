"""Eval reads the SAME sims resolver seam self-play does: every bot kind routes its sims through
`resolve_eval_model_sims` before it resolves or refuses."""
from __future__ import annotations

import mantis.config.resolve.nsims as nsims_mod
import pytest
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.resolve import resolve_bot
from mantis.config.resolve.nsims import resolve_eval_model_sims


@pytest.mark.parametrize("opponent", ["random", "strix"])
def test_unknown_opponent_and_none_value_raise_pre_existing_green(opponent: str) -> None:
    """Pre-existing HEAD behavior (NOT new-RED) — kept for completeness of the contract pin."""
    assert resolve_eval_model_sims(opponent, 96) == 96
    with pytest.raises(ValueError):
        resolve_eval_model_sims(opponent, None)


# `resolve_bot` is fed the ALREADY-RESOLVED config value as `opponent_sims` (a real int; None
# is reserved for "this rung has no sims dimension at all") and must route it through
# `resolve_eval_model_sims(kind, opponent_sims)` for every kind — the resolver call itself is
# the consumer, independent of whether the constructed bot uses the int.
class _RoutingReached(Exception):
    """Raised BY THE SPY, from inside `resolve_eval_model_sims`, and by nothing else.

    The sentinel can only escape `resolve_bot` if the routing call executed before any `return`
    or `raise` on every path, so the ordering is observed directly rather than inferred.
    """


def test_strix_rung_model_sims_route_through_resolve_eval_model_sims(monkeypatch) -> None:
    """`resolve_bot("strix", …)` must route sims through `resolve_eval_model_sims` FIRST.

    The spy RAISES rather than returning, so the sentinel reaches the caller only if the routing
    ran before `resolve_bot` could return a factory or raise its own refusal — the assertion IS
    the ordering. Observing it through `pytest.raises(RungUnresolvable)` instead was an accident
    of the rung being unresolvable: measured `1 failed` with a built vendor tree and `1 passed`
    without, so CI had never once observed the ordering the row is named for."""
    calls: list[tuple[str, int | None]] = []

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        raise _RoutingReached

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)

    with pytest.raises(_RoutingReached):
        resolve_bot("strix", opponent_sims=128)

    assert calls == [("strix", 128)], (
        f"resolve_bot('strix', …) must route model_sims through resolve_eval_model_sims "
        f"exactly once, BEFORE it resolves or refuses; observed {calls}"
    )


# The routing must survive the resolver rewrite: this row asserts routing PER KIND while being
# agnostic about whether a kind resolves or raises, so it holds both in CI (no vendor tree,
# strix raises) and on a box with the extension built. The rows above are HEAD's pins.
# THE TRAP: `eval.{kraken,strix}_model_sims` have exactly ONE live consumer each, reached only
# through this call, so hoisting a refusal above the routing would falsify two consumer-registry
# citations while the LAW-08 bijection test stayed green. A single aggregated "the spy was
# called" assertion would be green under a mutation that broke one kind's routing.
@pytest.mark.parametrize("kind", ["random", "strix"])
def test_every_bot_kind_routes_its_sims_through_the_resolver_after_the_rewrite(
    monkeypatch, kind: str
) -> None:
    calls: list[tuple[str, int | None]] = []
    real = nsims_mod.resolve_eval_model_sims

    def spy(opponent: str, cfg_value: int | None) -> int:
        calls.append((opponent, cfg_value))
        return real(opponent, cfg_value)

    monkeypatch.setattr(nsims_mod, "resolve_eval_model_sims", spy)
    try:
        resolve_bot(kind, opponent_sims=128)
    except RungUnresolvable:
        pass  # strix's refusal is expected without a vendor tree; the routing already happened.

    assert (kind, 128) in calls, (
        f"resolve_bot({kind!r}, ...) did not reach resolve_eval_model_sims. The routing must "
        f"execute BEFORE the refusal, exactly as it does at HEAD. calls={calls}"
    )
