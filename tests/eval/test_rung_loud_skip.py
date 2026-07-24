"""⊕ WP11-A DESIGN §a.4 census verdict — "0 of 6 ladder rungs resolve locally at HEAD" — and
dispatch success criterion "unresolvable rungs loud-skip with visible log + event (test)".

RED-at-import: `mantis.eval.rounds` / `mantis.eval.pipeline` / `mantis.bots.resolve` do not
exist yet.

ORACLE-CHOSEN SEAM (design leaves the exact skip-resolution call site unnamed — worker.py
records `skipped_rungs` into the sidecar result JSON; the PARENT pipeline is the one holding
the injected `sink`/logger, so event+log emission is necessarily parent-side, over the
worker's returned `skipped_rungs` list). Two small units, factored so this file does not need
a live subprocess round (that integration lives in test_round_end_to_end.py):

  * `mantis.eval.rounds.resolve_ladder_rungs(rungs, resolve_bot_fn) -> (resolved: dict[str,
    Any], skipped: list[dict])` — attempts `resolve_bot_fn(rung.bot, depth=rung.depth,
    opponent_sims=rung.opponent_sims)` per rung; a `RungUnresolvable` is CAUGHT and appended
    to `skipped` as `{"rung": rung.name, "reason": <exc.reason>}` — NEVER raised further
    (never fatal to the round).
  * `mantis.eval.pipeline.emit_rung_skip_events(round_id, skipped, sink) -> None` — for every
    skipped entry: one `eval_rung_skipped` event {round_id, rung, reason} via `sink.emit`, AND
    one ERROR-level log line naming the rung + reason (via the module logger, `caplog`-visible).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from mantis.eval.pipeline import emit_rung_skip_events  # noqa: F401 — RED-at-import anchor
from mantis.eval.rounds import resolve_ladder_rungs  # noqa: F401 — RED-at-import anchor


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _rung(name: str, bot: str = "sealbot", depth=5, opponent_sims=None) -> SimpleNamespace:
    return SimpleNamespace(name=name, bot=bot, depth=depth, opponent_sims=opponent_sims,
                           opening_book="book_v1_s20260625_p4", deploy_matched=True,
                           games_max=32, variant="d5")


_SIX_MINTED_RUNGS = [
    _rung("sealbot_d5", "sealbot", depth=5),
    _rung("kraken_raw", "kraken", depth=None, opponent_sims=None),
    _rung("sealbot_d6", "sealbot", depth=6),
    _rung("kraken_mcts200", "kraken", depth=None, opponent_sims=200),
    _rung("strix_128", "strix", depth=None, opponent_sims=128),
    _rung("strix_256", "strix", depth=None, opponent_sims=256),
]


def _always_unresolvable(kind, *, depth, opponent_sims):
    from mantis.bots.protocol import RungUnresolvable

    raise RungUnresolvable(rung=kind, reason=f"no adapter installed (WP12-R): {kind}")


def test_unresolvable_rung_emits_skip_event_and_log(caplog) -> None:
    from mantis.eval.pipeline import emit_rung_skip_events
    from mantis.eval.rounds import resolve_ladder_rungs

    resolved, skipped = resolve_ladder_rungs([_SIX_MINTED_RUNGS[0]], _always_unresolvable)
    assert resolved == {}
    assert skipped == [{"rung": "sealbot_d5", "reason": "no adapter installed (WP12-R): sealbot"}]

    sink = _SpySink()
    with caplog.at_level(logging.ERROR):
        emit_rung_skip_events("r000001_100", skipped, sink)

    skip_events = sink.named("eval_rung_skipped")
    assert len(skip_events) == 1
    assert skip_events[0]["round_id"] == "r000001_100"
    assert skip_events[0]["rung"] == "sealbot_d5"
    assert "reason" in skip_events[0]
    assert any(r.levelno >= logging.ERROR and "sealbot_d5" in r.getMessage() for r in caplog.records), (
        "an unresolvable rung must ALSO log at ERROR level naming the rung (never silent)"
    )


def test_all_six_head_rungs_skip_loud_at_head() -> None:
    """The census verdict: 0/6 minted ladder rungs resolve at HEAD (no bot adapters, empty
    vendor pins). All six must loud-skip; the round-level machinery (gate + random floor) is
    NOT part of this unit test (see test_round_end_to_end.py) — here we assert the resolver
    layer alone produces exactly 6 skips and zero resolved entries."""
    from mantis.eval.rounds import resolve_ladder_rungs

    resolved, skipped = resolve_ladder_rungs(_SIX_MINTED_RUNGS, _always_unresolvable)
    assert resolved == {}
    assert len(skipped) == 6
    assert {s["rung"] for s in skipped} == {r.name for r in _SIX_MINTED_RUNGS}
    assert all(s["reason"] for s in skipped), "every skip must carry a non-empty reason"


def test_resolved_stub_rung_plays_and_does_not_skip() -> None:
    from mantis.eval.rounds import resolve_ladder_rungs

    def resolver(kind, *, depth, opponent_sims):
        return SimpleNamespace(kind=kind)  # a stub "BotFactory"

    resolved, skipped = resolve_ladder_rungs([_SIX_MINTED_RUNGS[0]], resolver)
    assert skipped == []
    assert "sealbot_d5" in resolved


def test_skip_is_per_round_not_cached_forever() -> None:
    """A rung becoming resolvable NEXT round must play — resolution is re-evaluated per
    round, never cached/sticky from a prior failure."""
    from mantis.eval.rounds import resolve_ladder_rungs

    calls = {"n": 0}

    def flaky_resolver(kind, *, depth, opponent_sims):
        calls["n"] += 1
        if calls["n"] == 1:
            from mantis.bots.protocol import RungUnresolvable

            raise RungUnresolvable(rung=kind, reason="transiently unavailable")
        return SimpleNamespace(kind=kind)

    rung = [_SIX_MINTED_RUNGS[0]]
    resolved_round1, skipped_round1 = resolve_ladder_rungs(rung, flaky_resolver)
    assert resolved_round1 == {} and len(skipped_round1) == 1

    resolved_round2, skipped_round2 = resolve_ladder_rungs(rung, flaky_resolver)
    assert skipped_round2 == [], "a rung resolvable THIS round must not still be skipped"
    assert "sealbot_d5" in resolved_round2
