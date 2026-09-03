"""⊕ WP12-R Phase A / O-A8, O-A8d (DESIGN_A §1.4, PREREG_A §1) — R118/A-1, config-only.

A-1 says `wr_sealbot` populates **config-only**: no producer changes. That is a claim about
today's code, and it is derived rather than assumed — `rounds.py:211` already sets the field
on every round, `_first_sealbot_wr` (`:135-149`) already selects on `bot == "sealbot"`, and
run5 already mints two sealbot rungs. The ONLY reason the value is `None` today is that
`rung_results` never contains a sealbot entry, because `resolve_bot` refuses.

So Phase A adds no producer, no config key and no code to `rounds.py` — and these rows are
what make that falsifiable rather than merely stated.

The defect each row is the ONLY witness to:

- **O-A8 arm 1** — a phase that "delivered `wr_sealbot`" by editing the producer. It drives
  the REAL `resolve_bot` with a loadable module double and then the REAL `build_round_result`
  over run5's MINTED ladder, so the float it reads is produced by the shipped chain, not by
  a stub in the test. MUTATION (M-A18): `_first_sealbot_wr` returns `None` unconditionally —
  the `isinstance` assertion fails. No `raises`, no working-tree assertion, no firing-order
  hazard.
- **O-A8 arm 2** — the selection ORDER against the ladder as MINTED. The existing producer
  test (`test_wr_sealbot_handshake.py`) pins the rule on a synthetic three-rung ladder; this
  row pins it on `configs/run5.yaml`'s own rung sequence, which is the object A-1's
  "config-only" claim is actually about. Neither subsumes the other and neither is rewritten.
- **O-A8d — RETIRED by R332(b), which LIFTED the R118/A-1 freeze on the producer.** The row
  was a working-tree `git diff --stat` over `src/mantis/eval/rounds.py`, firing on any
  uncommitted edit. Its subject — A-1's "value populates in WP12-R Phase A" — has been merged
  since Phase T closed (R162), so the guard was refusing edits to a live producer on behalf of
  a discharged claim, and it is what banked AUDIT-1 F-14's producer half in REPAIR-1. **A
  freeze outlives its subject only by ruling**, and this is the ruling. The behavioural rows
  above are untouched: they still prove the value populates with no producer change required.

**Not duplicated here** (R79): `test_wr_sealbot_handshake.py::test_round_result_always_
carries_wr_sealbot` and `::test_sealbot_rung_with_zero_games_this_round_is_skipped_for_the_
handshake` already execute the unconditional-presence and zero-games rules. Re-running them
is Phase A's evidence; re-writing them would be a second authority over a live producer test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis.eval.rounds import build_round_result

_REPO = Path(__file__).resolve().parents[2]
_RUN5 = _REPO / "configs" / "run5.yaml"


def _run5_rungs() -> list[Any]:
    from mantis.config.loader import load_config

    return list(load_config(_RUN5).eval.ladder.rungs)


def _rung_result(*, games: int, wr: float | None) -> dict[str, Any]:
    return {"games": games, "wins": 0, "losses": 0, "draws": 0, "wr": wr,
            "wr_ci_lower": None, "wr_ci_upper": None, "eff_n": games,
            "regime_key": "k", "status": "active"}


def _round_kwargs(rungs_config: list[Any], rung_results: dict[str, Any]) -> dict[str, Any]:
    return dict(
        step=1000, round_id="r000001_1000", rungs_config=rungs_config,
        rung_results=rung_results, gate_result=None, skipped_rungs=[],
        bt={"ratings": {}, "p_hat": {}}, schedule_next={}, eval_round_wall_sec=1.0,
        reason=None, detail=None, random_wr=None,
    )


def test_a_resolvable_sealbot_rung_makes_wr_sealbot_a_float_with_no_producer_edit(
    monkeypatch, tmp_path: Path
) -> None:
    """O-A8 arm 1.

    FIRING ORDER: (1) the resolver returns a factory, (2) the factory constructs, (3) the
    producer returns a float. Under M-A18 statements (1)-(2) pass and (3)'s `isinstance`
    fails — the mutation is attributable to the producer and not to the resolver.
    """
    import mantis.bots.sealbot as sealbot_mod
    from mantis.bots.resolve import resolve_bot

    class _LoadableMinimax:
        def MinimaxBot(self, **_kwargs: Any) -> Any:  # noqa: N802 — vendored name
            raise AssertionError("this row resolves a rung; it does not play one")

    class _LoadableGame:
        class Player:
            A = object()
            B = object()

    monkeypatch.setattr(sealbot_mod, "find_vendor_root", lambda: tmp_path)
    monkeypatch.setattr(
        sealbot_mod, "load_sealbot_modules", lambda: (_LoadableMinimax(), _LoadableGame)
    )

    factory = resolve_bot("sealbot", depth=5, opponent_sims=128)
    assert callable(factory), (
        "with the vendored modules loadable, `sealbot` must RESOLVE — A-1's premise is that "
        "resolution is the only thing standing between run5 and a real `wr_sealbot`"
    )

    rungs = _run5_rungs()
    sealbot_names = [rung.name for rung in rungs if rung.bot == "sealbot"]
    assert sealbot_names == ["sealbot_d5", "sealbot_d6"], sealbot_names

    result = build_round_result(
        **_round_kwargs(rungs, {sealbot_names[0]: _rung_result(games=6, wr=0.75)})
    )
    assert isinstance(result["wr_sealbot"], float), (
        f"the round recorded a sealbot rung with 6 games and `wr_sealbot` is still "
        f"{result['wr_sealbot']!r}. A-1/R118's claim is that the value populates config-only; "
        f"a producer that cannot deliver it falsifies the ruling's premise."
    )
    assert 0.0 <= result["wr_sealbot"] <= 1.0


def test_wr_sealbot_selects_the_first_sealbot_rung_of_run5s_minted_ladder() -> None:
    """O-A8 arm 2. Ladder ORDER is read from the minted config, never transcribed: a re-mint
    that reordered the rungs would change which depth `wr_sealbot` means, and every monitor
    threshold reading it (`configs/run5.yaml:211-218`) would silently re-aim."""
    rungs = _run5_rungs()
    rung_results = {
        "kraken_raw": _rung_result(games=8, wr=0.50),
        "sealbot_d5": _rung_result(games=6, wr=0.75),
        "sealbot_d6": _rung_result(games=6, wr=0.10),
    }
    result = build_round_result(**_round_kwargs(rungs, rung_results))

    minted_first_sealbot = next(rung.name for rung in rungs if rung.bot == "sealbot")
    expected = rung_results[minted_first_sealbot]["wr"]
    assert result["wr_sealbot"] == expected, (
        f"`wr_sealbot` must come from {minted_first_sealbot!r} — the FIRST sealbot rung in "
        f"the ladder AS MINTED — never from a later one and never from a non-sealbot rung"
    )
