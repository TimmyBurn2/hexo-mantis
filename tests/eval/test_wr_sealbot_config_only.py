"""`wr_sealbot` populates config-only: no producer change is required.

The producer already sets the field on every round and already selects on `bot == "sealbot"`;
the value is `None` only because `resolve_bot` refuses, so `rung_results` never carries a
sealbot entry. Both rows drive the REAL resolver and the REAL `build_round_result` over the
MINTED ladder, so the float read is produced by the shipped chain rather than by a stub.

The unconditional-presence and zero-games rules live in `test_wr_sealbot_handshake.py` and are
deliberately not duplicated here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis.eval.rounds import build_round_result

_REPO = Path(__file__).resolve().parents[2]
_RUN5 = _REPO / "configs" / "run6.yaml"


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
    """A resolvable sealbot rung makes `wr_sealbot` a float.

    FIRING ORDER: the resolver returns a factory, the factory constructs, the producer returns a
    float — so a failure is attributable to the producer and not to the resolver.
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
    assert sealbot_names == ["sealbot_d5"], sealbot_names

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
    """Ladder ORDER is read from the minted config, never transcribed: a re-mint that reordered
    the rungs would change which depth `wr_sealbot` means and silently re-aim every threshold."""
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
