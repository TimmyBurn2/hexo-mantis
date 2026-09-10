"""resolve_bot — the ONE rung -> bot resolver (design §a.2 resolve.py).

`kind="random"` resolves to the in-repo `RandomBot` unconditionally. `kind="sealbot"` resolves
to the vendored fixed-depth engine through `mantis.bots.sealbot`, or refuses with a reason
naming the ONE step that is missing. A sealbot rung at a depth
R326(e) excluded from the default battery refuses with R139's OPERATOR-AUTHORIZED marker. The
kraken and strix kinds are DELETED (R346(f)): both were permanently refused rungs, so what the
tree carried was two refusal strings and their config keys.

THE ENV-KEY CHANNEL IS DELETED (WP12-R Phase A, DESIGN_A §2.2(2)), and the deletion is argued
rather than convenient (R125/R79). For `sealbot` the key became simply WRONG: the authority
for where the engine lives is `vendor/pins.toml` plus `make vendor` (CLAUDE.md's vendoring
law), and two authorities for one fact is R79's exact prohibition — an env key that can point
anywhere is a host-path surface wearing a disguise. Nothing is lost diagnostically — the replacement reasons carry strictly MORE
information — and `tests/bots/test_sealbot_resolve.py` pins both halves (behaviour and a
source scan), because a dead key still reads to an operator as an arming surface.

The sims routing runs BEFORE any refusal, exactly as it did at HEAD.
`tests/eval/test_resolver_wiring.py` re-verifies the routing per kind BY MUTATION.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mantis.bots.sealbot as _sealbot_mod
import mantis.config.resolve.nsims as _nsims_mod
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.random_bot import RandomBot

BotFactory = Callable[..., Any]

_KNOWN_KINDS: tuple[str, ...] = ("random", "sealbot")

#: The marker every R139 refusal carries, and the `operator_authorized` skip class.
_R139_SKIP_MARKER = "operator-authorized skip (R139)"

#: The sealbot rung's own precondition: LAW-15's bar IS the fixed depth, so a sealbot rung
#: minted without one has no bar to play. Deliberately NOT one of the four skip classes —
#: it is a config defect, not an environment state, and the in-run counter reports it as
#: unclassifiable (loudly) rather than inventing a fifth bucket for it.
_NO_DEPTH_REASON = (
    "sealbot rung declares no fixed depth; LAW-15's reproducible bar IS `depth`, and a "
    "sealbot rung without one names an instrument that does not exist"
)

#: Sealbot DEPTHS excluded from the default battery by ruling, depth -> grounds. R326(e).
#:
#: THE GROUNDS ARE ARITHMETIC. Depth 6 measured 30.900 s per first move (SITTING4-PREP-1, three
#: book positions, 15.331–42.176); `run5.yaml` mints the rung at `games_max: 32` under
#: `round_timeout_sec: 3600.0`. The whole round budget buys 116 opponent moves — 3.6 per game
#: across 32 — before the candidate has moved once. A rung that cannot finish does not produce a
#: weaker bar; it produces a KILLED round.
#:
#: A SKIP RATHER THAN AN UNMINTED RUNG: the rung is a minted row in all seven configs, so
#: removing it is a config act, and this one is reversible by deleting one row at the
#: gate-geometry re-adjudication. The strings are EXACT for the same reason kraken's and strix's
#: are (R143).
_R326_EXCLUDED_SEALBOT_DEPTHS: dict[int, str] = {
    6: ("sealbot depth 6 cannot finish its minted games inside eval.round_timeout_sec at the "
        "measured 30.9 s/move — the whole round budget buys ~3.6 opponent moves per game. "
        "Revisited at the gate-geometry re-adjudication"),
}

#: reason-class -> the marker substring that identifies it. ONE authority: every value here is
#: the same object the reason strings are built from, so a reason cannot drift out of the
#: classifier's reach without this mapping moving with it. Consumed by
#: `mantis.eval.pipeline`'s in-run skip-class counter (LAW-18/R164).
SKIP_REASON_MARKERS: dict[str, str] = {
    "operator_authorized": _R139_SKIP_MARKER,
    "vendor_absent": _sealbot_mod.VENDOR_ABSENT_MARKER,
    "build_absent": _sealbot_mod.BUILD_ABSENT_MARKER,
    "load_failed": _sealbot_mod.LOAD_FAILED_MARKER,
}


def _resolve_sealbot(depth: int | None) -> BotFactory:
    """Probe the vendored engine EAGERLY, then hand back a factory over what was loaded.

    Eager on purpose: a factory that resolved and only failed when the rung tried to play
    would leave every skip oracle green while a scored round died. The refusal has to happen
    where the caller catches it — `worker.py:350-356` catches `RungUnresolvable` per rung and
    nothing else, so any other exception type is fatal to a whole eval round.
    """
    if depth is None:
        raise RungUnresolvable(rung="sealbot", reason=_NO_DEPTH_REASON)
    # R326(e): before the probe, so the skip is a property of the RULING and not of whether the
    # extension happens to be built on this box — an excluded rung must read the same in a warm
    # checkout and a cold one, or the log stops distinguishing a ruled skip from a broken box.
    if depth in _R326_EXCLUDED_SEALBOT_DEPTHS:
        raise RungUnresolvable(
            rung=f"sealbot_d{depth}",
            reason=f"{_R139_SKIP_MARKER}: {_R326_EXCLUDED_SEALBOT_DEPTHS[depth]}",
        )
    try:
        minimax_module, game_module = _sealbot_mod.load_sealbot_modules()
    except RungUnresolvable:
        raise
    except Exception as exc:  # noqa: BLE001 — re-raised as a typed, per-rung refusal below
        raise RungUnresolvable(
            rung="sealbot",
            reason=(
                f"{_sealbot_mod.LOAD_FAILED_MARKER}: {exc!r}. The underlying failure is carried "
                f"verbatim rather than collapsed into 'not built' — an ABI mismatch reported as "
                f"a missing build is R145's predicted failure wearing the wrong label."
            ),
        ) from exc

    def _factory() -> Any:
        return _sealbot_mod.SealBotAdapter(
            depth=depth, minimax_module=minimax_module, game_module=game_module
        )

    return _factory


def resolve_bot(kind: str, *, depth: int | None, opponent_sims: int | None) -> BotFactory:
    """Resolve `kind` to a `BotFactory`, or raise.

    Unknown kind -> `ValueError` naming the known set. A known kind that cannot be resolved
    here -> `RungUnresolvable` (never fatal to a round — the caller catches it per rung).
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown bot kind {kind!r}; known kinds: {sorted(_KNOWN_KINDS)}")

    if opponent_sims is not None:
        _nsims_mod.resolve_eval_model_sims(kind, opponent_sims)

    if kind == "random":
        def _factory(seed: int = 0) -> RandomBot:
            return RandomBot(seed=seed)

        return _factory

    return _resolve_sealbot(depth)


__all__ = ["SKIP_REASON_MARKERS", "BotFactory", "resolve_bot"]
