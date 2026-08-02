"""resolve_bot — the ONE rung -> bot resolver (design §a.2 resolve.py).

`kind="random"` resolves to the in-repo `RandomBot` unconditionally. `kind="sealbot"` resolves
to the vendored fixed-depth engine through `mantis.bots.sealbot`, or refuses with a reason
naming the ONE step that is missing. `kind in {"kraken","strix"}` refuses with R139's
OPERATOR-AUTHORIZED grounds, verbatim and per rung.

THE ENV-KEY CHANNEL IS DELETED (WP12-R Phase A, DESIGN_A §2.2(2)), and the deletion is argued
rather than convenient (R125/R79). For `sealbot` the key became simply WRONG: the authority
for where the engine lives is `vendor/pins.toml` plus `make vendor` (CLAUDE.md's vendoring
law), and two authorities for one fact is R79's exact prohibition — an env key that can point
anywhere is a host-path surface wearing a disguise. For `kraken`/`strix` it was a
silent-arming surface with nothing behind it: R139 rules both out for run5 with named grounds,
and a key whose only effect is to change which of two refusal strings is printed is not a
feature. Nothing is lost diagnostically — the replacement reasons carry strictly MORE
information — and `tests/bots/test_sealbot_resolve.py` pins both halves (behaviour and a
source scan), because a dead key still reads to an operator as an arming surface.

ORDERING IS LOAD-BEARING AND IT IS A TRAP (DESIGN_A §2.2(4)). The sims routing below executes
BEFORE any refusal, exactly as it did at HEAD. `eval.kraken_model_sims` and
`eval.strix_model_sims` have exactly ONE live consumer each — `resolve_eval_model_sims`, and
the only route to it for those kinds is this call. Hoisting the grounds-bearing raise above it
would instantly turn two consumer-registry citations (`tests/config/
test_every_key_has_consumer.py:52-53`) into the precise falsehood R93 exists to catch, and the
LAW-08 bijection test would stay green while it happened. `tests/eval/test_resolver_wiring.py`
re-verifies the routing per kind BY MUTATION, which is the only thing that can see it.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mantis.bots.sealbot as _sealbot_mod
import mantis.config.resolve.nsims as _nsims_mod
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.random_bot import RandomBot

BotFactory = Callable[..., Any]

_KNOWN_KINDS: tuple[str, ...] = ("random", "sealbot", "kraken", "strix")

#: R139's own words, per rung. The strings are EXACT and they are the deliverable: R143 calls
#: these skips OPERATOR-AUTHORIZED rather than a dispatcher shortfall, so a reader of the log
#: has to be able to tell a ruled skip from a broken box. A paraphrase is a drift, and
#: kraken's grounds appearing on a strix skip is a false diagnosis — both are pinned.
_R139_SKIP_GROUNDS: dict[str, str] = {
    "kraken": "weights not cleanly accessible",
    "strix": "actively changing",
}

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

    if kind == "sealbot":
        return _resolve_sealbot(depth)

    raise RungUnresolvable(
        rung=kind, reason=f"{_R139_SKIP_MARKER}: {kind} — {_R139_SKIP_GROUNDS[kind]}"
    )


__all__ = ["SKIP_REASON_MARKERS", "BotFactory", "resolve_bot"]
