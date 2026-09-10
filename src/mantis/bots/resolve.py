"""resolve_bot — the ONE rung -> bot resolver.

`kind="random"` resolves to the in-repo `RandomBot` unconditionally. `kind="sealbot"`
resolves to the vendored fixed-depth engine through `mantis.bots.sealbot`, or refuses with
a reason naming the ONE step that is missing; a sealbot rung at a ruled-out depth refuses
with the operator-authorized marker. There is NO env-key channel: the authority for where
the engine lives is `vendor/pins.toml` plus `make vendor`, and a second authority that can
point anywhere is a host-path surface in disguise. The sims routing runs BEFORE any refusal.
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

#: The sealbot rung's own precondition: the reproducible bar IS the fixed depth, so a rung
#: minted without one has no bar to play. Deliberately NOT one of the four skip classes — it
#: is a config defect, not an environment state, and is reported as unclassifiable.
_NO_DEPTH_REASON = (
    "sealbot rung declares no fixed depth; LAW-15's reproducible bar IS `depth`, and a "
    "sealbot rung without one names an instrument that does not exist"
)

#: Sealbot DEPTHS excluded from the default battery by ruling, depth -> grounds.
#:
#: The grounds are arithmetic: depth 6 measured 30.900 s per first move (three book
#: positions, 15.331-42.176), and at `games_max: 32` under `round_timeout_sec: 3600.0` the
#: whole round budget buys ~3.6 opponent moves per game before the candidate has moved once.
#: A rung that cannot finish produces a KILLED round, not a weaker bar. It is a skip rather
#: than an unminted rung because the row is minted in all seven configs, so this is reversible
#: by deleting one row. The strings are EXACT.
_R326_EXCLUDED_SEALBOT_DEPTHS: dict[int, str] = {
    6: ("sealbot depth 6 cannot finish its minted games inside eval.round_timeout_sec at the "
        "measured 30.9 s/move — the whole round budget buys ~3.6 opponent moves per game. "
        "Revisited at the gate-geometry re-adjudication"),
}

#: reason-class -> the marker substring that identifies it. ONE authority: every value here is
#: the same object the reason strings are built from, so a reason cannot drift out of the
#: classifier's reach. Consumed by `mantis.eval.pipeline`'s in-run skip-class counter.
SKIP_REASON_MARKERS: dict[str, str] = {
    "operator_authorized": _R139_SKIP_MARKER,
    "vendor_absent": _sealbot_mod.VENDOR_ABSENT_MARKER,
    "build_absent": _sealbot_mod.BUILD_ABSENT_MARKER,
    "load_failed": _sealbot_mod.LOAD_FAILED_MARKER,
}


def _resolve_sealbot(depth: int | None) -> BotFactory:
    """Probe the vendored engine EAGERLY, then hand back a factory over what was loaded.

    Eager on purpose: a factory that failed only when the rung tried to play would leave every
    skip oracle green while a scored round died. `worker.py` catches `RungUnresolvable` per
    rung and nothing else, so any other exception type is fatal to a whole eval round.
    """
    if depth is None:
        raise RungUnresolvable(rung="sealbot", reason=_NO_DEPTH_REASON)
    # Before the probe, so an excluded rung reads the same in a warm checkout and a cold one
    # and the log still distinguishes a ruled skip from a broken box.
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
