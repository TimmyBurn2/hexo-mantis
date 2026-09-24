"""resolve_bot — the ONE rung -> bot resolver: `random` in-repo; `strix` through its vendored tree,
or a refusal naming the ONE missing step. NO env-key channel: `vendor/pins.toml` + `make vendor` is
the one authority for where an engine lives. The sims routing runs BEFORE any refusal."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mantis.bots.strix as _strix_mod
import mantis.config.resolve.nsims as _nsims_mod
from mantis.bots.random_bot import RandomBot

BotFactory = Callable[..., Any]

_KNOWN_KINDS: tuple[str, ...] = ("random", "strix")


def resolve_bot(kind: str, *, opponent_sims: int | None,
                variant: str = _strix_mod.PIN_NAME) -> BotFactory:
    """Resolve `kind` to a `BotFactory`, or raise.

    Unknown kind -> `ValueError` naming the known set. A known kind that cannot be resolved
    here -> `RungUnresolvable` (never fatal to a round — the caller catches it per rung);
    `variant` is read by the strix kind only (the pinned checkpoint's stem)."""
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown bot kind {kind!r}; known kinds: {sorted(_KNOWN_KINDS)}")

    if opponent_sims is not None:
        _nsims_mod.resolve_eval_model_sims(kind, opponent_sims)

    if kind == "random":
        def _factory(seed: int = 0) -> RandomBot:
            return RandomBot(seed=seed)

        return _factory

    return _strix_mod.resolve_strix(opponent_sims=opponent_sims, variant=variant)


__all__ = ["BotFactory", "resolve_bot"]
