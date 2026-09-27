"""resolve_bot — the ONE rung -> bot resolver: `random` in-repo; `strix` and `six` through their vendored
pins, or a refusal naming the ONE missing step. NO env-key channel: `vendor/pins.toml` is the one
authority for where an engine lives. The sims routing runs BEFORE any refusal."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mantis.bots.six as _six_mod
import mantis.bots.strix as _strix_mod
import mantis.config.resolve.nsims as _nsims_mod
from mantis.bots.random_bot import RandomBot

BotFactory = Callable[..., Any]

_KNOWN_KINDS: tuple[str, ...] = ("random", "six", "strix")


def resolve_bot(kind: str, *, opponent_sims: int | None,
                variant: str = _strix_mod.PIN_NAME, device: str | None = None) -> BotFactory:
    """`kind`'s factory (`variant`: strix's stem or six's network; `device`: six's, the round's); Raises: ValueError on an unknown kind, RungUnresolvable per rung."""
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown bot kind {kind!r}; known kinds: {sorted(_KNOWN_KINDS)}")

    if opponent_sims is not None:
        _nsims_mod.resolve_eval_model_sims(kind, opponent_sims)

    if kind == "random":
        def _factory(seed: int = 0) -> RandomBot:
            return RandomBot(seed=seed)

        return _factory

    if kind == "six":
        return _six_mod.resolve_six(opponent_sims=opponent_sims, variant=variant, device=device)
    return _strix_mod.resolve_strix(opponent_sims=opponent_sims, variant=variant)


__all__ = ["BotFactory", "resolve_bot"]
