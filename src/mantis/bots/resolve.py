"""resolve_bot — the ONE rung -> bot resolver (design §a.2 resolve.py).

`kind="random"` resolves to the in-repo `RandomBot` unconditionally. `kind in {"sealbot",
"kraken","strix"}` checks the ENV-KEY contract (`MANTIS_BOT_SEALBOT` / `MANTIS_BOT_KRAKEN`
/ `MANTIS_BOT_STRIX`) — each names an adapter entry a WP12-R package would provide. At
HEAD no adapter is installed, so with OR without the env key set, resolution raises
`RungUnresolvable` with a reason string that DISTINGUISHES the two cases. NO host path,
NO default endpoint appears in this file (dispatch law: env keys / vendor pins only).

Every resolved kind routes its opponent-side sims THROUGH `resolve_eval_model_sims` (the
ONE sims resolver seam eval shares with self-play, rule 3 / dispatch item 6) whenever the
caller supplies a real `opponent_sims` int — this is precisely how the two previously
zero-consumer keys (`eval.random_model_sims` / `eval.sealbot_model_sims`) gain a live,
EXERCISED consumer. `opponent_sims=None` means "this rung has no sims dimension the
caller resolved" (e.g. a bare protocol-level call) and is never routed — routing a None
would incorrectly raise the resolver's own "required field absent" error for a caller
that never had a config value to give it.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import mantis.config.resolve.nsims as _nsims_mod
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.random_bot import RandomBot

BotFactory = Callable[..., Any]

_KNOWN_KINDS: tuple[str, ...] = ("random", "sealbot", "kraken", "strix")

#: The env-key contract for each externally-adapted kind (WP12-R property). NO host path
#: or default endpoint lives here — only the KEY NAME an operator's environment may set.
_ENV_KEYS: dict[str, str] = {
    "sealbot": "MANTIS_BOT_SEALBOT",
    "kraken": "MANTIS_BOT_KRAKEN",
    "strix": "MANTIS_BOT_STRIX",
}


def resolve_bot(kind: str, *, depth: int | None, opponent_sims: int | None) -> BotFactory:
    """Resolve `kind` to a zero-arg `BotFactory`, or raise.

    Unknown kind -> `ValueError` naming the known set. A known external kind with no
    adapter installed at HEAD -> `RungUnresolvable` (never fatal to a round — the caller
    catches it per rung, `mantis.eval.rounds.resolve_ladder_rungs`).
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown bot kind {kind!r}; known kinds: {sorted(_KNOWN_KINDS)}")

    if opponent_sims is not None:
        _nsims_mod.resolve_eval_model_sims(kind, opponent_sims)

    if kind == "random":
        def _factory(seed: int = 0) -> RandomBot:
            return RandomBot(seed=seed)

        return _factory

    env_key = _ENV_KEYS[kind]
    env_value = os.environ.get(env_key)
    if env_value is None:
        raise RungUnresolvable(rung=kind, reason=f"env key {env_key} unset")
    raise RungUnresolvable(
        rung=kind,
        reason=(
            f"no adapter installed (WP12-R): {kind} (env key {env_key} is set but no "
            "WP12-R adapter package is installed at HEAD)"
        ),
    )


__all__ = ["BotFactory", "resolve_bot"]
