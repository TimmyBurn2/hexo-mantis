"""Eval opponent model_sims resolver (REBUILD of frozen resolve/nsims.py).

The frozen code-side {random:96, sealbot:128} default dict DIES: the per-opponent value is a
required schema field (eval.random_model_sims / eval.sealbot_model_sims) the resolver READS.
Unknown opponent → ValueError (no silent fallback); a None value → ValueError (Δ-REBUILD —
the config field is required, there is no code default to fall to).
"""
from __future__ import annotations

# The known eval opponents (name authority; NOT a value default — the values live in the config).
# WP11-A extends this from ("random", "sealbot") to add the ladder's kraken/strix kinds
# (design §a.4) — semantics unchanged: the config value always wins, None still raises.
_KNOWN_OPPONENTS = ("random", "sealbot", "kraken", "strix")


def resolve_eval_model_sims(opponent: str, cfg_value: int | None) -> int:
    """Resolve eval ``model_sims`` for ``opponent`` from the config value.

    The config value always wins. Unknown opponent → ValueError (the caller named an opponent
    with no defined sims field). ``cfg_value is None`` → ValueError (required field absent —
    no code-side default, Δ-REBUILD vs frozen's 96/128 dict).
    """
    if opponent not in _KNOWN_OPPONENTS:
        raise ValueError(
            f"unknown eval opponent {opponent!r}; known: {list(_KNOWN_OPPONENTS)}"
        )
    if cfg_value is None:
        raise ValueError(
            f"eval model_sims for {opponent!r} is a required config field, got None "
            "(no code-side default — declare eval.{opponent}_model_sims explicitly)".replace(
                "{opponent}", opponent
            )
        )
    return int(cfg_value)
