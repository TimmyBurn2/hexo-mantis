"""RegimeKey — the canonical eval-game regime tag (A3; design §a.2 regime.py).

Every eval game record carries a `RegimeKey`; `mantis.eval.aggregate.aggregate_rung`
raises `MixedRegimeError` the instant more than one distinct key is pooled into one
aggregation call — A3's core invariant.
"""
from __future__ import annotations

from dataclasses import dataclass

_SEP = "|"


class MixedRegimeError(ValueError):
    """An aggregation call saw >1 distinct `regime_key` — never silently pooled (A3)."""


@dataclass(frozen=True)
class RegimeKey:
    """`(bot, variant, model_sims, opponent_spec, opening_book, deploy_matched, encoding)`.

    Equality/hash consider EVERY field (the dataclass default); `canonical()` is a
    stable `|`-joined string form used as the wire/record tag.
    """

    bot: str
    variant: str
    model_sims: int
    opponent_spec: str
    opening_book: str
    deploy_matched: bool
    encoding: str

    def canonical(self) -> str:
        parts = (
            self.bot, self.variant, str(self.model_sims), self.opponent_spec,
            self.opening_book, "1" if self.deploy_matched else "0", self.encoding,
        )
        for part in parts:
            if _SEP in part:
                raise ValueError(
                    f"RegimeKey field contains the canonical separator {_SEP!r}: {part!r}"
                )
        return _SEP.join(parts)


__all__ = ["MixedRegimeError", "RegimeKey"]
