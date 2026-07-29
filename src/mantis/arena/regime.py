"""RegimeKey — the canonical eval-game regime tag (A3; design §a.2 regime.py).

Every eval game record carries a `RegimeKey`; `mantis.eval.aggregate.aggregate_rung`
raises `MixedRegimeError` the instant more than one distinct key is pooled into one
aggregation call — A3's core invariant.
"""
from __future__ import annotations

from dataclasses import dataclass

_SEP = "|"
_N_FIELDS = 7


class MixedRegimeError(ValueError):
    """An aggregation call saw >1 distinct `regime_key` — never silently pooled (A3)."""


@dataclass(frozen=True)
class RegimeKey:
    """`(bot, variant, model_sims, opponent_spec, opening_book, deploy_matched, encoding)`.

    Equality/hash consider EVERY field (the dataclass default); `canonical()` is a
    stable, round-trippable `|`-joined string form used as the wire/record tag.
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

    @classmethod
    def from_canonical(cls, canonical: str) -> RegimeKey:
        parts = canonical.split(_SEP)
        if len(parts) != _N_FIELDS:
            raise ValueError(
                f"malformed RegimeKey canonical form (expected {_N_FIELDS} fields): {canonical!r}"
            )
        bot, variant, model_sims, opponent_spec, opening_book, deploy_matched, encoding = parts
        return cls(
            bot=bot, variant=variant, model_sims=int(model_sims),
            opponent_spec=opponent_spec, opening_book=opening_book,
            deploy_matched=(deploy_matched == "1"), encoding=encoding,
        )


__all__ = ["MixedRegimeError", "RegimeKey"]
