"""mantis.eval error taxonomy (design §a.3 errors.py) — one import surface.

Re-exports `MixedRegimeError`/`BookError`/`RungUnresolvable` alongside the eval-owned
errors so a consumer never needs to know which sub-package originally raised.
"""
from __future__ import annotations

from mantis.arena.books import BookError
from mantis.arena.regime import MixedRegimeError
from mantis.bots.protocol import RungUnresolvable


class EvalBrokenError(RuntimeError):
    """An eval round could not complete cleanly (join timeout / crash / garbage result)."""


class LadderStateError(RuntimeError):
    """`LadderState` persistence (save/load) failed — LAW-14, never a silent except."""


class EvalDecodeUnsupportedError(RuntimeError):
    """The round's declared encoding requires a decode this eval path does not implement.

    Raised once per round, at spec-resolution time, before any model is loaded — the
    alternative is reporting an eval result pooled differently from the encoding's own
    declaration, which is a plausible-looking number nobody can attribute (LAW-11's shape:
    the unimplemented case is an ERROR, never a silent approximation).
    """


class ResultContractError(RuntimeError):
    """The worker's sidecar result JSON does not satisfy the result-contract shape."""


__all__ = [
    "BookError",
    "EvalBrokenError",
    "EvalDecodeUnsupportedError",
    "LadderStateError",
    "MixedRegimeError",
    "ResultContractError",
    "RungUnresolvable",
]
