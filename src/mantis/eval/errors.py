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


class ResultContractError(RuntimeError):
    """The worker's sidecar result JSON does not satisfy the result-contract shape."""


__all__ = [
    "BookError",
    "EvalBrokenError",
    "LadderStateError",
    "MixedRegimeError",
    "ResultContractError",
    "RungUnresolvable",
]
