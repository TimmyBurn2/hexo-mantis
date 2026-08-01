"""mantis.eval error taxonomy (design §a.3 errors.py) — one import surface.

Re-exports `MixedRegimeError`/`BookError`/`RungUnresolvable` alongside the eval-owned
errors so a consumer never needs to know which sub-package originally raised.
"""
from __future__ import annotations

from enum import StrEnum

from mantis.arena.books import BookError
from mantis.arena.regime import MixedRegimeError
from mantis.bots.protocol import RungUnresolvable


class EvalBrokenReason(StrEnum):
    """WHY an eval round broke — the ONE authority (WP12-R Phase O, R152).

    Seven members, one per censused failure route in `mantis.eval.pipeline`, with wire
    spellings BYTE-IDENTICAL to the bare literals this taxonomy replaces: the change is a
    SHAPE change (a typed value that only this enum can author) and never a VALUE change,
    so every event-stream reason already in the ONE channel keeps its spelling.

    `StrEnum` and not `Enum`: the value crosses a JSON round trip (the round-result mapping
    is consumed on the train side and serialized into the event stream), so a member that
    was not its own wire string would need a second member→string table — the duplicated
    authority R1 exists to kill. Re-parsing an unregistered spelling
    (`EvalBrokenReason(raw)`) RAISES `ValueError`, which is what makes a reason no member
    spells loud at the process boundary instead of a silent clean exit (LAW-11's posture
    applied to the taxonomy; `mantis.run.compose_run` is the consumer).

    `phase` is deliberately NOT folded in: it is a FUNCTION of the reason, stays on the
    event payload, and the map is pinned by
    `tests/eval/test_eval_broken_reason_routes.py::test_the_reason_to_phase_map_is_fixed`.
    There is no member for a healthy drain either — the ABSENCE of a reason (`None`) is the
    clean state, and a second value saying so would be the R79 shape this taxonomy deletes.
    """

    JOIN_TIMEOUT = "join_timeout"
    KILLED = "killed"
    EXIT_NONZERO = "exit_nonzero"
    RESULT_MISSING = "result_missing"
    RESULT_INVALID = "result_invalid"
    LADDER_PERSIST_FAILED = "ladder_persist_failed"
    ROUND_COMPLETION_ERROR = "round_completion_error"


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
    "EvalBrokenReason",
    "EvalDecodeUnsupportedError",
    "LadderStateError",
    "MixedRegimeError",
    "ResultContractError",
    "RungUnresolvable",
]
