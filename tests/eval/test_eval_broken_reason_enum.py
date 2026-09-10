"""`EvalBrokenReason` is the ONE eval-failure authority.

Before it, a round's failure reason was a bare `str` literal typed at six sites in
`mantis/eval/pipeline.py` and copied verbatim into the routed result, so "which ways can an eval
round break" had no answer a reader could enumerate and a typo produced a NEW reason silently.

The member set is EXACTLY the censused routes, with wire spellings byte-identical to the
literals already emitted, and the enum is a `StrEnum` so a member IS its wire spelling: anything
else needs a second member-to-string table. An unregistered spelling raises, which is what the
composition root's re-parse depends on.
"""
from __future__ import annotations

import json
from enum import StrEnum

import pytest

from mantis.eval.errors import EvalBrokenReason

#: The censused routes, spelled EXACTLY as `mantis/eval/pipeline.py` emits them. Transcribed on
#: purpose — this file IS the independent statement of the census, and deriving it from the enum
#: under test would make the assertion self-satisfying.
#:
#: `round_timeout` is a route that always existed under another route's name: a round killed for
#: exceeding `eval.round_timeout_sec` (a PROGRESS budget) was reported as `join_timeout`, which
#: names the kill sequence rather than the cause. It is one existing mode ceasing to wear
#: another's label, not a new failure mode.
_CENSUSED_REASONS = {
    "join_timeout",
    "round_timeout",
    "killed",
    "exit_nonzero",
    "result_missing",
    "result_invalid",
    "ladder_persist_failed",
    "round_completion_error",
}


def test_the_enum_declares_exactly_the_seven_censused_reasons() -> None:
    """Exact set equality in BOTH directions, member count derived from the census itself.

    An extra member is a reason with no producer; a missing one is a live failure route whose
    reason is unrepresentable and therefore back to being a bare string.
    """
    members = list(EvalBrokenReason)
    # DERIVED from the census set, not a literal: a hard `== 7` had to be re-edited the first
    # time a route was correctly named, and a count re-edited on every edit is the defect class
    # this oracle exists to catch.
    assert len(members) == len(_CENSUSED_REASONS), (
        f"the taxonomy is the censused routes (DESIGN_O §a.2); got {len(members)}: "
        f"{[m.name for m in members]}"
    )
    values = {member.value for member in members}
    assert values == _CENSUSED_REASONS, (
        "the enum's VALUES must be the HEAD literals byte-for-byte — a spelling change "
        "silently re-labels every event-stream reason already in the ONE channel.\n"
        f"  missing from the enum: {sorted(_CENSUSED_REASONS - values)}\n"
        f"  present but uncensused: {sorted(values - _CENSUSED_REASONS)}"
    )
    assert len(values) == len(members), (
        "two members sharing one value collapse two distinguishable failures into one "
        f"observable; got {sorted((m.name, m.value) for m in members)}"
    )


def test_a_member_is_its_own_wire_spelling_and_an_unregistered_spelling_is_refused() -> None:
    """The JSON boundary and the refusal. `StrEnum` members ARE their wire spelling, so there is
    no member/value drift and no second table; `EvalBrokenReason(<unknown>)` RAISES, because an
    unregistered value is an ERROR, never a default."""
    assert issubclass(EvalBrokenReason, StrEnum), (
        "the reason crosses a JSON round trip; a non-str enum would need a second "
        "member→wire table, which is the duplicated-authority shape R1 kills"
    )
    for member in EvalBrokenReason:
        assert isinstance(member, str) and member == member.value
        round_tripped = json.loads(json.dumps({"reason": member}))["reason"]
        assert round_tripped == member.value, (
            f"{member.name} does not survive a JSON round trip as its own spelling: "
            f"{round_tripped!r} != {member.value!r}"
        )
        assert EvalBrokenReason(member.value) is member, (
            f"re-parsing {member.value!r} must return the SAME member (the root's re-parse "
            "is what makes an unregistered spelling loud)"
        )

    with pytest.raises(ValueError, match="not_a_registered_reason"):
        EvalBrokenReason("not_a_registered_reason")
