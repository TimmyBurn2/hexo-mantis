"""⊕ WP12-R Phase O / O-01 (R152) — `EvalBrokenReason` is the ONE eval-failure authority.

RED-at-import until IMPL writes `EvalBrokenReason` into `src/mantis/eval/errors.py`.
ORACLE-FIRST (⊕): the top-level import raises ImportError before any port code exists.

R152's first clause is "one authority = the reason enum". At HEAD an eval round's failure
reason is a bare `str` literal typed at six sites in `mantis/eval/pipeline.py` and copied
verbatim into the routed result's `error` key — so "which ways can an eval round break" has
no single answer any reader can enumerate, and a typo produces a NEW reason silently. This
file is the enumeration.

The two oracles and the defect each is the only witness to:

- `test_the_enum_declares_exactly_the_seven_censused_reasons` — the member set is EXACTLY
  the seven routes DESIGN_O §a.2 censused, with the wire spellings byte-identical to the
  literals HEAD already emits. Sole witness to an EIGHTH member arriving without a route
  (or a route losing its member): a `>=` census goes on passing while the surface drifts.
  It is also what forces a later author who adds a reason to touch this file, which is
  where the phase→reason map (O-04) and the per-route drive (O-02) are anchored.
  MUTATION THAT REDS IT (M-O1): rename any member's VALUE (e.g. `KILLED = "worker_killed"`).
  MUTATION THAT REDS IT (M-O3): collapse two members onto one value (`EXIT_NONZERO =
  "killed"`) — the member count survives, the value set does not.

- `test_a_member_is_its_own_wire_spelling_and_an_unregistered_spelling_is_refused` —
  `StrEnum`, not `Enum`: the value crosses a JSON round trip (the round-result mapping is
  consumed on the train side and serialized into the event stream), so a member that is not
  ITS OWN wire string would need a second member→string table, which is the duplicated
  authority R1 exists to kill. The refusal half is the runtime leg of §b.3's
  unrepresentability claim: pyright alone cannot stop a `# type: ignore`, so an unregistered
  spelling must be a loud `ValueError` at every parse boundary.
  MUTATION THAT REDS IT: make the enum a plain `Enum` (the json round trip stops returning
  the member's own spelling), or give it a `_missing_` hook that coerces an unknown value.
"""
from __future__ import annotations

import json
from enum import StrEnum

import pytest

# RED-at-import anchor: the name does not exist at HEAD.
from mantis.eval.errors import EvalBrokenReason

#: The censused routes (DESIGN_O §a.2), spelled EXACTLY as `mantis/eval/pipeline.py` emits them.
#: Byte-identity is load-bearing and deliberate: it keeps every existing event-stream assertion
#: pointing at the same values. Transcribed here on purpose — this file IS the independent
#: statement of the census, and deriving it from the enum under test would make the assertion
#: self-satisfying (R81).
#:
#: `round_timeout` JOINS THE CENSUS under the R316(c) frozen-file grant, and it is a route that
#: always existed under another route's name. `_poll_loop` kills a round that exceeds
#: `eval.round_timeout_sec` — a PROGRESS budget — and reported it as `join_timeout`, which names
#: the kill sequence rather than the cause; the 2026-08-27 re-sit's every in-run round ended that
#: way and the operator was told the child would not exit. `_drain_escalate`'s genuine join
#: timeout keeps the name. Adding the member is therefore NOT a new failure mode: it is one
#: existing mode ceasing to wear another's label, which is the stale-text defect class inside an
#: instrument.
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
    """O-01. Exact set equality in BOTH directions, member count derived from the census itself.

    An extra member means a reason with no producer (the phantom-input class LAW-07 exists
    to stop); a missing one means a live failure route whose reason is unrepresentable and
    therefore back to being a bare string.
    """
    members = list(EvalBrokenReason)
    # DERIVED from the census set, not a literal. A hard `== 7` had to be re-edited the first
    # time a route was correctly named (R316(c)) — and a count that must be re-edited on every
    # edit is the derive-or-delete class (G-DFIX-4 / R192(e)) sitting inside the oracle that
    # exists to catch it. The set below is still the INDEPENDENT statement of the census; only
    # its cardinality stops being transcribed twice.
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
    """O-01, second half — the JSON boundary and the LAW-11-shaped refusal.

    `StrEnum` members ARE their wire spelling, so there is no `.value`/member drift and no
    second table mapping members to strings. And `EvalBrokenReason(<unknown>)` RAISES: an
    absent/unregistered value is an ERROR, never a default (LAW-11's posture applied to the
    taxonomy). The composition root's re-parse (O-10) depends on this raise.
    """
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
