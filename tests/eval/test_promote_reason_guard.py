"""⊕ WP12-R Phase O / O-12 (R152/LAW-11) — `apply_gate_decision` reads the reason FIRST and
UNCONDITIONALLY, and an ABSENT reason is an error rather than "assume clean".

RED-at-HEAD on its own mechanism for three of four arms (⊕): no module-level import anchor
is used, so the reds below are evidence about the guard, not about a missing module. The one
arm that needs an `EvalBrokenReason` member imports it inside its own body, so a missing
enum reds THAT arm and no other.

`promote.py:43` is the ONLY production consumer of "was this round broken?" in all of `src/`
(DESIGN_O §a.3, verified). At HEAD it reads `result.get("eval_broken")` — a silent-`None`-is
-falsy read, so a result mapping that never carried the key at all is indistinguishable from
one that carried `False`. That is the stale-fixture class: a hand-built or half-migrated
round mapping promotes, and nothing anywhere says a decision was taken on an absent fact.

Two things must therefore hold, and they are DIFFERENT things:

  1. The read is a SUBSCRIPT, not a `.get` — an absent `eval_broken_reason` raises
     `KeyError`. This is LAW-11's posture one layer over: absent is an ERROR, never a
     default.
  2. The read comes FIRST in the `or`. This is not style. With
     `if not result.get("promoted") or result["eval_broken_reason"] is not None:` Python's
     `or` short-circuits on a NON-promoted mapping and the reason is never read — so exactly
     the stale fixture the guard claims to catch sails through. The promoted arm alone
     cannot see that (`not True` is False, so the second operand IS evaluated); only the
     non-promoted arm can.

The four arms, and what each is the ONLY witness to:

- arm (a) `..._absent_reason_on_a_promoted_result_raises` — sole witness to the `.get` →
  subscript change on the path that would actually have promoted. MUTATION (M-O12).
- arm (b) `..._absent_reason_on_a_NON_promoted_result_also_raises` — sole witness to the
  OPERAND ORDER. MUTATION (M-O12b): reorder the two operands; arm (a) stays GREEN and only
  this arm reds. A promoted-only oracle is blind to it.
- arm (c) `..._a_present_reason_refuses_to_promote` — the guard still does its original job.
  Sole witness that the reason (not the deleted `eval_broken` bool) is what vetoes.
- arm (d) `..._a_clean_round_still_promotes_and_the_reason_was_actually_read` — the
  over-fire direction, plus the ACCESS assertion that makes arm (d) mean something: a guard
  that never reads the reason at all also promotes a clean round, so without the access
  record this arm would pass at HEAD for precisely the wrong reason.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.promote import DeployTagHooks, apply_gate_decision


class _RecordingResult(dict):
    """A round-result mapping that records WHICH keys the guard actually looked at.

    The access log is what turns "a clean round promotes" from a statement that is true of
    a guard which reads nothing into a statement about a guard which read the reason and
    then promoted (R81: the assertion must not be satisfiable by the absence of the thing
    under test). `get` is overridden alongside `__getitem__` so a `.get`-shaped read is
    recorded too — otherwise M-O12 would restore the silent read and leave this file green.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: list[str] = []

    def __getitem__(self, key: Any) -> Any:
        self.accessed.append(str(key))
        return super().__getitem__(key)

    def get(self, key: Any, *args: Any) -> Any:      # noqa: A003 -- Mapping's own name
        self.accessed.append(str(key))
        return super().get(key, *args)


class _RecordingHooks:
    """The deploy-tag collaborators, recording. A promotion is observable ONLY as a
    `save_anchor` call plus the returned step — nothing else in this module has a side
    effect a test can read."""

    def __init__(self, tmp_path: Path) -> None:
        self.saved: list[dict] = []
        self.loaded: list[dict] = []
        self.anchor = SimpleNamespace(best_model=object(), best_model_step=None)
        self.hooks = DeployTagHooks(
            anchor_state=self.anchor,
            best_model_path=tmp_path / "best_model.pt",
            run_id="oracle_test_run",
            encoding="v6_live2_ls",
            save_anchor=self._save_anchor,
            guarded_load=self._guarded_load,
        )

    def _save_anchor(self, model: Any, path: Any, **kwargs: Any) -> None:
        self.saved.append({"path": str(path), **kwargs})

    def _guarded_load(self, model: Any, state_dict: Any) -> None:
        self.loaded.append(dict(state_dict))


def _result(**fields: Any) -> _RecordingResult:
    """A round-result mapping with the fields the guard and the promotion path read. No
    `eval_broken_reason` unless the caller supplies one — its ABSENCE is the subject of two
    of the four arms, so it is never defaulted in here."""
    base = {"step": 7, "round_id": "r000001_7", "wr_sealbot": 0.6}
    base.update(fields)
    return _RecordingResult(base)


# ══ arm (a) — absent reason, promoted ══════════════════════════════════════════════════
def test_an_absent_reason_on_a_promoted_result_raises(tmp_path) -> None:
    """O-12 arm (a). The promoted path is the one with consequences: a stale mapping that
    never carried the reason must not advance the deploy tag off a fact nobody supplied.

    MUTATION THAT REDS IT (M-O12): restore `result.get("eval_broken_reason")`. The guard
    then reads `None`, treats it as clean, and promotes — silently, which is the whole
    defect."""
    rig = _RecordingHooks(tmp_path)
    result = _result(promoted=True)

    with pytest.raises(KeyError, match="eval_broken_reason"):
        apply_gate_decision(rig.hooks, result)

    assert rig.saved == [], (
        "…and nothing may have been written before the raise: a half-applied promotion is "
        f"worse than a refused one. Got {rig.saved}"
    )


# ══ arm (b) — absent reason, NOT promoted (the operand-order arm) ══════════════════════
def test_an_absent_reason_on_a_NON_promoted_result_also_raises(tmp_path) -> None:
    """O-12 arm (b) — the arm a promoted-only oracle cannot see (REVIEW note N3).

    Behaviourally this round was not going to promote either way, so it looks harmless. It
    is not: it is the ONLY observation that distinguishes "the reason is read
    unconditionally" from "the reason is read when Python happens to get that far". If the
    guard is written `if not result.get("promoted") or result["eval_broken_reason"] is not
    None:` then `or` short-circuits here and the stale mapping passes through in silence —
    and the next stale mapping, the one that DOES carry `promoted=True`, is arm (a)'s.

    MUTATION THAT REDS IT (M-O12b): that exact reorder. Arm (a) stays GREEN under it."""
    rig = _RecordingHooks(tmp_path)
    result = _result(promoted=False)

    with pytest.raises(KeyError, match="eval_broken_reason"):
        apply_gate_decision(rig.hooks, result)

    assert "eval_broken_reason" in result.accessed, (
        "the reason must be read FIRST and unconditionally; on this mapping it was never "
        f"read at all. Keys the guard looked at, in order: {result.accessed}"
    )


# ══ arm (c) — a present reason vetoes ══════════════════════════════════════════════════
def test_a_present_reason_refuses_to_promote(tmp_path) -> None:
    """O-12 arm (c). The guard's original job, re-pointed onto the one authority: a round
    that broke does not advance the deploy tag, and what says it broke is the REASON, not a
    boolean beside it.

    The enum is imported HERE rather than at module scope on purpose: this is the only arm
    that needs a member, so a missing `EvalBrokenReason` reds this node alone and leaves the
    other three reddening (or passing) on their own mechanism."""
    from mantis.eval.errors import EvalBrokenReason

    rig = _RecordingHooks(tmp_path)
    result = _result(promoted=True, eval_broken_reason=EvalBrokenReason.RESULT_INVALID)

    assert apply_gate_decision(rig.hooks, result) is None, (
        "a broken round must return None (no promoted step)"
    )
    assert rig.saved == [] and rig.loaded == [], (
        f"…and must touch neither the anchor nor the loader. saved={rig.saved} "
        f"loaded={rig.loaded}"
    )


# ══ arm (d) — a clean round still promotes, and the reason WAS read ════════════════════
def test_a_clean_round_still_promotes_and_the_reason_was_actually_read(tmp_path) -> None:
    """O-12 arm (d) — the over-fire direction, with the anti-vacuity assertion that makes it
    an oracle instead of a tautology.

    "A clean round promotes" is also true of a guard that reads nothing at all, which is
    exactly the state of the tree before this phase. The access record is what separates the
    two: the reason must have been LOOKED AT and found to be `None`.

    MUTATION THAT REDS IT: veto on `reason is None` (inverted sense), or drop the reason
    read entirely (the access assertion catches the second one; nothing else here would).
    """
    rig = _RecordingHooks(tmp_path)
    result = _result(promoted=True, eval_broken_reason=None)

    promoted_step = apply_gate_decision(rig.hooks, result)

    assert "eval_broken_reason" in result.accessed, (
        "the guard promoted WITHOUT ever reading the reason — a decision taken on a fact it "
        f"never consulted. Keys the guard looked at, in order: {result.accessed}"
    )
    assert promoted_step == 7, (
        f"a clean promoted round advances the deploy tag to its step; got {promoted_step!r}"
    )
    assert len(rig.saved) == 1 and rig.saved[0]["step"] == 7, (
        f"…through exactly one `save_anchor` call carrying that step; got {rig.saved}"
    )
    assert rig.anchor.best_model_step == 7, (
        "…and the resolved anchor's recorded step follows it; got "
        f"{rig.anchor.best_model_step!r}"
    )
