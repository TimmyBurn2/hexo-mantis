"""Prove `apply_gate_decision` reads the broken-reason first and unconditionally.

Two separate things must hold: the read is a SUBSCRIPT, so an absent reason raises rather than
reading as clean; and it is the FIRST operand of the `or`, because otherwise a non-promoted
mapping short-circuits and the stale fixture the guard exists to catch sails through.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.promote import DeployTagHooks, apply_gate_decision


class _RecordingResult(dict):
    """Record which keys the guard actually looked at.

    Without the log, "a clean round promotes" is also true of a guard that reads nothing. `get`
    is overridden beside `__getitem__` so a restored `.get`-shaped silent read is recorded too.
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
    """Record the deploy-tag collaborators; a promotion is observable only as a `save_anchor`
    call plus the returned step."""

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
    """Build a round-result mapping; `eval_broken_reason` is never defaulted in, because its
    absence is the subject of two arms."""
    base = {"step": 7, "round_id": "r000001_7", "wr_sealbot": 0.6}
    base.update(fields)
    return _RecordingResult(base)


def test_an_absent_reason_on_a_promoted_result_raises(tmp_path) -> None:
    """Prove an absent reason on a promoted result raises instead of advancing the deploy tag.

    Killer: restore `result.get("eval_broken_reason")` — the guard then reads `None`, treats it
    as clean, and promotes silently.
    """
    rig = _RecordingHooks(tmp_path)
    result = _result(promoted=True)

    with pytest.raises(KeyError, match="eval_broken_reason"):
        apply_gate_decision(rig.hooks, result)

    assert rig.saved == [], (
        "…and nothing may have been written before the raise: a half-applied promotion is "
        f"worse than a refused one. Got {rig.saved}"
    )


def test_an_absent_reason_on_a_NON_promoted_result_also_raises(tmp_path) -> None:
    """Prove an absent reason raises on a NON-promoted result too — the only observation that
    separates "read unconditionally" from "read when Python gets that far".

    Killer: put the `promoted` test first in the `or`. The promoted-only arm stays green.
    """
    rig = _RecordingHooks(tmp_path)
    result = _result(promoted=False)

    with pytest.raises(KeyError, match="eval_broken_reason"):
        apply_gate_decision(rig.hooks, result)

    assert "eval_broken_reason" in result.accessed, (
        "the reason must be read FIRST and unconditionally; on this mapping it was never "
        f"read at all. Keys the guard looked at, in order: {result.accessed}"
    )


def test_a_present_reason_refuses_to_promote(tmp_path) -> None:
    """Prove a present reason vetoes promotion, with the reason as the one authority.

    The enum is imported in-body so a missing member reds this arm alone.
    """
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


def test_a_clean_round_still_promotes_and_the_reason_was_actually_read(tmp_path) -> None:
    """Prove a clean round still promotes AND that the reason was actually looked at.

    Killer: invert the veto sense, or drop the reason read entirely — only the access assertion
    catches the second.
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
