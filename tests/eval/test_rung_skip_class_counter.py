"""The fourth skip channel: a per-class skip counter emitted while the round is still going.

The three existing channels (event, ERROR log, `skipped_rungs` list) cannot tell an operator
whether the skips are the ones authorised or the box is misconfigured; a per-class count can.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mantis.bots.protocol import RungUnresolvable
from mantis.eval.pipeline import emit_rung_skip_events

_ROUND_ID = "r000001_100"

#: The excluded sealbot depth reaches `operator_authorized`; the other three are the three
#: ways a sealbot rung can fail to resolve.
_CLASSES = ("operator_authorized", "vendor_absent", "build_absent", "load_failed")


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]


def _reason(kind: str, *, monkeypatch: pytest.MonkeyPatch, root: Path | None,
            loader_raises: bool, depth: int | None = None) -> str:
    """Return one real refusal reason, derived from the shipped resolver rather than transcribed."""
    import mantis.bots.sealbot as sealbot_mod
    from mantis.bots.resolve import resolve_bot

    with monkeypatch.context() as patch:
        patch.setattr(sealbot_mod, "find_vendor_root", lambda: root)
        if loader_raises:
            def _explode() -> tuple[Any, Any]:
                raise ImportError("undefined symbol: _ZTIN8pybind116detail13type_casterE")

            patch.setattr(sealbot_mod, "load_sealbot_modules", _explode)
        with pytest.raises(RungUnresolvable) as exc:
            if depth is None and kind == "sealbot":
                depth = 5
            resolve_bot(kind, depth=depth, opponent_sims=128)
    return exc.value.reason


def _one_skip_per_class(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, str]]:
    """Four skip entries, one per class, in `_CLASSES` order."""
    from mantis.bots.resolve import _R326_EXCLUDED_SEALBOT_DEPTHS

    excluded = next(iter(_R326_EXCLUDED_SEALBOT_DEPTHS))
    return [
        {"rung": f"sealbot_d{excluded}",
         "reason": _reason("sealbot", monkeypatch=monkeypatch, root=None, loader_raises=False,
                           depth=excluded)},
        {"rung": "sealbot_d5",
         "reason": _reason("sealbot", monkeypatch=monkeypatch, root=None, loader_raises=False)},
        {"rung": "sealbot_d6",
         "reason": _reason("sealbot", monkeypatch=monkeypatch, root=tmp_path,
                           loader_raises=False)},
        {"rung": "sealbot_d7",
         "reason": _reason("sealbot", monkeypatch=monkeypatch, root=tmp_path,
                           loader_raises=True)},
    ]


@pytest.mark.parametrize("reason_class", _CLASSES)
def test_each_skip_reason_class_counts_itself_in_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reason_class: str
) -> None:
    """Prove each skip class counts itself in-run; one row per class, all four driven per row."""
    entries = _one_skip_per_class(monkeypatch, tmp_path)
    sink = _SpySink()
    emit_rung_skip_events(_ROUND_ID, entries, sink)

    counted = [e for e in sink.named("eval_rung_skip_class") if e["reason_class"] == reason_class]
    assert len(counted) == 1, (
        f"class {reason_class!r} produced {len(counted)} counter events for one skip; "
        f"stream={[(e['rung'], e['reason_class']) for e in sink.named('eval_rung_skip_class')]}"
    )
    assert counted[0]["class_count"] == 1
    assert counted[0]["round_id"] == _ROUND_ID
    assert counted[0]["rung"] == entries[_CLASSES.index(reason_class)]["rung"], (
        "the counter event must name the rung it counted; a class total with no rung cannot "
        "tell an operator WHICH rung fell into the misconfigured bucket"
    )


def test_the_class_set_is_closed_and_a_repeated_class_does_not_over_fire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prove the class set is closed and repeated skips of one class count 1 then 2, not per event."""
    from mantis.eval.pipeline import SKIP_REASON_CLASSES

    assert tuple(SKIP_REASON_CLASSES) == _CLASSES, (
        "the skip-class partition is CLOSED (DESIGN_A §2.7(4)): a reason that matched no "
        "class must be a loud failure, never a fifth bucket invented at emission time"
    )

    entries = _one_skip_per_class(monkeypatch, tmp_path)
    both_vendor_absent = [entries[1], {"rung": "sealbot_d6", "reason": entries[1]["reason"]}]
    sink = _SpySink()
    emit_rung_skip_events(_ROUND_ID, both_vendor_absent, sink)

    counter_events = sink.named("eval_rung_skip_class")
    totals = {c: [e for e in counter_events if e["reason_class"] == c] for c in _CLASSES}
    assert [e["class_count"] for e in totals["vendor_absent"]] == [1, 2], (
        f"two skips of one class must count 1 then 2 within the round; got "
        f"{[e['class_count'] for e in totals['vendor_absent']]}"
    )
    assert all(totals[c] == [] for c in _CLASSES if c != "vendor_absent"), (
        f"a class that did not fire must not appear: "
        f"{ {c: len(v) for c, v in totals.items()} }"
    )
    assert len(counter_events) == len(sink.named("eval_rung_skipped")), (
        "the counter must reach the sink ALONGSIDE each skip, one per rung — a single "
        "aggregate emitted at round end is exactly the 'log line somewhere' R164 ruled out"
    )
