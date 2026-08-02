"""⊕ WP12-R Phase A / O-A15 (DESIGN_A §2.7, PREREG_A §1) — the fourth skip channel.

Skips are already three-channel at HEAD: an `eval_rung_skipped` event (`pipeline.py:161-163`,
registered at `event_manifest.md:93`), an ERROR log line (`:164-165`), and the round result's
own `skipped_rungs` list (`rounds.py:218`). None of the three answers the question R143 says
must be legible: *are these the four skips the operator AUTHORISED, or is the box
misconfigured?* Four skips and six skips are both "some rungs skipped" to every existing
consumer, and the difference is the whole of run5's external instrument.

R164 made LAW-18 mean IN-RUN FIRE-RATE, not "there is a log line somewhere". So the fourth
channel is a per-class counter that reaches the injected sink alongside each skip, while the
round is still going.

The defect each row is the ONLY witness to:

- **the four class rows** — a class that stops counting. Each drives ALL FOUR classes in one
  emission and asserts on ONE of them, so M-A7 (delete the `vendor_absent` increment) reds
  exactly the `vendor_absent` parametrization and the other three stay green `[absent]`.
  A single row asserting "four counter events arrived" would be GREEN under a mutation that
  mislabelled every one of them.
- **the no-over-fire row** — a counter that fires per EVENT rather than per class, or a
  classifier whose set is open. Two skips of one class must read 2 on that class and 0 on
  the other three, and the class set is CLOSED: a reason nothing recognises must not
  silently become a fifth bucket.

**Not a `producer_manifest.yaml` monitor input** (PREREG_A §1, C-19): it is an event-stream
counter with no consuming rule, and registering it as a manifest input would create a
producer-without-consumer — the mirror of the LAW-08 hazard DESIGN_A §2.2(4) exists to
avoid. If a later phase wants it as a gate input, that is a new row with its own consumer.

SEAM (frozen here, ORACLE-FIRST):
  * `mantis.eval.pipeline.SKIP_REASON_CLASSES: tuple[str, ...]` — the CLOSED class set.
  * `emit_rung_skip_events(round_id, skipped, sink)` emits, per skipped rung and through the
    SAME injected sink (SR-4), one `eval_rung_skip_class` event carrying `round_id`, `rung`,
    `reason_class` and `class_count` (the running count for that class within the round).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mantis.bots.protocol import RungUnresolvable
from mantis.eval.pipeline import emit_rung_skip_events

_ROUND_ID = "r000001_100"

#: DESIGN_A §2.7(4). `operator_authorized` is the kraken/strix skip R139 ruled; the other
#: three are the three ways a sealbot rung can fail to resolve, and telling them apart is
#: the entire point — "4 rungs skipped as ruled" versus "6 rungs skipped because the box is
#: misconfigured", WHILE the run is going.
_CLASSES = ("operator_authorized", "vendor_absent", "build_absent", "load_failed")


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]


def _reason(kind: str, *, monkeypatch: pytest.MonkeyPatch, root: Path | None,
            loader_raises: bool) -> str:
    """One REAL refusal reason, taken from the shipped resolver rather than transcribed.

    Deriving the reasons instead of hard-coding them is what makes this file two-sided: a
    resolver whose wording drifted out of the classifier's reach reds HERE, at the channel
    that has to classify it, rather than silently collapsing two classes into one.
    """
    import mantis.bots.sealbot as sealbot_mod
    from mantis.bots.resolve import resolve_bot

    with monkeypatch.context() as patch:
        patch.setattr(sealbot_mod, "find_vendor_root", lambda: root)
        if loader_raises:
            def _explode() -> tuple[Any, Any]:
                raise ImportError("undefined symbol: _ZTIN8pybind116detail13type_casterE")

            patch.setattr(sealbot_mod, "load_sealbot_modules", _explode)
        with pytest.raises(RungUnresolvable) as exc:
            resolve_bot(kind, depth=5 if kind == "sealbot" else None, opponent_sims=128)
    return exc.value.reason


def _one_skip_per_class(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, str]]:
    """Four skip entries, one per class, in `_CLASSES` order."""
    return [
        {"rung": "kraken_raw",
         "reason": _reason("kraken", monkeypatch=monkeypatch, root=None, loader_raises=False)},
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
    """O-A15, one row per class. TWO-SIDED by construction: the same emission carries all
    four classes, so a mutation that deleted ONE increment is attributable to that class and
    a mutation that broke all four is still distinguishable from it."""
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
    """O-A15's no-over-fire row, plus the closed-set conjunct.

    FIRING ORDER: the closed-set assertion runs FIRST (a classifier that grew a fifth bucket
    invalidates every count below it), then the per-class totals, then the pairing assertion.
    """
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
