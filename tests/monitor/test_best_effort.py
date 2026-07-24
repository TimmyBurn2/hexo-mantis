"""O-09 / P-09 — `best_effort()` with a MANDATORY named counter (LAW-14).

The ONLY sanctioned optional-effect path in `monitor/**` (the other outcome is the
except-pass census, O-20). RED-at-import until IMPL writes `mantis.monitor.best_effort`:
this file's top-level import raises ModuleNotFoundError, which is the oracle-first proof.

PASS bars (PREREG P-09):
  * omitting `counters` ⇒ TypeError (the mandatory counter is enforced BY SIGNATURE —
    keyword-only, no default);
  * a failing fn ⇒ the named counter +1, WARN logged, returns (False, None), NEVER raises;
  * success ⇒ (True, value), counter unchanged.
A swallowed failure with counter +0 is the exact LAW-14 bug this bites.
"""
from __future__ import annotations

import logging

import pytest

from mantis.monitor.best_effort import BestEffortCounters, best_effort


def test_success_returns_true_value_and_leaves_counter_unchanged() -> None:
    """P-09 — a succeeding fn returns (True, value); the label counter is NOT incremented."""
    counters = BestEffortCounters()
    ok, value = best_effort("snapshot", lambda: 41 + 1, counters=counters)
    assert ok is True
    assert value == 42
    assert counters.get("snapshot") == 0


def test_failure_counts_warns_and_returns_false_none() -> None:
    """P-09 — a raising fn ⇒ counter +1, (False, None), and NO exception propagates."""
    counters = BestEffortCounters()

    def boom() -> int:
        raise OSError("wedged FS")

    ok, value = best_effort("snapshot", boom, counters=counters)  # must not raise
    assert ok is False
    assert value is None
    assert counters.get("snapshot") == 1


def test_failure_logs_a_warning(caplog) -> None:
    """P-09 — the failure path emits a WARN (a swallowed-silent failure is banned)."""
    counters = BestEffortCounters()
    with caplog.at_level(logging.WARNING):
        best_effort("snapshot", lambda: (_ for _ in ()).throw(RuntimeError("x")),
                    counters=counters)
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records), (
        "best_effort must WARN on failure, never swallow silently"
    )


def test_counters_is_mandatory_keyword_only() -> None:
    """P-09 — calling WITHOUT `counters` is a TypeError: the mandatory counter is enforced by
    the signature (keyword-only, no default). A default-None counters would silently permit an
    uncounted optional effect — the LAW-14 hole."""
    with pytest.raises(TypeError):
        best_effort("snapshot", lambda: 1)  # type: ignore[call-arg]


def test_repeated_failures_accumulate_per_label() -> None:
    """P-09 — the counter is per-label and monotonic; two labels are tracked independently."""
    counters = BestEffortCounters()
    for _ in range(3):
        best_effort("a", lambda: (_ for _ in ()).throw(ValueError()), counters=counters)
    best_effort("b", lambda: (_ for _ in ()).throw(ValueError()), counters=counters)
    assert counters.get("a") == 3
    assert counters.get("b") == 1
    snap = counters.snapshot()
    assert snap["a"] == 3 and snap["b"] == 1


def test_counters_get_unknown_label_is_zero() -> None:
    """A never-incremented label reads 0 (not a KeyError) — the registry is total."""
    counters = BestEffortCounters()
    assert counters.get("never_touched") == 0
