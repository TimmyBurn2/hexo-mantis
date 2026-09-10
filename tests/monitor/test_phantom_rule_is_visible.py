"""A rule that has never been able to fire says so, and the registry-handshake skip is a WARNING.

Two instruments, one class: something that never ran was indistinguishable from something that
ran and found nothing.

`check_selfplay_entropy_collapse` reads `selfplay_model_entropy_batch`, which no producer in
`src/` writes, and falls back to `policy_entropy_selfplay`, which has none either — so the rule
has never once been able to fire, while sitting in `WARN_RULE_NAMES` with no per-rule skip
count. `mantis.encoding._registry_sha_handshake` returned after an INFO line when no on-disk
`registry.toml` was found, and `mantis.run` installs no logging handler, so lastResort drops
INFO; the skip means the stale-`.so` guard DID NOT RUN, a statement about the run's provenance.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

import mantis.monitor.rules as rules
from mantis.monitor.config import MonitorConfig


@pytest.fixture(autouse=True)
def _reset_counters() -> Any:
    """The counters are module state; each row starts from zero and leaves it there."""
    before = dict(rules.WARN_RULE_SKIPS)
    for key in rules.WARN_RULE_SKIPS:
        rules.WARN_RULE_SKIPS[key] = 0
    yield
    rules.WARN_RULE_SKIPS.update(before)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))


def _run(payload: dict[str, Any]) -> list[str]:
    return rules.emit_training_step_alerts(payload, MonitorConfig(), [], sink=_Sink())



def test_every_WARN_rule_declares_the_input_its_verdict_depends_on() -> None:
    """The mapping is the thing that makes 'did not run' expressible at all, so it must cover
    the rule set exactly — a rule missing from it would go back to being silently quiet."""
    assert set(rules.WARN_RULE_INPUTS) == set(rules.WARN_RULE_NAMES)
    assert set(rules.WARN_RULE_SKIPS) == set(rules.WARN_RULE_NAMES)


def test_the_selfplay_entropy_rule_counts_itself_as_UNABLE_TO_RUN() -> None:
    """THE PIN. A production `training_step` payload carries neither of this rule's inputs."""
    _run({"loss_total": 1.0, "grad_norm": 0.5, "policy_entropy": None,
          "policy_entropy_selfplay": None, "step": 1})
    assert rules.WARN_RULE_SKIPS["selfplay_entropy_collapse"] == 1


def test_the_count_ACCUMULATES_so_a_permanently_dead_rule_is_visible() -> None:
    """One skip is a quiet step; a hundred is a rule with no producer, and the number is what
    tells them apart."""
    for step in range(5):
        _run({"loss_total": 1.0, "grad_norm": 0.5, "step": step})
    assert rules.WARN_RULE_SKIPS["selfplay_entropy_collapse"] == 5


def test_a_rule_WITH_its_input_is_not_counted_as_skipped() -> None:
    """The control: a rule that ran and found nothing wrong must not be counted absent, or
    the number stops meaning anything."""
    _run({"loss_total": 1.0, "grad_norm": 0.5, "selfplay_model_entropy_batch": 3.0, "step": 1})
    assert rules.WARN_RULE_SKIPS["selfplay_entropy_collapse"] == 0


def test_the_entropy_rule_is_counted_absent_too_now_that_policy_entropy_is_None() -> None:
    """`policy_entropy` used to be a fabricated `0.0`, so `entropy_collapse` FIRED every step.
    It is `None` now, and absence must be counted or the repair trades a false alarm for a new
    silence."""
    _run({"loss_total": 1.0, "grad_norm": 0.5, "step": 1})
    assert rules.WARN_RULE_SKIPS["entropy_collapse"] == 1


def test_the_window_rule_has_no_payload_input_and_is_never_counted_absent() -> None:
    """It reads the caller's loss window, so there is nothing about the payload that could
    stop it running — declaring it as absent would be a fabricated skip."""
    assert rules.WARN_RULE_INPUTS["loss_increase_window"] == ()
    _run({"step": 1})
    assert rules.WARN_RULE_SKIPS["loss_increase_window"] == 0


def test_a_rule_that_FIRES_is_still_counted_as_having_run() -> None:
    """The two channels are orthogonal: firing and skipping are different events and a rule
    can never be both in one step."""
    fired = _run({"loss_total": 1.0, "grad_norm": 0.5, "policy_entropy": 0.1, "step": 1})
    assert "policy entropy" in " ".join(fired)
    assert rules.WARN_RULE_SKIPS["entropy_collapse"] == 0


def test_the_coordinator_publishes_the_counts_in_the_ONE_channel() -> None:
    """A counter nothing reads is the phantom-gate class one layer up."""
    import inspect

    from mantis.train.coordinator.step import StepCoordinator

    source = inspect.getsource(StepCoordinator._emit_monitor_gates)
    assert "warn_rule_skipped_absent" in source
    assert "WARN_RULE_SKIPS" in source



def test_a_skipped_handshake_logs_at_WARNING_not_INFO(
    caplog: pytest.LogCaptureFixture, tmp_path: Any
) -> None:
    """THE PIN. At INFO the line is dropped entirely by a run with no logging handler, and
    the skip means the stale-extension guard did not run at all."""
    from mantis.encoding import _registry_sha_handshake

    with caplog.at_level(logging.WARNING, logger="mantis.encoding"):
        _registry_sha_handshake(tmp_path / "definitely-absent.toml")
    assert caplog.records, "the skip produced no WARNING-level record"
    record = caplog.records[-1]
    assert record.levelno == logging.WARNING
    assert "SKIPPED" in record.getMessage()
    assert "WITHOUT COMPARING" in record.getMessage(), (
        "the message must say what did NOT happen, not only that something was skipped"
    )


def test_the_skip_is_readable_as_STATE_not_only_as_a_log_line() -> None:
    """A log line reaches a terminal or it does not; `handshake_ran()` is the fact itself, so a
    composition root can publish it into the event stream."""
    import mantis.encoding as enc

    assert callable(enc.handshake_ran)
    assert isinstance(enc.handshake_skipped, list)
    # In this repo layout the on-disk registry IS found, so the import-time handshake really
    # compared a sha. The row above calls the handshake with an EXPLICIT absent path, which is a
    # probe and must not poison the run's provenance state.
    assert enc.handshake_ran() is True
    assert enc.handshake_skipped == []
