# >300 justify (R8, and only just): O-D3 and O-D4 are two instruments on ONE config family and
# share its minted values, the injected-sink spy and the rigged filesystem. R5 bars cross-test
# imports, so splitting them writes the three minted numbers into a second file — a duplicated
# value authority, for the handful of lines it would save.
"""The `monitor.disk_guard` family — O-D3 (liveness) and O-D4 (structure).

`DiskGuard` was constructed at exactly one site, `build_subsystems`, which had ZERO callers,
and its `60.0/10.0/5.0` arrived as `dict.get`-shaped code-side defaults over a key that existed
in no schema and no config: four dead numbers and LAW-16's third leg unarmed. R121(b) mandates
the root construct the guard and R1 forbids literal or `dict.get` values.

O-D3 is LIVENESS — set the knob through the ONE loader, observe the consumer. O-D4 is
STRUCTURE — a live key can grow a pydantic default tomorrow, and a defaulted key is a second
authority no liveness drive sees, because the drive supplies the value either way.

Fakes: the filesystem and only the filesystem, so the thresholds can be crossed on demand. The
guard, the schema and the resolver are real, and every config is a minted file through the ONE
loader.
"""
from __future__ import annotations

import collections
import os
import shutil
import signal
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.resolve.disk_guard import DiskGuardSpec, resolve_disk_guard  # RED anchor
from mantis.config.resolve.monitor import resolve_monitor_config
from mantis.config.schema import (  # RED anchor: DiskGuardConfig
    DiskGuardConfig,
    RunConfig,
    operational_default_fields,
)
from mantis.train.lifecycle.disk_guard import DiskGuard

_GB = 1_000_000_000  # decimal GB — the divisor `disk_guard.py` calibrates against

#: R122's minted family, stated here so a re-mint that quietly moves them is loud. The values
#: are revisable at mint prereg — the literals were dead, so nothing has ever measured them —
#: and that is a mint decision, not an IMPL edit.
_MINTED = {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0}
_FIELDS = sorted(_MINTED)


def _payload(**over: object) -> dict:
    out = dict(_MINTED)
    out.update(over)
    return out


class _SpySink:
    """The injected `EventSink` seam, recording what the guard published."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [event for event in self.events if event.get("event") == name]


def _fake_disk_usage(free_gb: float):
    usage = collections.namedtuple("usage", "total used free")

    def _fn(_path):
        return usage(total=100 * _GB, used=int((100 - free_gb) * _GB),
                     free=int(free_gb * _GB))

    return _fn


def test_disk_guard_valid_payload_constructs_clean() -> None:
    """O-D4, premise. The minted family validates as itself."""
    cfg = DiskGuardConfig.model_validate(_payload())
    assert (cfg.interval_sec, cfg.warn_gb, cfg.fail_gb) == (60.0, 10.0, 5.0)


@pytest.mark.parametrize("field", _FIELDS)
def test_an_omitted_disk_guard_key_lands_on_its_declared_default(field: str) -> None:
    """O-D4 arm 1, INVERTED: all three thresholds are declared operational constants now, so a
    missing key is legal and the claim moves to the VALUE. What R122 forbade was `.get(name,
    60.0)` at a CALL SITE — a second authority no config could override."""
    payload = _payload()
    del payload[field]
    cfg = DiskGuardConfig.model_validate(payload)
    assert getattr(cfg, field) == DiskGuardConfig.model_fields[field].get_default(
        call_default_factory=True), (
        f"monitor.disk_guard.{field} did not land on its schema default"
    )


def test_disk_guard_extra_key_rejected() -> None:
    """O-D4, arm 2 — `extra="forbid"`: `keep_all` gets NO key (the root passes `False` with a
    disclosure comment), so writing one must be refused rather than silently ignored."""
    with pytest.raises(ValidationError, match="keep_all"):
        DiskGuardConfig.model_validate(_payload(keep_all=True))


def test_disk_guard_has_no_pydantic_level_default() -> None:
    """O-D4, arm 3 — the no-code-side-defaults census every OTHER schema block gets.

    MUTATION THAT REDS IT: `interval_sec: float = 60.0`. The liveness drives supply a value on
    every path, so a default is invisible to them. Asserted here as well as in the shared
    census, on purpose: one census that can be forgotten is one census.
    """
    # INVERTED rather than deleted: the thresholds are operational constants, so they carry
    # schema defaults and leave the YAML, and the guard becomes the equality — every field is
    # DECLARED in `OPERATIONAL_DEFAULT_KEYS`, so a fourth leaf without a registry row reds.
    assert set(DiskGuardConfig.model_fields) == operational_default_fields(
        "monitor.disk_guard"), (
        "a DiskGuardConfig field is not declared in OPERATIONAL_DEFAULT_KEYS (or a declared "
        "one is gone) — the registry and the block must say the same thing"
    )
    for name, field in DiskGuardConfig.model_fields.items():
        assert not field.is_required(), (
            f"DiskGuardConfig.{name} is declared operational but is still required"
        )


def test_a_fail_threshold_at_or_above_the_warn_threshold_is_refused() -> None:
    """O-D3's validator arm: `fail_gb >= warn_gb` SIGTERMs the run before it ever warns, which
    reads as normal in a config diff. Inert at the minted 60/10/5 deliberately, on the
    `_policy_target_completed_q_consistency` precedent.

    MUTATION THAT REDS IT: drop the model validator — the equal case reads legal to every
    field-level `gt=0` bound."""
    for fail_gb in (10.0, 12.0):
        with pytest.raises(ValidationError):
            DiskGuardConfig.model_validate(_payload(fail_gb=fail_gb))


def _minted(smoke_run_config, **disk_guard) -> RunConfig:
    """A REAL minted config with the disk-guard block overridden, through the ONE loader.
    `smoke_run_config` is the root conftest's factory, and overrides are re-validated so a value
    this file writes is one the loader would accept."""
    return smoke_run_config("smoke_preflight_armed.yaml", monitor={"disk_guard": dict(disk_guard)})


@pytest.mark.parametrize(("field", "value"), [
    ("interval_sec", 7.5), ("warn_gb", 42.0), ("fail_gb", 3.5),
])
def test_each_disk_guard_key_arrives_whole_at_its_one_resolver(
    field: str, value: float, smoke_run_config
) -> None:
    """O-D3, arm 1 — the per-key mutation, through the resolver R122 mandates.

    MUTATION THAT REDS IT: a resolver that reads a constant or the wrong leaf — a transposed
    `warn_gb`/`fail_gb` kills the run at the warning threshold while every field bound passes.
    Three keys, three independent values, so a transposition cannot alias into a green. A
    resolver at all because the pop in `resolve_monitor_config` is legitimate ONLY while a
    second reader exists.
    """
    spec = resolve_disk_guard(_minted(smoke_run_config, **_payload(**{field: value})).monitor)
    assert isinstance(spec, DiskGuardSpec)
    assert getattr(spec, field) == value, (
        f"monitor.disk_guard.{field} must arrive whole at its resolver; got {spec}"
    )
    for other in _FIELDS:
        if other != field:
            assert getattr(spec, other) == _MINTED[other], (
                f"…and setting {field} must not move {other} ({spec})"
            )


def test_the_monitor_resolver_drops_disk_guard_by_name_never_by_a_filter(smoke_run_config) -> None:
    """O-D3, arm 2 — the MEASURED BLOCKER and the constraint on how it is fixed.

    `resolve_monitor_config` dumps, pops `drain`, and rebuilds a frozen `MonitorConfig` with no
    `disk_guard` field, so the block breaks on an unexpected kwarg unless it is popped too — and
    the pop must be ENUMERATED. MUTATION THAT REDS IT: generalise it to a comprehension over
    `__dataclass_fields__`, which fixes today's break and silently swallows every future
    unmatched key.
    """
    source = Path(resolve_monitor_config.__globals__["__file__"]).read_text(encoding="utf-8")
    # The THIRD drop, `gate_interval`, is asserted in the SAME enumerated list: a third member
    # is exactly the pressure that tempts a reviewer to collapse the pops into a filter.
    for key in ('pop("gate_interval")', 'pop("drain")', 'pop("disk_guard")'):
        assert key in source, (
            f"the drop must name the block: `data.{key}` — an enumerated pop is auditable, a "
            "filter is not"
        )
    for banned in ("__dataclass_fields__", "for key in data", "if key in "):
        assert banned not in source, (
            f"{banned!r} in the monitor resolver is the generalised drop: it makes every "
            "future unmatched key vanish silently (DR-11, F-10)"
        )
    monitor_cfg = resolve_monitor_config(_minted(smoke_run_config, **_payload()).monitor)
    assert not hasattr(monitor_cfg, "disk_guard"), (
        "the runtime MonitorConfig carries neither block — both have their own resolver"
    )


def test_the_warn_and_fail_thresholds_each_govern_the_guards_real_behaviour(
    monkeypatch, smoke_run_config
) -> None:
    """O-D3, arm 3 — the config values reach the REAL guard and decide what it does. Three rigged
    readings against one resolved spec (warn 10, fail 5): 20 GB quiet, 8 GB warns, 3 GB critical
    and SIGTERMs, with `os.kill` captured rather than delivered.

    MUTATION THAT REDS IT: build the guard from literals instead of the resolved spec — a config
    that moves warn_gb to 42 then changes nothing, which the fourth assertion drives.
    """
    kills: list = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    spec = resolve_disk_guard(_minted(smoke_run_config, **_payload()).monitor)

    def _guard(sink) -> DiskGuard:
        return DiskGuard(watch_path=Path("."), interval_sec=spec.interval_sec,
                         warn_gb=spec.warn_gb, fail_gb=spec.fail_gb, keep_all=False, sink=sink)

    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(20))
    quiet = _SpySink()
    _guard(quiet).check_once()
    assert quiet.named("disk_free") and not quiet.named("disk_alert")

    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(8))
    warned = _SpySink()
    _guard(warned).check_once()
    assert warned.named("disk_alert")[-1]["level"] == "warn"
    assert kills == [], "a warning must not kill the run"

    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(3))
    critical = _SpySink()
    _guard(critical).check_once()
    assert critical.named("disk_alert")[-1]["level"] == "critical"
    assert kills == [(os.getpid(), signal.SIGTERM)], (
        "below fail_gb the guard SIGTERMs itself — which, with the root's handlers now "
        "installed, is save-then-exit rather than a lost run (F-1 and F-2 were coupled)"
    )

    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(20))
    raised = resolve_disk_guard(_minted(smoke_run_config, **_payload(warn_gb=42.0)).monitor)
    moved = _SpySink()
    DiskGuard(watch_path=Path("."), interval_sec=raised.interval_sec, warn_gb=raised.warn_gb,
              fail_gb=raised.fail_gb, keep_all=False, sink=moved).check_once()
    assert moved.named("disk_alert")[-1]["level"] == "warn", (
        "moving warn_gb in the CONFIG must move the guard's behaviour — the whole point of "
        "the key existing (LAW-08)"
    )


def test_the_interval_key_governs_the_guard_thread_not_just_the_constructor(
    monkeypatch, smoke_run_config
) -> None:
    """O-D3, arm 4 — `interval_sec`'s observable, which is neither threshold's. It reaches only
    `self._stop_event.wait(timeout=self._interval)` inside the guard's thread, so a ctor-kwarg
    assertion cannot tell a live interval from a dead one: two real guards over the same rigged
    filesystem and window, a short interval publishing and a long one not.

    MUTATION THAT REDS IT: hardcode the loop's sleep — every ctor assertion stays green."""
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(50))
    fast_sink, slow_sink = _SpySink(), _SpySink()
    fast = resolve_disk_guard(_minted(smoke_run_config, **_payload(interval_sec=0.02)).monitor)
    slow = resolve_disk_guard(_minted(smoke_run_config, **_payload(interval_sec=300.0)).monitor)
    guards = [
        DiskGuard(watch_path=Path("."), interval_sec=spec.interval_sec, warn_gb=spec.warn_gb,
                  fail_gb=spec.fail_gb, keep_all=False, sink=sink)
        for spec, sink in ((fast, fast_sink), (slow, slow_sink))
    ]
    for guard in guards:
        guard.start()
    try:
        time.sleep(0.5)
    finally:
        for guard in guards:
            guard.stop()
    assert fast_sink.named("disk_free"), (
        "a 0.02 s interval must publish inside a 0.5 s window (~25 expected ticks against "
        "an assertion of >= 1)"
    )
    assert not slow_sink.named("disk_free"), (
        "…and a 300 s interval must not: the CONFIG decides the cadence, not a literal"
    )
