# >300 justify (R8, and only just): O-D3 and O-D4 are two instruments on ONE config family and
# share its minted values (`_MINTED` = 60/10/5), the injected-sink spy and the rigged
# filesystem. R5 bars cross-test imports, so splitting them writes R122's three minted
# numbers into a second file — a duplicated value authority, which is the shape R1 exists to
# kill and a poor trade for the handful of lines it would save.
"""⊕ WPMAIN ORACLE — the `monitor.disk_guard` family (DESIGN §5.5/§7/§9, O-D3 + O-D4).

RED-at-import until IMPL lands `mantis.config.schema.DiskGuardConfig` +
`mantis.config.resolve.disk_guard.resolve_disk_guard` (R122's grant: ONE config block, ONE
resolver, THREE typed leaves, minted 60/10/5).

What this file exists to stop, measured at `b482243`:

- `DiskGuard` is constructed at exactly one site in the tree — `build_subsystems`
  (`subsystems.py:150`) — which has ZERO callers. Its `60.0/10.0/5.0` arrive as
  `config.get("disk_guard", {}).get("interval_sec", 60.0)`-shaped code-side defaults over a
  key that exists in no schema and no config. Four dead numbers, an unconstructed guard, and
  LAW-16's third leg unarmed.
- R121(b) MANDATES the root construct the guard; R1 FORBIDS the construction values being
  literals or `dict.get` defaults. R122 rules the only disposition that satisfies both.

The two oracles are deliberately different instruments and neither substitutes for the other:

- **O-D3 is LIVENESS** — each key, set through the ONE loader, changes something a run can
  observe. That is R93's house standard verbatim: set the knob, observe the consumer. It is
  what the DR-11 finding (four minted keys read by nothing) proves a citation cannot do.
- **O-D4 is STRUCTURE** — the SC-A pair every schema block in this repo carries. A key that
  is live today can grow a pydantic default tomorrow, and a defaulted key is a second
  authority that no liveness drive sees (the drive supplies the value either way).

Fakes: the filesystem, and only the filesystem. `shutil.disk_usage` is replaced so the
thresholds can be crossed on demand — the house precedent is
`tests/train/test_lifecycle_contract.py`'s `_fake_disk_usage`, for the same reason: the
alternative is filling a real volume. The GUARD is the real `DiskGuard`, the SCHEMA is the
real schema, the RESOLVER is the real resolver, and every config comes from a minted file
through the ONE loader.
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
from mantis.config.schema import DiskGuardConfig, RunConfig  # RED anchor: DiskGuardConfig
from mantis.train.lifecycle.disk_guard import DiskGuard

_GB = 1_000_000_000  # decimal GB — the divisor `disk_guard.py` calibrates against

#: R122's minted family. Stated here so a re-mint that quietly moves them is loud; the values
#: themselves are revisable at mint prereg (R85 pattern — the literals were dead, so nothing
#: has ever measured them), and THAT is a mint decision, not an IMPL edit.
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


# ══ O-D4 — the SC-A structural pair ═══════════════════════════════════════════════════
def test_disk_guard_valid_payload_constructs_clean() -> None:
    """O-D4, premise. The minted family validates as itself."""
    cfg = DiskGuardConfig.model_validate(_payload())
    assert (cfg.interval_sec, cfg.warn_gb, cfg.fail_gb) == (60.0, 10.0, 5.0)


@pytest.mark.parametrize("field", _FIELDS)
def test_disk_guard_missing_field_rejected(field: str) -> None:
    """O-D4, arm 1 — R1 completeness: a missing key is an ERROR naming the key, never a
    code-side stand-in. MUTATION THAT REDS IT: give the field a default, which is precisely
    the shape (`.get(name, 60.0)`) this whole family replaces."""
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        DiskGuardConfig.model_validate(payload)


def test_disk_guard_extra_key_rejected() -> None:
    """O-D4, arm 2 — `extra="forbid"`: `keep_all` gets NO key (an inert carried knob; the
    root passes `False` with a disclosure comment), so writing one must be refused rather
    than silently ignored."""
    with pytest.raises(ValidationError, match="keep_all"):
        DiskGuardConfig.model_validate(_payload(keep_all=True))


def test_disk_guard_has_no_pydantic_level_default() -> None:
    """O-D4, arm 3 — the census `test_o16_all_fields_required_no_code_side_defaults`
    (`tests/config/test_schema.py:247-251`) performs for every OTHER schema block, and which
    `DiskGuardConfig` would otherwise be the one block in the tree to lack (§3.1 MISS-9).

    MUTATION THAT REDS IT: `interval_sec: float = 60.0`. Nothing else sees that — the
    liveness drives below supply a value on every path, so a default is invisible to them,
    and this is the direct structural guard on R122's "not literals, not `dict.get`
    defaults". IMPL additionally extends the shared `test_o16` tuple (MISS-9, an R50 row);
    this assertion stands whether or not that edit lands, on purpose — one census that can
    be forgotten is one census."""
    for name, field in DiskGuardConfig.model_fields.items():
        assert field.is_required(), f"DiskGuardConfig.{name} has a code-side default"


def test_a_fail_threshold_at_or_above_the_warn_threshold_is_refused() -> None:
    """O-D3's validator arm. `fail_gb >= warn_gb` means the run SIGTERMs itself before it
    ever warns — a guard that skips its own warning stage, which is a misconfiguration no
    operator intends and which reads as normal in a config diff.

    Inert at the minted 60/10/5 (5 < 10 holds), deliberately: the house precedent for an
    inert-at-mint validator is `RunConfig._policy_target_completed_q_consistency`
    (`schema/core.py:253+`), cited so this is not mistaken for R116 dead weight.

    MUTATION THAT REDS IT: drop the model validator — the equal case in particular reads
    perfectly legal to every field-level `gt=0` bound."""
    for fail_gb in (10.0, 12.0):
        with pytest.raises(ValidationError):
            DiskGuardConfig.model_validate(_payload(fail_gb=fail_gb))


# ══ O-D3 — liveness: set the knob, observe the consumer (R93) ═════════════════════════
def _minted(smoke_run_config, **disk_guard) -> RunConfig:
    """A REAL minted config with the disk-guard block overridden, through the ONE loader.

    `smoke_run_config` is the root conftest's factory (R5: no cross-test import exists or is
    wanted — the fixture IS the shared surface). Overrides are re-validated, so a value this
    file writes is a value the loader would accept."""
    return smoke_run_config("smoke_gnn.yaml", monitor={"disk_guard": dict(disk_guard)})


@pytest.mark.parametrize(("field", "value"), [
    ("interval_sec", 7.5), ("warn_gb", 42.0), ("fail_gb", 3.5),
])
def test_each_disk_guard_key_arrives_whole_at_its_one_resolver(
    field: str, value: float, smoke_run_config
) -> None:
    """O-D3, arm 1 — the per-key mutation, through the resolver R122 mandates.

    MUTATION THAT REDS IT: a resolver that reads a constant, or reads the wrong leaf (a
    transposed `warn_gb`/`fail_gb` is a guard that kills the run at the warning threshold —
    and every field-level bound still passes). Three keys, three independent values, so a
    transposition cannot alias into a green.

    Why a resolver at all rather than a direct attribute read at the root: `disk_guard` would
    otherwise be the ONE `monitor.*` sub-block without one, and — measured, A.1.2 — the pop
    in `resolve_monitor_config` is legitimate ONLY because a second reader exists. Without
    this function that pop is the DR-11 defect verbatim."""
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
    """O-D3, arm 2 — the MEASURED BLOCKER (A.1.2) and the F-10 constraint on how it is fixed.

    `resolve_monitor_config` is `data = cfg.model_dump(); data.pop("drain");
    MonitorConfig(**data)`, and `MonitorConfig` is a frozen dataclass with no `disk_guard`
    field — so adding the block breaks it on an unexpected kwarg unless `disk_guard` is
    popped too.

    The pop must be ENUMERATED. MUTATION THAT REDS IT: generalise it to a comprehension over
    `MonitorConfig.__dataclass_fields__` — which fixes today's break and silently swallows
    EVERY future unmatched key, i.e. re-creates the DR-11 defect the file's own docstring
    (`monitor.py:13-14`) says this line becomes the moment its second reader disappears.
    That is a weaken-class change, and it is exactly the kind P-11 already forbids on the
    sibling census."""
    source = Path(resolve_monitor_config.__globals__["__file__"]).read_text(encoding="utf-8")
    # R242 (ADJ-D12) adds the THIRD drop, `gate_interval` — a schema-only scalar whose reader
    # is `mantis.run.compose_run` -> `StepCoordinatorConfig.gate_interval`. It is asserted in
    # the SAME enumerated list, deliberately: a third member is exactly the pressure that
    # tempts a reviewer to collapse the three pops into a filter, which is the move this
    # test exists to red.
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
    """O-D3, arm 3 — the config values reach the REAL guard and decide what it does.

    Three rigged free-space readings against one resolved spec (warn 10, fail 5): 20 GB is
    quiet, 8 GB warns, 3 GB is critical and SIGTERMs. `os.kill` is captured rather than
    delivered — a real critical alert during the test suite would send SIGTERM to pytest,
    which is the guard working correctly and is not something to demonstrate on the box the
    suite runs on.

    MUTATION THAT REDS IT: build the guard from literals instead of the resolved spec (the
    R1 breach) — a config that moves warn_gb to 42 then changes nothing, which is the
    dead-value state at HEAD. The fourth assertion drives exactly that: a 42 GB warn
    threshold must make a 20 GB reading WARN."""
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
    """O-D3, arm 4 — `interval_sec`'s observable, which is neither of the thresholds'.

    `interval_sec` reaches only `self._stop_event.wait(timeout=self._interval)` inside the
    guard's own thread, so a constructor-kwarg assertion alone cannot tell a live interval
    from a dead one. Two real guards over the same rigged filesystem and the same wall-clock
    window: a short interval publishes, a long one does not.

    MUTATION THAT REDS IT: hardcode the loop's sleep (`wait(timeout=60.0)`) — every ctor
    assertion in the repo stays green while the operator's cadence does nothing."""
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
