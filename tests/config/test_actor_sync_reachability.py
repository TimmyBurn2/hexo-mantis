"""Cadence and threshold must be REACHABLE within the run (RED-TEAM F-2).

`ge=1` did not make "never sync" inexpressible. RED-TEAM put
`actor_sync_cadence_steps: 10**9` through the real loader, drove 200k steps, and the actor
synced exactly once and then froze for the rest of the run — run3's failure, in a config
that validated clean.

The two knobs failed OPEN TOGETHER, which is the part worth remembering: because
`actor_lag_threshold_steps` must exceed the cadence, an out-of-reach cadence also forces
the lag threshold out of reach — so the exit-45 invariant that exists precisely to catch a
frozen actor could never fire on one. A bound on either knob alone would not have closed it.

Reuses the frozen schema oracle's payload builders by IMPORT (reading a frozen file is not
editing it, R43).

NOT frozen: written after ORACLE-WRITE in response to a RED-TEAM finding.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema.core import RunConfig


def _load_frozen_schema_oracle():
    """Load the frozen oracle by PATH, not by dotted import.

    `tests` is deliberately not a package and no `sys.path` mutation is permitted (R5),
    so its payload builders are reached the same way the gate-11 producer test reaches
    its subject. Reading a frozen file is not editing it (R43).
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("test_actor_sync_schema.py")
    spec = importlib.util.spec_from_file_location("_frozen_actor_sync_schema", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_payload = _load_frozen_schema_oracle()._payload


def _total_steps() -> int:
    return int(_payload()["train"]["total_steps"])


def test_the_baseline_payload_is_still_valid():
    """Guard the premise: these bounds must not reject a healthy minted config."""
    RunConfig(**_payload())


def test_cadence_at_or_beyond_total_steps_is_a_named_error():
    total = _total_steps()
    with pytest.raises(ValidationError, match="must be < train.total_steps"):
        RunConfig(**_payload(
            train_over={"actor_sync_cadence_steps": total},
            monitor_over={"actor_lag_threshold_steps": total + 1},
        ))


def test_red_teams_exact_billion_step_cadence_is_rejected():
    """The literal reproduction from the RED-TEAM report."""
    with pytest.raises(ValidationError, match="must be < train.total_steps"):
        RunConfig(**_payload(
            train_over={"actor_sync_cadence_steps": 10**9},
            monitor_over={"actor_lag_threshold_steps": 10**9 + 1},
        ))


def test_threshold_at_or_beyond_total_steps_is_a_named_error():
    """An armed invariant the run can never reach is absent in effect, not armed."""
    total = _total_steps()
    with pytest.raises(ValidationError, match="never fire"):
        RunConfig(**_payload(monitor_over={"actor_lag_threshold_steps": total}))


def test_a_cadence_just_inside_the_run_is_accepted():
    """The bound is `>=`, not a blanket ban on large cadences — pin the boundary."""
    total = _total_steps()
    RunConfig(**_payload(
        train_over={"actor_sync_cadence_steps": total - 2},
        monitor_over={"actor_lag_threshold_steps": total - 1},
    ))


def test_the_pre_existing_threshold_exceeds_cadence_invariant_still_holds():
    """The new bound must not have displaced the old one."""
    with pytest.raises(ValidationError, match="must exceed train.actor_sync_cadence_steps"):
        RunConfig(**_payload(
            train_over={"actor_sync_cadence_steps": 100},
            monitor_over={"actor_lag_threshold_steps": 100},
        ))
