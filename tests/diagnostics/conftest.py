"""The worker-sweep suites' shared committed-plan fixture."""
from __future__ import annotations

from pathlib import Path

import pytest

import mantis.diagnostics.worker_sweep as ws

_PLAN = Path(__file__).resolve().parents[2] / "tools" / "worker_sweep_plan.toml"


@pytest.fixture()
def plan() -> ws.SweepPlan:
    return ws.load_plan(_PLAN)
