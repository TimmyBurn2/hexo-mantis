"""B-1 (R355(e)): the inference server serves the learner's own module (ActorSync is a self-copy)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.armed_aborts import MANIFEST


def test_ema_enabled_true_is_refused_by_name(smoke_run_config) -> None:
    with pytest.raises(ValidationError, match="server-owned copy"):
        smoke_run_config("dev_example.yaml", train={"ema": {"enabled": True}})


def test_ema_enabled_false_still_validates(smoke_run_config) -> None:
    assert smoke_run_config("dev_example.yaml").train.ema.enabled is False


def test_the_actor_lag_row_is_retired_from_the_manifest() -> None:
    """A row that cannot fire is a lie: a self-copy's lag never exceeds the sync cadence."""
    assert "actor_lag" not in [row.name for row in MANIFEST]
