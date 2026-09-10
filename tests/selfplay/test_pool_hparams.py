"""The `SelfPlayRunnerConfig` assembly golden.

The Rust config exposes getters for its post-ctor attributes only — no ctor kwarg is readable
back — so the golden IS the ctor-kwarg dict, recorded by a proxy that captures kwargs and
attribute sets while still constructing the real Rust object, so Rust-side validation fires.

Three captured ctor kwargs deliberately do not cross and are asserted absent: the retired
`legal_move_radius_jitter`, and `feature_len` / `policy_len`, which moved onto
`InferenceBatcher` and are rejected by the current ctor.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis import _engine
from mantis.selfplay import hparams as hparams_mod
from mantis.selfplay.hparams import (
    SelfPlayHParams,
    build_runner_config,
    resolve_pool_encoding,
)

# name → why it must not reach the new Rust ctor.
NOT_CROSSING_CTOR_KWARGS = {
    "legal_move_radius_jitter": "DV-6 / WP6 KILL — the field no longer exists",
    "feature_len": "moved to InferenceBatcher (WP7); the new Rust ctor rejects it",
    "policy_len": "moved to InferenceBatcher (WP7); the new Rust ctor rejects it",
}
NOT_CROSSING_POST_CTOR_ATTRS = {
    "interior_selector": "DV-6 / WP6 KILL — and with it the old hard-read KeyError path",
}


class RecordingRunnerConfig:
    """Proxy over the real Rust config, recording ctor kwargs and post-ctor attribute sets."""

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "recorded_kwargs", dict(kwargs))
        object.__setattr__(self, "recorded_attrs", {})
        object.__setattr__(self, "real", _engine.SelfPlayRunnerConfig(**kwargs))

    def __setattr__(self, name: str, value: Any) -> None:
        self.recorded_attrs[name] = value
        setattr(self.real, name, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "real"), name)


@pytest.fixture
def record_runner_config(monkeypatch):
    """Return a factory building the RecordingRunnerConfig for an assembled `config`."""
    built: list[RecordingRunnerConfig] = []

    class _Factory(RecordingRunnerConfig):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            built.append(self)

    monkeypatch.setattr(hparams_mod, "SelfPlayRunnerConfig", _Factory)

    def build(config: dict[str, Any]) -> RecordingRunnerConfig:
        before = len(built)
        hp = SelfPlayHParams.from_config(config)
        enc = resolve_pool_encoding(config, arch=None)
        # Called by KEYWORD, with the ResolvedPoolEncoding handed over as `spec_dims`.
        build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name)
        assert len(built) == before + 1, (
            "expected exactly ONE SelfPlayRunnerConfig construction per assembly"
        )
        return built[-1]

    return build


@pytest.mark.parametrize("case", ["full_config", "minimal_config"])
def test_runner_config_assembly_golden(runner_config_goldens, record_runner_config, case):
    """Assembling a captured config hands the Rust ctor exactly the captured kwargs and attrs.

    That wire is the whole self-play behaviour surface and is write-only from Python, so a
    wrong kwarg changes what the runner does for a whole run with nothing to read back.
    """
    golden = runner_config_goldens["cases"][case]
    recorded = record_runner_config(golden["config"])

    expected_kwargs = {k: v for k, v in golden["ctor_kwargs"].items()
                       if k not in NOT_CROSSING_CTOR_KWARGS}
    assert set(recorded.recorded_kwargs) == set(expected_kwargs), (
        f"{case}: ctor kwarg set drift — missing "
        f"{set(expected_kwargs) - set(recorded.recorded_kwargs)}, extra "
        f"{set(recorded.recorded_kwargs) - set(expected_kwargs)}"
    )
    for key, want in expected_kwargs.items():
        assert recorded.recorded_kwargs[key] == want, (
            f"{case}: ctor kwarg {key} = {recorded.recorded_kwargs[key]!r} != {want!r}"
        )

    expected_attrs = {k: v for k, v in golden["post_ctor_attrs"].items()
                      if k not in NOT_CROSSING_POST_CTOR_ATTRS}
    assert set(recorded.recorded_attrs) == set(expected_attrs), (
        f"{case}: post-ctor attribute set drift — missing "
        f"{set(expected_attrs) - set(recorded.recorded_attrs)}, extra "
        f"{set(recorded.recorded_attrs) - set(expected_attrs)}"
    )
    for key, want in expected_attrs.items():
        assert recorded.recorded_attrs[key] == want, (
            f"{case}: post-ctor attr {key} = {recorded.recorded_attrs[key]!r} != {want!r}"
        )


@pytest.mark.parametrize("case", ["full_config", "minimal_config"])
def test_killed_and_relocated_fields_never_reach_the_runner(
        runner_config_goldens, record_runner_config, case):
    """The retired and relocated names never reach the runner, in kwargs or post-ctor attrs."""
    recorded = record_runner_config(runner_config_goldens["cases"][case]["config"])

    for name, why in NOT_CROSSING_CTOR_KWARGS.items():
        assert name not in recorded.recorded_kwargs, f"{case}: {name} must not cross — {why}"
    for name, why in NOT_CROSSING_POST_CTOR_ATTRS.items():
        assert name not in recorded.recorded_attrs, f"{case}: {name} must not cross — {why}"

    omitted = runner_config_goldens["_killed_fields_omitted"]
    assert "legal_move_radius_jitter" in omitted["ctor_kwargs"]
    assert "interior_selector" in omitted["post_ctor_attrs"]


def test_playout_cap_temperature_threshold_reaches_the_runner(
        runner_config_goldens, record_runner_config):
    """The playout-cap temperature schedule reaches the runner under its ctor-kwarg spelling,
    and a config with no schedule resolves to the off `(0, 0.5)` pair."""
    recorded = record_runner_config(runner_config_goldens["cases"]["full_config"]["config"])
    assert recorded.recorded_kwargs["temp_threshold_compound_moves"] == 12, (
        "temperature schedule was NOT read — the config key is "
        "`temperature_threshold_compound_moves`, not the ctor-kwarg spelling"
    )
    assert recorded.recorded_kwargs["temp_min"] == 0.35

    minimal = record_runner_config(
        runner_config_goldens["cases"]["minimal_config"]["config"])
    assert minimal.recorded_kwargs["temp_threshold_compound_moves"] == 0
    assert minimal.recorded_kwargs["temp_min"] == 0.5
