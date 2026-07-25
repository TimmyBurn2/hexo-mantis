"""⊕ D-15 — the `SelfPlayRunnerConfig` assembly golden (WP-SP).

Written oracle-first against the dispatcher's old-side capture (#C3d, wp/WPSP/CAPTURE_LOG.md)
BEFORE any port code. RED at import until IMPL writes `mantis.selfplay.hparams`.

This file carries D-15 ONLY; the rest of Suite D is IMPL-written.

Method (forced by the surface, not chosen): the Rust `SelfPlayRunnerConfig` exposes getters
for only its 10 post-ctor `#[pyo3(get,set)]` attributes — none of the ctor kwargs is readable
back. So the golden IS the ctor-kwarg dict, and the oracle records it the same way the capture
did: a proxy that records kwargs + attribute sets while still constructing the REAL Rust
object underneath, so any Rust-side validation still fires.

THREE captured ctor kwargs do NOT cross (see ORACLE_NOTES §gaps):
  * `legal_move_radius_jitter` — DV-6 / WP6 KILL, declared in DESIGN;
  * `feature_len`, `policy_len`  — NOT declared anywhere; the committed new Rust ctor REJECTS
    both (they moved onto `InferenceBatcher` in WP7). Verified against `mantis._engine`.
All three are asserted ABSENT here, and their captured values live only in CAPTURE_LOG.
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
    """Proxy over the REAL Rust config: records ctor kwargs + post-ctor attribute sets."""

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
    """Factory → the RecordingRunnerConfig produced by assembling `config`."""
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
        # DESIGN §a.1 names these four parameters; the oracle calls them by KEYWORD and hands
        # the ResolvedPoolEncoding as `spec_dims` (ORACLE_NOTES §J3 — fixed here, before IMPL).
        build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name,
                            seed_prefixes=None)
        assert len(built) == before + 1, (
            "expected exactly ONE SelfPlayRunnerConfig construction per assembly"
        )
        return built[-1]

    return build


@pytest.mark.parametrize("case", ["full_config", "minimal_config"])
def test_runner_config_assembly_golden(runner_config_goldens, record_runner_config, case):
    """D-15 — PASS iff assembling each captured config dict hands the Rust
    `SelfPlayRunnerConfig` EXACTLY the captured ctor kwargs and post-ctor attributes (the
    WP6-KILLed / WP7-relocated names absent), and the real Rust ctor accepts them.

    FAIL = the config→runner wire drifted. That wire is the whole self-play behaviour surface
    and it is write-only from Python (no getters), so a silently wrong kwarg here changes what
    the runner does for an entire run with nothing to read back."""
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
    """D-15 (KILL arm) — PASS iff none of `legal_move_radius_jitter`, `feature_len`,
    `policy_len` appears in the ctor kwargs and `interior_selector` is never set post-ctor.
    FAIL = a WP6-KILLed knob was resurrected, or a WP7-relocated length was passed to a ctor
    that rejects it (which would be a hard TypeError at run start, not a silent drift)."""
    recorded = record_runner_config(runner_config_goldens["cases"][case]["config"])

    for name, why in NOT_CROSSING_CTOR_KWARGS.items():
        assert name not in recorded.recorded_kwargs, f"{case}: {name} must not cross — {why}"
    for name, why in NOT_CROSSING_POST_CTOR_ATTRS.items():
        assert name not in recorded.recorded_attrs, f"{case}: {name} must not cross — {why}"

    omitted = runner_config_goldens["_killed_fields_omitted"]
    assert "legal_move_radius_jitter" in omitted["ctor_kwargs"]
    assert "interior_selector" in omitted["post_ctor_attrs"]


def test_temperature_schedule_reaches_the_runner(runner_config_goldens, record_runner_config):
    """D-15 (behavior-named per R38/ADJ-02 — WPSC Phase 2 SC-A2 retires the historical
    spelling-mismatch framing) — PASS iff the FULL config's `playout_cap.
    temperature_threshold_compound_moves`/`temp_min` arrive at the runner as
    `temp_threshold_compound_moves=12`, `temp_min=0.35`, and the minimal config's absence of
    a schedule resolves to the cosine-OFF `(0, 0.5)` fallback."""
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
