"""⊕ WPUF Phase U ORACLE — O-U5: the three knobs and nothing else (DESIGN_U §5/§9).

`mantis.config.resolve.actor_sync.resolve_actor_sync_cadence` (K1's ONE read path) and
the three schema fields are the subject.

R1/LAW-08: missing key = named error at load, never a fallback; `ge=1` on the cadence
means NO representable "off" value exists (R49 at the type level); the cross-field
validator (`RunConfig`-level, since it spans sections) rejects a threshold at or below
the cadence with a NAMED message. Payload builders mirror
tests/config/test_train_policy_value_target_consistency.py's full-RunConfig shape.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence
from mantis.config.resolve import resolve_monitor_config
from mantis.config.schema import RunConfig, SCHEMA_VERSION, TrainConfig, MonitorSchemaConfig
from _monitor_config import monitor_config


def _schema_blocks():
    """Spec-load the sibling helper by path (by-path loaders lack this dir on sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "_schema_blocks", Path(__file__).resolve().parent / "_schema_blocks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sb = _schema_blocks()
eval_block, inference_block = _sb.eval_block, _sb.inference_block
monitor_block, selfplay_block, train_block = _sb.monitor_block, _sb.selfplay_block, _sb.train_block

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = ("dev_example.yaml", "run6.yaml", "smoke_preflight_armed.yaml")

_NEW_KEYS = (
    ("train", "actor_sync_cadence_steps"),
    ("monitor", "actor_lag_threshold_steps"),
    ("monitor", "actor_lag_abort_enabled"),
)


def _payload(*, train_over: dict | None = None, monitor_over: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "run_id": "unit_test", "seed": 1,
        "eval_enabled": True,
        # RECAL-PREP (R308(g)(i)): a REQUIRED top-level leaf. `null` is R119's
        # placeholder — refused at boot on a cuda process, valued only by the
        # re-calibration sitting under R282(b).
        "allocator_posture": None,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "model": {"gnn": {"hidden": 128, "num_layers": 4}, "aux_soft_policy": None},
        "eval": eval_block(), "train": train_block(**(train_over or {})),
        "deploy": {"search": {"kind": "puct"}},
        "selfplay": selfplay_block(), "inference": inference_block(),
        "monitor": monitor_block(**(monitor_over or {})),
    }


# construction + named absence errors
def test_valid_payload_with_the_three_knobs_constructs_clean() -> None:
    cfg = RunConfig.model_validate(_payload())
    assert cfg.train.actor_sync_cadence_steps == 1
    assert cfg.monitor.actor_lag_threshold_steps == 100
    assert cfg.monitor.actor_lag_abort_enabled is False


@pytest.mark.parametrize(("section", "key"), _NEW_KEYS)
def test_missing_knob_is_a_named_error_at_load(section: str, key: str) -> None:
    """R1: no code-side default — a missing key is a ValidationError NAMING the key."""
    payload = _payload()
    del payload[section][key]
    with pytest.raises(ValidationError, match=key):
        RunConfig.model_validate(payload)


# bounds: no representable "off" (R49)
@pytest.mark.parametrize("bad_cadence", [0, -1])
def test_cadence_has_no_representable_off_value(bad_cadence: int) -> None:
    """`ge=1`: the schema CANNOT express "don't sync" — R49 enforced at the type level."""
    with pytest.raises(ValidationError, match="actor_sync_cadence_steps"):
        RunConfig.model_validate(
            _payload(train_over={"actor_sync_cadence_steps": bad_cadence}))


@pytest.mark.parametrize("bad_threshold", [0, -1])
def test_lag_threshold_rejects_nonpositive(bad_threshold: int) -> None:
    """`ge=1`; disablement is the arming flag's job — one authority, no zero-sentinel."""
    with pytest.raises(ValidationError, match="actor_lag_threshold_steps"):
        RunConfig.model_validate(
            _payload(monitor_over={"actor_lag_threshold_steps": bad_threshold}))


# the cross-field validator (RunConfig-level; DESIGN §5's named message)
@pytest.mark.parametrize("threshold", [8, 4])
def test_threshold_at_or_below_cadence_rejected_with_named_message(threshold: int) -> None:
    with pytest.raises(ValidationError,
                       match="must exceed train.actor_sync_cadence_steps"):
        RunConfig.model_validate(_payload(
            train_over={"actor_sync_cadence_steps": 8},
            monitor_over={"actor_lag_threshold_steps": threshold}))


def test_threshold_just_above_cadence_accepted() -> None:
    cfg = RunConfig.model_validate(_payload(
        train_over={"actor_sync_cadence_steps": 8},
        monitor_over={"actor_lag_threshold_steps": 9}))
    assert cfg.monitor.actor_lag_threshold_steps == 9


# resolvers: the ONE read path per knob
def test_resolver_returns_the_configured_cadence() -> None:
    cfg = RunConfig.model_validate(
        _payload(train_over={"actor_sync_cadence_steps": 7}))
    assert resolve_actor_sync_cadence(cfg.train) == 7


def test_resolve_monitor_config_copies_the_lag_fields() -> None:
    section = MonitorSchemaConfig.model_validate(
        monitor_block(actor_lag_threshold_steps=77, actor_lag_abort_enabled=True))
    resolved = resolve_monitor_config(section)
    assert resolved.actor_lag_threshold_steps == 77
    assert resolved.actor_lag_abort_enabled is True


def test_runtime_monitor_config_carries_the_smoke_posture() -> None:
    """The established monitor pattern: schema REQUIRED, runtime dataclass carries the
    smoke value — threshold 100 (inert at cadence 1), abort False (config arms it)."""
    runtime = monitor_config()
    assert runtime.actor_lag_threshold_steps == 100
    assert runtime.actor_lag_abort_enabled is False


# the minted configs carry all three keys (a hand-revert fails LOCALLY)
@pytest.mark.parametrize("name", _CONFIGS)
def test_minted_config_carries_all_three_keys(name: str) -> None:
    data = yaml.safe_load((_REPO / "configs" / name).read_text(encoding="utf-8"))
    for section, key in _NEW_KEYS:
        assert key in data.get(section, {}), (
            f"configs/{name}: missing {section}.{key} — configs are minted complete (R1); "
            "a hand-reverted file must fail here, not only in CI gate 7"
        )
