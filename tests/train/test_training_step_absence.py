"""AUDIT-1 F-01 + F-28 (INST-C01/C02/C03) — the `training_step` payload never fabricates.

THE DEFECT THIS PINS. `emit_training_step_event` built `policy_entropy` as
`float(loss_info.get("policy_entropy", 0.0))`. No trainer tail produces that key, every
minted config sets `alert_entropy_min: 1.0`, and `check_entropy_collapse` fires on
`ent < 1.0` — so every production run emitted a red `entropy_collapse` alert at every
`log_interval`, and a REAL collapse would have been the same event with the same text.

WHY THE PRODUCER IS REAL HERE AND WAS NOT BEFORE. Every coordinator test of this path
injects a hand-built `loss_info` carrying `"policy_entropy": 2.0` — a shape production never
emits — so LAW-07's producer test was satisfied against a fiction. These rows drive the two
REAL tails (`_graph_step` through the production dispatch, and `train_step_from_tensors`)
and feed their ACTUAL return dicts to the builder. If a trainer ever starts producing
`policy_entropy`, row one reds and says so rather than quietly re-arming the alert.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import emit_training_step_alerts
from mantis.train.events import emit_training_step_event

import _microbatch_harness as H  # the shared graph-step harness (rootdir-relative, house convention)

# Every minted config carries this value; the rule fires strictly below it.
MINTED_ENTROPY_FLOOR = 1.0

# The keys the two tails GUARANTEE. Everything else in the payload is a measurement that may
# be absent, and absence must travel as None.
GUARANTEED = ("loss", "policy_loss", "value_loss", "grad_norm", "lr")

# The payload fields that carry `None` when their producer did not supply them.
ABSENCE_CAPABLE = (
    "loss_aux", "loss_ownership", "loss_threat", "loss_chain", "avg_sigma",
    "policy_entropy", "policy_entropy_pretrain", "policy_entropy_selfplay",
    "policy_entropy_recent", "policy_target_entropy", "n_rows_policy_loss",
    "n_rows_total", "value_accuracy", "quiescence_fires_per_step",
)


def _real_graph_loss_info(tmp_path: Path) -> dict[str, float]:
    """One real graph training step through the PRODUCTION dispatch; its return dict."""
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import _graph_step as production_graph_step

    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    buffer = H.uniform_graph_buffer()
    wire, _targets = buffer.sample_graph_batch(4, augment=False, recent_frac=0.0)
    max_edges, max_nodes = H.non_binding_caps(wire)
    return production_graph_step(
        trainer, buffer, H.GSPEC,
        batch_size=4, augment=False, recency_weight=0.0, recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=max_edges, max_nodes=max_nodes),
        sample_threads_provider=lambda: 1,
    )


def _real_dense_loss_info(full_train_hparams: Any) -> dict[str, float]:
    """One real DENSE training step; its return dict. Mirrors the driver in
    `tests/train/test_losses.py` (checkpoint_interval=0, so no schema validation is reached)."""
    from mantis.train.trainer.core import Trainer

    spec = lookup("v6_live2_ls")
    arch = arch_from_spec_and_config(spec, {})
    config = {
        "schema_version": 1, "run_id": "run5", "seed": 7,
        "identity": {"encoding": "v6_live2_ls", "representation": "grid"},
        "eval": {"random_model_sims": 1, "sealbot_model_sims": 1},
        "selfplay": {"legal_move_radius_schedule": None},
        "train": {"amp_dtype": "fp16"},
    }
    hp = full_train_hparams(fp16=False, lr_schedule="none", checkpoint_interval=0)
    torch.manual_seed(11)
    tr = Trainer(build_net(arch), config, arch=arch, train_hparams=hp, sink=None)

    b, planes, hw = 3, int(spec.n_planes), int(spec.board_size)
    n_actions = hw * hw + 1
    states = np.zeros((b, planes, hw, hw), dtype=np.float32)
    policies = np.full((b, n_actions), 1.0 / n_actions, dtype=np.float32)
    outcomes = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    return tr.train_step_from_tensors(states, policies, outcomes)


def _alerts(payload: dict[str, Any]) -> list[str]:
    """The 4 WARN rules over one payload, at the MINTED thresholds, through a spy sink."""
    fired: list[dict[str, Any]] = []

    class _Sink:
        def emit(self, event: Any) -> None:
            fired.append(dict(event))

    emit_training_step_alerts(
        payload, MonitorConfig(alert_entropy_min=MINTED_ENTROPY_FLOOR), [], sink=_Sink()
    )
    return [e["rule"] for e in fired]


# ── the premise, re-derived rather than assumed ────────────────────────────────────────

def test_neither_real_trainer_tail_produces_policy_entropy(
    tmp_path: Path, full_train_hparams: Any
) -> None:
    """F-01's premise. If this reds, a producer appeared and the builder should carry it —
    the alert would then be measuring something, which it never has been."""
    graph = _real_graph_loss_info(tmp_path)
    dense = _real_dense_loss_info(full_train_hparams)
    assert "policy_entropy" not in graph, sorted(graph)
    assert "policy_entropy" not in dense, sorted(dense)
    # and the keys they DO guarantee are the ones the builder may read without a default
    for key in GUARANTEED:
        assert key in graph, f"graph tail dropped {key}: {sorted(graph)}"
        assert key in dense, f"dense tail dropped {key}: {sorted(dense)}"


# ── the audit's PIN, on both arms ──────────────────────────────────────────────────────

def test_the_real_graph_tail_yields_absent_entropy_and_fires_no_alert(
    tmp_path: Path
) -> None:
    """THE PIN. Before the repair: `policy_entropy` 0.0 and `entropy_collapse` fired."""
    payload = emit_training_step_event(0, _real_graph_loss_info(tmp_path), None, _NullSink())
    assert payload["policy_entropy"] is None
    assert "entropy_collapse" not in _alerts(payload)


def test_the_real_dense_tail_yields_absent_entropy_and_fires_no_alert(
    full_train_hparams: Any
) -> None:
    payload = emit_training_step_event(
        0, _real_dense_loss_info(full_train_hparams), None, _NullSink()
    )
    assert payload["policy_entropy"] is None
    assert "entropy_collapse" not in _alerts(payload)


# ── the control: the rule is silent because nothing measured it, NOT because it is dead ──

def test_a_MEASURED_entropy_below_the_floor_still_fires(tmp_path: Path) -> None:
    """The other half of the finding: masking. A real collapse must still be reported, or
    the repair would have replaced a false alarm with a dead rule."""
    loss_info = dict(_real_graph_loss_info(tmp_path))
    loss_info["policy_entropy"] = MINTED_ENTROPY_FLOOR - 0.5
    payload = emit_training_step_event(0, loss_info, None, _NullSink())
    assert payload["policy_entropy"] == pytest.approx(MINTED_ENTROPY_FLOOR - 0.5)
    assert "entropy_collapse" in _alerts(payload)


# ── the rest of the family (F-28 INST-C02/C03) ─────────────────────────────────────────

def test_every_unproduced_field_travels_as_None_never_a_fabricated_zero(
    tmp_path: Path
) -> None:
    """`docs/contracts/event_manifest.md`: an unproduced field carries `None`. A constant 0
    in the ONE channel reads as a measurement."""
    loss_info = _real_graph_loss_info(tmp_path)
    payload = emit_training_step_event(0, loss_info, None, _NullSink())
    for key in ABSENCE_CAPABLE:
        assert key in payload, f"{key} vanished from the payload shape"
        if key not in loss_info:
            assert payload[key] is None, f"{key} = {payload[key]!r}, expected None (no producer)"


def test_the_payload_is_valid_JSON_with_no_NaN(tmp_path: Path) -> None:
    """INST-C02. The three `policy_entropy_*` rows defaulted to `float('nan')`, which
    `json.dumps` writes as the bare token `NaN` — not valid JSON, and read back as a number
    by anything permissive. Absence is `null`."""
    payload = emit_training_step_event(0, _real_graph_loss_info(tmp_path), None, _NullSink())
    text = json.dumps(payload, allow_nan=False)  # raises ValueError on any NaN/Inf
    assert "NaN" not in text
    for key in ("policy_entropy_pretrain", "policy_entropy_selfplay", "policy_entropy_recent"):
        assert payload[key] is None


def test_a_produced_field_is_carried_through_unchanged(tmp_path: Path) -> None:
    """The absence convention must not eat real readings: a produced key survives."""
    loss_info = dict(_real_graph_loss_info(tmp_path))
    loss_info["n_rows_total"] = 4096
    loss_info["avg_sigma"] = 0.0  # a MEASURED zero, which must NOT become None
    payload = emit_training_step_event(0, loss_info, None, _NullSink())
    assert payload["n_rows_total"] == 4096
    assert payload["avg_sigma"] == 0.0 and payload["avg_sigma"] is not None
    assert math.isfinite(float(payload["loss_total"]))


class _NullSink:
    def emit(self, event: Any) -> None:
        return None
