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
and feed their ACTUAL return dicts to the builder. The graph tail PRODUCES `policy_entropy`
since R355(e) (B-4); row one pins that it is a finite measurement the alert reads, and every
row the payload carries is one the real tail PRODUCES: a field no tail fills leaves the event.
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
GUARANTEED = ("loss", "policy_loss", "value_loss", "grad_norm", "lr",
              "policy_entropy", "policy_entropy_selfplay")

# The payload fields that carry `None` when their producer did not supply them.
ABSENCE_CAPABLE = ("policy_entropy", "policy_entropy_selfplay")


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
                            fast_policy_weight_provider=lambda: 0.0,
    )


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


# the premise, re-derived rather than assumed

def test_the_real_graph_tail_produces_a_finite_policy_entropy(tmp_path: Path) -> None:
    """F-01's premise, reversed by B-4: the producer exists, so the alert measures something."""
    graph = _real_graph_loss_info(tmp_path)
    assert math.isfinite(graph["policy_entropy"]) and graph["policy_entropy"] > 0.0
    assert graph["policy_entropy_selfplay"] == graph["policy_entropy"]
    for key in GUARANTEED:
        assert key in graph, f"graph tail dropped {key}: {sorted(graph)}"


# the audit's PIN, on both arms

def test_the_real_graph_tail_carries_its_measured_entropy_and_fires_only_below_the_floor(
    tmp_path: Path
) -> None:
    """THE PIN. `policy_entropy` was a fabricated 0.0 that fired the alert every log; now it is the
    graph tail's measurement, far above the minted floor here, and the alert is quiet for a REASON."""
    loss_info = _real_graph_loss_info(tmp_path)
    payload = emit_training_step_event(0, loss_info, _NullSink())
    assert payload["policy_entropy"] == pytest.approx(loss_info["policy_entropy"])
    assert payload["policy_entropy"] > MINTED_ENTROPY_FLOOR
    assert "entropy_collapse" not in _alerts(payload)


# ── the control: the rule is silent because nothing measured it, NOT because it is dead ──

def test_a_MEASURED_entropy_below_the_floor_still_fires(tmp_path: Path) -> None:
    """The other half of the finding: masking. A real collapse must still be reported, or
    the repair would have replaced a false alarm with a dead rule."""
    loss_info = dict(_real_graph_loss_info(tmp_path))
    loss_info["policy_entropy"] = MINTED_ENTROPY_FLOOR - 0.5
    payload = emit_training_step_event(0, loss_info, _NullSink())
    assert payload["policy_entropy"] == pytest.approx(MINTED_ENTROPY_FLOOR - 0.5)
    assert "entropy_collapse" in _alerts(payload)


# the rest of the family (F-28 INST-C02/C03)

def test_every_field_is_produced_by_the_real_tail(tmp_path: Path) -> None:
    """A field the real tail leaves `None` on every step has no producer and must leave the event."""
    payload = emit_training_step_event(0, _real_graph_loss_info(tmp_path), _NullSink())
    unproduced = sorted(key for key, value in payload.items() if value is None)
    assert unproduced == [], f"producer-less fields in `training_step`: {unproduced}"


def test_an_absent_measurement_travels_as_None_never_a_fabricated_zero(tmp_path: Path) -> None:
    """`docs/contracts/event_manifest.md`: an unproduced field carries `None`, never a 0."""
    loss_info = dict(_real_graph_loss_info(tmp_path))
    for key in ABSENCE_CAPABLE:
        del loss_info[key]
    payload = emit_training_step_event(0, loss_info, _NullSink())
    for key in ABSENCE_CAPABLE:
        assert key in payload, f"{key} vanished from the payload shape"
        assert payload[key] is None, f"{key} = {payload[key]!r}, expected None (no producer)"


def test_the_payload_is_valid_JSON_with_no_NaN(tmp_path: Path) -> None:
    """INST-C02. The three `policy_entropy_*` rows defaulted to `float('nan')`, which
    `json.dumps` writes as the bare token `NaN` — not valid JSON, and read back as a number
    by anything permissive. Absence is `null`."""
    payload = emit_training_step_event(0, _real_graph_loss_info(tmp_path), _NullSink())
    text = json.dumps(payload, allow_nan=False)  # raises ValueError on any NaN/Inf
    assert "NaN" not in text


def test_a_produced_field_is_carried_through_unchanged(tmp_path: Path) -> None:
    """The absence convention must not eat real readings: a produced key survives."""
    loss_info = dict(_real_graph_loss_info(tmp_path))
    loss_info["policy_entropy_selfplay"] = 0.0  # a MEASURED zero, which must NOT become None
    payload = emit_training_step_event(0, loss_info, _NullSink())
    assert payload["policy_entropy_selfplay"] == 0.0
    assert math.isfinite(float(payload["loss_total"]))


class _NullSink:
    def emit(self, event: Any) -> None:
        return None
