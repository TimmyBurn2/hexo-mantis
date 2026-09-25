"""B-4: the graph step MEASURES the model's policy entropy; the collapse rules read it."""
from __future__ import annotations

import math

import _microbatch_harness as H
import torch

from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.monitor.rules import WARN_RULE_INPUTS, rule_input_absent
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.losses import ragged_policy_ce_and_entropies


def test_the_models_entropy_is_the_segment_entropy_of_its_softmax() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0, 5.0, -5.0], dtype=torch.float32)
    offsets = torch.tensor([0, 3, 5])
    target = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    _ce, _h_target, h_model = ragged_policy_ce_and_entropies(logits, target, offsets)
    p2 = torch.softmax(torch.tensor([5.0, -5.0]), 0)
    h2 = float(-(p2 * p2.log()).sum())
    assert math.isclose(float(h_model), (math.log(3.0) + h2) / 2.0, rel_tol=1e-5)


def test_the_graph_step_publishes_policy_entropy_in_loss_info_and_the_step_event(tmp_path) -> None:
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=0)
    caps = H.non_binding_caps(replay.wire)
    info = run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
        sample_threads_provider=lambda: 1,
        fast_policy_weight_provider=lambda: 0.0,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    assert info["policy_entropy"] > 0.0, "a tiny net over a uniform buffer is nowhere near one-hot"
    assert info["policy_entropy_selfplay"] == info["policy_entropy"]
    event = sink.named("trainer_step")[-1]
    assert event["policy_entropy"] == info["policy_entropy"]
    # The rules' inputs are now PRESENT on a payload built from this loss_info.
    assert not rule_input_absent("entropy_collapse", info)
    assert not rule_input_absent("selfplay_entropy_collapse", info)
    assert WARN_RULE_INPUTS["entropy_collapse"] == ("policy_entropy",)
