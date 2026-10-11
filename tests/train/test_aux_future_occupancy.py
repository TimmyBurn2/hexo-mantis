"""The future-occupancy head's training path: its rows, the planted break, the split, the pairing and the held-out read."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import _microbatch_harness as H
from mantis._engine import HexgBuffer
from mantis.config.loader import load_config
from mantis.config.resolve.heldout_gap import HeldoutGapSpec
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.config.schema import RunConfig
from mantis.model.gnn_v2_future import FUTURE_PRIOR_LOGITS
from mantis.train import losses
from mantis.train.coordinator.dispatch import run_declared_eval_step, run_declared_train_step
from mantis.train.events import heldout_gap_event
from mantis.train.heldout import HeldoutSlice
from mantis.train.trainer.core import TrainHParams
from mantis.util.hashing import sha256_file

_REPO = Path(__file__).resolve().parents[2]


def _step(trainer: Any, replay: Any, m: int) -> dict[str, Any]:
    caps = H.caps_for_exactly(replay.wire, m)
    run_declared_train_step(trainer, replay, H.GSPEC, batch_size=8, augment=False,
                            caps_provider=lambda: MicrobatchCapsSpec(*caps), sample_threads_provider=lambda: 1)
    return trainer._sink.named("trainer_step")[-1]


def test_the_real_step_trains_the_head_on_the_rows_that_carry_the_field(tmp_path) -> None:
    """Producer test: rows seen are the field-bearing rows, the CE sits in the loss at `w`, and the head moves."""
    replay = H.ReplayWireBuffer(H.future_graph_buffer(), 8)
    trainer = H.future_graph_trainer(tmp_path, sink=H.SpySink(), w=0.5)
    before = {k: v.detach().clone() for k, v in trainer.model.future_head.state_dict().items()}
    event = _step(trainer, replay, 1)
    present = int(np.asarray(replay.targets.future_valid).sum())
    assert present == 4 and event["aux_future_rows"] == present
    assert 0.0 < event["aux_future_ce"] < 2.0 and event["aux_future_weight"] == 0.5
    assert 0.0 < event["aux_future_base_ce"] < np.log(3)
    assert 0.0 < event["aux_future_row_base_ce"] <= event["aux_future_base_ce"] + 1e-9, "a row's own marginal is the stricter base"
    assert event["loss"] == pytest.approx(event["policy_loss"] + event["value_loss"] + 4.0 * event["aux_soft_policy_loss"]
                                          + 0.5 * event["aux_future_ce"], rel=1e-5)
    after = trainer.model.future_head.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), "the head did not move"


def test_the_first_steps_ce_is_the_prior_read_against_each_rows_labels(tmp_path) -> None:
    """Oracle: at step 0 every cell's logits are the prior, so the CE is the per-row mean of −log softmax(prior)[label] over field rows."""
    ragged = HexgBuffer(16, H.GRAPH_ENCODING, 128)
    stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 1, -1), (1, 2, 1)]
    for i in range(8):
        future = [(q, 4 + (q + i) % 2, 1 if k % 4 < 2 else -1) for k, q in enumerate(range(-10, 10))] if i % 3 else None
        ragged.push_graph_position(stones[:2 + i % 4], [(3, 0, 0.6), (1, 1, 0.4)], 1, 2, 2 + i % 4, True, 1.0, True, 40,
                                   game_id=i, future_stones=future)
    replay = H.ReplayWireBuffer(ragged, 8)
    sink = H.SpySink()
    run_declared_train_step(H.future_graph_trainer(tmp_path, sink=sink), replay, H.GSPEC, batch_size=8, augment=False,
                            caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
                            sample_threads_provider=lambda: 1)
    (event,) = sink.named("trainer_step")
    nll = -torch.log_softmax(torch.tensor(FUTURE_PRIOR_LOGITS, dtype=torch.float64), dim=0).numpy()
    labels, valid = np.asarray(replay.targets.future_target), np.asarray(replay.targets.future_valid)
    offsets = np.asarray(replay.wire.legal_offsets)
    assert len(set(np.diff(offsets)[valid != 0].tolist())) > 1, "the oracle needs rows of different legal-set sizes"
    per_row = [nll[labels[offsets[g]:offsets[g + 1]]].mean() for g in range(len(valid)) if valid[g]]
    per_node = nll[np.concatenate([labels[offsets[g]:offsets[g + 1]] for g in range(len(valid)) if valid[g]])].mean()
    assert event["aux_future_ce"] == pytest.approx(float(np.mean(per_row)), rel=1e-6)
    assert event["aux_future_ce"] != pytest.approx(float(per_node), rel=1e-6), "the two normalisations must differ here"


def test_absent_rows_are_left_out_of_the_head_and_its_count(tmp_path, monkeypatch) -> None:
    """PLANTED BREAK: a marginal that counts every row as carrying the field shifts the rows seen and the CE; it reds here."""
    replay = H.ReplayWireBuffer(H.future_graph_buffer(), 8)
    honest = _step(H.future_graph_trainer(tmp_path / "a", sink=H.SpySink()), replay, 1)
    real = losses.future_marginal
    monkeypatch.setattr(losses, "future_marginal", lambda labels, valid, offsets: real(labels, np.ones_like(valid), offsets))
    broken = _step(H.future_graph_trainer(tmp_path / "b", sink=H.SpySink()), replay, 1)
    assert (honest["aux_future_rows"], broken["aux_future_rows"]) == (4, 8)
    assert broken["aux_future_ce"] != pytest.approx(honest["aux_future_ce"])


@pytest.mark.parametrize("m", [2, 4])
def test_the_parts_sum_to_the_whole_steps_ce(tmp_path, m: int) -> None:
    """The whole step's row count divides every part, so a split step reads the un-split CE."""
    replay = H.ReplayWireBuffer(H.future_graph_buffer(), 8)
    whole = _step(H.future_graph_trainer(tmp_path / "one", sink=H.SpySink()), replay, 1)
    split = _step(H.future_graph_trainer(tmp_path / f"m{m}", sink=H.SpySink()), replay, m)
    assert split["aux_future_ce"] == pytest.approx(whole["aux_future_ce"], rel=1e-5)
    assert split["aux_future_rows"] == whole["aux_future_rows"]


def test_a_step_with_no_field_row_reads_none_and_trains_no_head(tmp_path) -> None:
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(), 8)
    trainer = H.future_graph_trainer(tmp_path, sink=H.SpySink())
    before = {k: v.detach().clone() for k, v in trainer.model.future_head.state_dict().items()}
    event = _step(trainer, replay, 1)
    assert (event["aux_future_rows"], event["aux_future_ce"], event["aux_future_base_ce"]) == (0, None, None)
    assert all(torch.equal(before[k], v) for k, v in trainer.model.future_head.state_dict().items())


def test_without_the_head_the_rows_are_absent_not_zero(tmp_path) -> None:
    replay = H.ReplayWireBuffer(H.future_graph_buffer(), 8)
    event = _step(H.soft_policy_graph_trainer(tmp_path, sink=H.SpySink()), replay, 1)
    assert not any(k.startswith("aux_future") for k in event)


def test_the_head_its_kind_and_its_weight_are_one_fact(tmp_path) -> None:
    with pytest.raises(ValueError, match="one fact"):
        H.future_graph_trainer(tmp_path, aux_future_occupancy=None)
    with pytest.raises(ValueError, match="one fact"):
        H.soft_policy_graph_trainer(tmp_path, aux_future_occupancy=0.5)
    config = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    config["identity"]["arch_kind"] = "GnnArchV2SoftPolicyFuture"
    config["model"]["aux_soft_policy"] = {"target_temperature": 4.0, "weight": 4.0}
    with pytest.raises(ValueError, match="future-occupancy head but model.aux_future_occupancy is None"):
        RunConfig.model_validate(config)
    config["model"]["aux_future_occupancy"] = {"w": 0.5}
    assert TrainHParams.from_config(RunConfig.model_validate(config).model_dump()).aux_future_occupancy == 0.5
    config["identity"]["arch_kind"] = "GnnArchV2SoftPolicy"
    with pytest.raises(ValueError, match="does not carry the future-occupancy head"):
        RunConfig.model_validate(config)


def test_the_eval_step_reads_the_heads_ce_against_its_base_rate(tmp_path) -> None:
    """Forward-only: the same CE the step trains on, beside the CE of the batch's own label marginal."""
    replay = H.ReplayWireBuffer(H.future_graph_buffer(), 8)
    trainer = H.future_graph_trainer(tmp_path, sink=H.SpySink())
    caps = H.caps_for_exactly(replay.wire, 2)
    out = run_declared_eval_step(trainer, replay, H.GSPEC, batch_size=8, caps_provider=lambda: MicrobatchCapsSpec(*caps),
                                 sample_threads_provider=lambda: 1)
    rows, marginal = losses.future_marginal(replay.targets.future_target, replay.targets.future_valid,
                                            np.asarray(replay.wire.legal_offsets))
    assert out["aux_future_rows"] == rows == 4
    assert out["aux_future_base_ce"] == pytest.approx(losses.future_base_ce(marginal))
    assert out["aux_future_row_base_ce"] == pytest.approx(losses.future_row_base_ce(
        replay.targets.future_target, replay.targets.future_valid, np.asarray(replay.wire.legal_offsets)))
    event = _step(trainer, replay, 1)
    assert out["aux_future_ce"] == pytest.approx(event["aux_future_ce"], rel=1e-4), "the read is the trained CE's"


@pytest.mark.parametrize("field", [True, False])
def test_the_heldout_slice_reads_the_heads_ce_and_base_rate_or_says_none(tmp_path, field: bool) -> None:
    """The run's frozen-slice read carries the head's CE beside its base rate onto `heldout_gap`; a slice without the field reads None."""

    ring = tmp_path / "slice.hexg"
    (H.future_graph_buffer() if field else H.uniform_graph_buffer()).save_to_path(str(ring))
    spec = HeldoutGapSpec(str(ring), sha256_file(ring), 2, 7, 1)
    heldout = HeldoutSlice.open(spec, encoding="gnn_axis_v1", visit_capacity=128, capacity=64)
    trainer = H.future_graph_trainer(tmp_path, sink=H.SpySink())
    read = heldout.read(trainer, H.GSPEC, batch_size=8, caps_provider=lambda: MicrobatchCapsSpec(10**8, 10**8),
                        sample_threads_provider=lambda: 1)
    event = heldout_gap_event(step=1, slice_=heldout, heldout=read, train_policy=None, train_value=None, train_steps=0,
                              wall_ms=1.0)
    if field:
        assert 0.0 < event["aux_future_ce"] and 0.0 < event["aux_future_row_base_ce"] <= event["aux_future_base_ce"] + 1e-9 < 1.1
    else:
        assert (event["aux_future_ce"], event["aux_future_base_ce"], event["aux_future_row_base_ce"]) == (None, None, None)
    soft = heldout.read(H.soft_policy_graph_trainer(tmp_path / "s", sink=H.SpySink()), H.GSPEC, batch_size=8,
                        caps_provider=lambda: MicrobatchCapsSpec(10**8, 10**8), sample_threads_provider=lambda: 1)
    assert "aux_future_ce" not in soft, "a net without the head reads no row at all"
