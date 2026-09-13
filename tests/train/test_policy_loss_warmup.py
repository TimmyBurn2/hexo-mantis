"""The value warm-up (R350(b)(iii)): weight 0 for the first N steps, reported; the BC route refuses it."""
from __future__ import annotations

import pytest
import torch

from mantis.train.losses import policy_loss_weight_at
from mantis.train.trainer.core import TrainHParams


@pytest.mark.parametrize(("step", "warmup", "weight"), [
    (0, 0, 1.0), (5, 0, 1.0), (0, 3, 0.0), (2, 3, 0.0), (3, 3, 1.0), (10, 3, 1.0),
])
def test_the_weight_is_zero_below_the_warm_up_and_one_from_it_on(step: int, warmup: int, weight: float) -> None:
    assert policy_loss_weight_at(step, warmup) == weight


def test_a_negative_step_or_warm_up_is_refused() -> None:
    with pytest.raises(ValueError):
        policy_loss_weight_at(-1, 0)
    with pytest.raises(ValueError):
        policy_loss_weight_at(0, -1)


def test_the_hparams_read_the_block_and_refuse_its_absence(mk_config) -> None:
    config = mk_config()
    config["train"]["policy_loss_weight_schedule"] = {"warmup_steps": 7}
    assert TrainHParams.from_config(config).policy_loss_warmup_steps == 7
    del config["train"]["policy_loss_weight_schedule"]
    with pytest.raises(KeyError):
        TrainHParams.from_config(config)


def _step_once(trainer, buf, *, replay_n: int = 8):
    import _microbatch_harness as H  # noqa: PLC0415 — the tests/train rootdir harness
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import run_declared_train_step

    replay = H.ReplayWireBuffer(buf, replay_n)
    return run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=replay_n, augment=False, recency_weight=0.0,
        recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
        sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)


def test_a_warm_up_step_moves_the_value_head_and_not_the_policy_head(tmp_path) -> None:
    """Real step: at weight 0 the policy head is byte-untouched, the value head moves; after, both move."""
    import _microbatch_harness as H  # noqa: PLC0415 — the tests/train rootdir harness

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, policy_loss_warmup_steps=1)
    policy_before = {k: v.detach().clone() for k, v in trainer.model.policy_head.state_dict().items()}
    value_before = {k: v.detach().clone() for k, v in trainer.model.value_head.state_dict().items()}

    first = _step_once(trainer, buf)
    assert first["policy_loss"] > 0.0, "the reported policy loss must stay the unweighted CE"
    assert all(torch.equal(trainer.model.policy_head.state_dict()[k], v) for k, v in policy_before.items()), (
        "the policy head moved during the warm-up")
    assert any(not torch.equal(trainer.model.value_head.state_dict()[k], v) for k, v in value_before.items()), (
        "the value head did not move during the warm-up")

    second = _step_once(trainer, buf)
    assert second["policy_loss"] > 0.0
    assert any(not torch.equal(trainer.model.policy_head.state_dict()[k], v) for k, v in policy_before.items()), (
        "the policy head did not move once the warm-up ended")
    weights = [e["policy_loss_weight"] for e in sink.named("trainer_step")]
    assert weights == [0.0, 1.0]


def test_the_bc_route_refuses_a_config_with_a_warm_up(mk_config) -> None:
    from mantis.train.pretrain.graph_route import GraphPretrainError, refuse_policy_warm_up

    config = mk_config()
    refuse_policy_warm_up(config)
    config["train"]["policy_loss_weight_schedule"] = {"warmup_steps": 2000}
    with pytest.raises(GraphPretrainError, match="warmup_steps is 2000"):
        refuse_policy_warm_up(config)
