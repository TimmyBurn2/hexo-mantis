"""The auxiliary soft-policy head's training path (R366(b)): the target's construction, the step's rows, the LAW-07 planted break, the rows-and-head pairing, and the warm start onto the new kind."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from mantis.model import GnnArchV2, build_net, load_from_bc, net_param_hash
from mantis.train.losses import soft_policy_target
from mantis.train.trainer.core import TrainHParams

_REPO = Path(__file__).resolve().parents[2]


def _row_target(explicit: list[float], alpha: float, prior: list[float], temperature: float) -> torch.Tensor:
    n = len(prior)
    target = torch.tensor(explicit + [0.0] * (n - len(explicit)))
    mask = torch.tensor([1.0] * len(explicit) + [0.0] * (n - len(explicit)))
    return soft_policy_target(target, torch.tensor([0, n]), mask, torch.tensor([alpha]),
                              torch.tensor(prior), temperature)


def test_the_temperature_flattens_the_explicit_entries_and_the_tail_is_carried() -> None:
    """[0.5, 0.25, 0.25] at T = 4 becomes ∝ [0.5^¼, 0.25^¼, 0.25^¼] over the explicit mass; the tail keeps α over the prior."""
    prior = [0.1, 0.1, 0.1, 0.35, 0.35]
    soft = _row_target([0.4, 0.2, 0.2], 0.2, prior, 4.0)
    sharpened = torch.tensor([0.4, 0.2, 0.2]) ** 0.25
    expected_explicit = 0.8 * sharpened / sharpened.sum()
    assert torch.allclose(soft[:3], expected_explicit, atol=1e-6)
    assert torch.allclose(soft[3:], torch.tensor([0.1, 0.1]), atol=1e-6), "the tail is α over the prior's tail shape"
    assert soft.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert soft[0] < 0.4 and soft[1] > 0.2, "the soft target is flatter than the hard one on the explicit set"


def test_a_one_hot_row_and_a_tail_free_row_behave() -> None:
    one_hot = _row_target([1.0, 0.0, 0.0], 0.0, [0.2, 0.4, 0.4], 4.0)
    assert torch.allclose(one_hot, torch.tensor([1.0, 0.0, 0.0]))
    flat = _row_target([0.6, 0.4], 0.0, [0.5, 0.5], 4.0)
    assert flat.sum().item() == pytest.approx(1.0, abs=1e-6) and flat[0] > flat[1]


def test_a_temperature_at_or_below_one_is_refused() -> None:
    for t in (1.0, 0.5, float("nan")):
        with pytest.raises(ValueError, match="temperature must be > 1"):
            _row_target([0.6, 0.4], 0.0, [0.5, 0.5], t)


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


def test_the_real_step_trains_the_aux_head_and_publishes_its_rows(tmp_path) -> None:
    """Producer test (LAW-18): the aux CE, KL(hard‖soft) > 0, both heads' grad norms, and the aux head moves."""
    import _microbatch_harness as H  # noqa: PLC0415

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    sink = H.SpySink()
    trainer = H.soft_policy_graph_trainer(tmp_path, sink=sink, target_temperature=4.0, weight=4.0)
    aux_before = {k: v.detach().clone() for k, v in trainer.model.aux_policy_head.state_dict().items()}
    result = _step_once(trainer, buf)
    event = sink.named("trainer_step")[0]
    assert event["aux_soft_policy_loss"] > 0.0 and result["loss"] > 0.0
    assert event["aux_soft_policy_kl_hard_vs_soft"] > 0.0, "the soft target differs from the hard one on a spread row"
    assert (event["aux_soft_policy_target_temperature"], event["aux_soft_policy_weight"]) == (4.0, 4.0)
    assert event["policy_head_grad_norm"] > 0.0 and event["aux_policy_head_grad_norm"] > 0.0
    assert any(not torch.equal(trainer.model.aux_policy_head.state_dict()[k], v) for k, v in aux_before.items()), (
        "the aux head did not move")
    assert result["loss"] == pytest.approx(result["policy_loss"] + result["value_loss"] + 4.0 * event["aux_soft_policy_loss"], rel=1e-5)


def test_the_planted_break_a_dead_soft_target_is_caught_by_the_producer_row(tmp_path, monkeypatch) -> None:
    """LAW-07: a construction that returns the HARD target (a temperature of 1 in effect) makes the KL row read 0, which the producer test refuses."""
    import _microbatch_harness as H  # noqa: PLC0415
    from mantis.train.trainer import core as core_module

    def dead_target(policy_target, legal_offsets, explicit_mask, tail_mass, prior_probs, temperature):
        return core_module._hard_target(torch.log(prior_probs.clamp_min(1e-12)), type("I", (), {
            "policy_target": policy_target, "legal_offsets": legal_offsets,
            "explicit_mask": explicit_mask, "tail_mass": tail_mass})())

    monkeypatch.setattr(core_module, "soft_policy_target", dead_target)
    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    sink = H.SpySink()
    trainer = H.soft_policy_graph_trainer(tmp_path, sink=sink)
    _step_once(trainer, buf)
    kl = sink.named("trainer_step")[0]["aux_soft_policy_kl_hard_vs_soft"]
    assert kl == pytest.approx(0.0, abs=1e-6), "the planted break must read as a dead row"
    with pytest.raises(AssertionError):
        assert kl > 0.0


def test_the_rows_are_omitted_on_an_arch_without_the_head(tmp_path) -> None:
    import _microbatch_harness as H  # noqa: PLC0415

    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    sink = H.SpySink()
    _step_once(H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=0), buf)
    event = sink.named("trainer_step")[0]
    assert "aux_soft_policy_loss" not in event and "aux_policy_head_grad_norm" not in event


def test_the_head_and_its_rows_are_one_fact_at_the_trainer_and_at_the_schema(tmp_path) -> None:
    import _microbatch_harness as H  # noqa: PLC0415

    with pytest.raises(ValueError, match="carries auxiliary soft-policy head but model.aux_soft_policy is None"):
        H.soft_policy_graph_trainer(tmp_path, aux_soft_policy=None)
    with pytest.raises(ValueError, match="carries no auxiliary soft-policy head"):
        H.tiny_graph_trainer(tmp_path, aux_soft_policy=(4.0, 4.0))
    config = load_config(_REPO / "configs" / "run9.yaml").model_dump()
    config["model"]["aux_soft_policy"] = {"target_temperature": 4.0, "weight": 4.0}
    with pytest.raises(ValueError, match="carries no soft-policy head"):
        RunConfig.model_validate(config)
    config["identity"]["arch_kind"] = "GnnArchV2SoftPolicy"
    assert TrainHParams.from_config(RunConfig.model_validate(config).model_dump()).aux_soft_policy == (4.0, 4.0)
    config["model"]["aux_soft_policy"] = None
    with pytest.raises(ValueError, match="carries the auxiliary soft-policy head but model.aux_soft_policy is null"):
        RunConfig.model_validate(config)


def test_a_V2_parent_warm_starts_the_new_kind_when_reinit_names_the_fresh_head() -> None:
    """The parent never had `aux_policy_head`: named in `reinit` it stays fresh, every other tensor lands byte-equal; unnamed, the transfer is refused."""
    import _microbatch_harness as H  # noqa: PLC0415

    torch.manual_seed(3)
    parent = build_net(GnnArchV2(**{f: getattr(H.tiny_soft_policy_arch(), f) for f in ("in_dim", "edge_dim", "hidden", "num_layers", "policy_hidden", "value_hidden")}))
    torch.manual_seed(4)
    child = build_net(H.tiny_soft_policy_arch())
    fresh_aux = {k: v.clone() for k, v in child.aux_policy_head.state_dict().items()}
    with pytest.raises(RuntimeError, match="missing=\\['aux_policy_head"):
        load_from_bc(child, parent.state_dict(), reinit=[])
    report = load_from_bc(child, parent.state_dict(), reinit=["aux_policy_head"])
    assert set(report["reinit_keys"]) == {f"aux_policy_head.{k}" for k in fresh_aux}
    assert all(torch.equal(child.aux_policy_head.state_dict()[k], v) for k, v in fresh_aux.items())
    for key, tensor in parent.state_dict().items():
        assert torch.equal(child.state_dict()[key], tensor)
    assert net_param_hash(child) != net_param_hash(parent), "a different net: it carries a head the parent lacks"
