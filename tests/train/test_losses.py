"""O-CHAIN — the chain_head smooth-L1 loss self-reports its fire-rate in-run (WP10 Slice-2 gate).

LAW-07/LAW-18 (WP9-owed): a lever under test logs its own fire-rate in-run. The `chain_head`
Q13-aux smooth-L1 loss, when `aux_chain_weight > 0`, fires AND publishes its fire-rate on the
`chain_planes` target through the injected sink; at weight 0 it reports fire-rate 0 (a disabled
lever stays VISIBLE). Bites F-10: an aux loss reading the wrong sub-buffer / firing at weight 0
with no self-report. IMPL-authored gating oracle (non-⊕⊕); PREREG "Non-⊕⊕ oracle verdicts".
"""
from __future__ import annotations

import numpy as np
import torch

from mantis.encoding import lookup
from mantis.model import CnnArch, build_net
from mantis.train.losses import (
    chain_loss_with_fire_rate,
    chain_target_fire_rate,
    compute_chain_loss,
)


def _chain_pred_target(b: int = 4, planes: int = 6, hw: int = 19, *, active_rows: int | None = None):
    """A (B, 6, H, W) prediction + target. `active_rows` rows carry a nonzero (firing) target;
    the rest are all-zero (silent). None → every row active."""
    pred = torch.zeros(b, planes, hw, hw)
    target = torch.zeros(b, planes, hw, hw)
    n_active = b if active_rows is None else active_rows
    if n_active > 0:
        target[:n_active] = 0.5  # normalized chain values live in [0, 1]
    return pred, target


# ── the fire-rate self-report ────────────────────────────────────────────────────────────
def test_chain_loss_fires_and_reports_at_positive_weight(spy_sink):
    """O-CHAIN(a) — weight>0 → the loss FIRES (a real tensor) and the sink carries an
    `aux_chain_loss` event with fired=True + a positive fire-rate on the chain target."""
    pred, target = _chain_pred_target(active_rows=4)  # all 4 rows carry signal
    loss = chain_loss_with_fire_rate(pred, target, weight=0.5, sink=spy_sink, step=7)
    assert loss is not None and torch.is_tensor(loss)
    reports = spy_sink.named("aux_chain_loss")
    assert reports, "the chain loss must self-report through the sink"
    r = reports[-1]
    assert r["fired"] is True
    assert r["weight"] == 0.5
    assert r["fire_rate"] > 0.0
    assert r["step"] == 7


def test_chain_loss_reports_zero_at_weight_zero(spy_sink):
    """O-CHAIN(b) — weight==0 → the lever is OFF: no loss is returned and the report publishes
    fire_rate 0.0 (the disabled lever is still VISIBLE, not silent)."""
    pred, target = _chain_pred_target(active_rows=4)
    loss = chain_loss_with_fire_rate(pred, target, weight=0.0, sink=spy_sink, step=1)
    assert loss is None
    reports = spy_sink.named("aux_chain_loss")
    assert reports and reports[-1]["fired"] is False
    assert reports[-1]["fire_rate"] == 0.0


def test_fire_rate_tracks_the_chain_target_not_a_wrong_buffer():
    """O-CHAIN(c) — the fire-rate is measured on the `chain_planes` TARGET: an all-zero target
    reports 0, a fully-active target reports 1, a half-active target reports 0.5 (bites the
    F-10 wrong-sub-buffer class — a loss reading the wrong buffer could not track this)."""
    _, all_zero = _chain_pred_target(b=4, active_rows=0)
    assert chain_target_fire_rate(all_zero) == 0.0
    _, all_active = _chain_pred_target(b=4, active_rows=4)
    assert chain_target_fire_rate(all_active) == 1.0
    _, half = _chain_pred_target(b=4, active_rows=2)
    assert chain_target_fire_rate(half) == 0.5


def test_chain_loss_math_is_smooth_l1():
    """O-CHAIN(d) — `chain_loss_with_fire_rate`'s loss value equals `compute_chain_loss` (the
    self-report seam does NOT perturb the loss math)."""
    torch.manual_seed(0)
    pred = torch.randn(3, 6, 19, 19)
    target = torch.rand(3, 6, 19, 19)
    reported = chain_loss_with_fire_rate(pred, target, weight=1.0, sink=None)
    direct = compute_chain_loss(pred, target)
    assert torch.allclose(reported, direct)


# ── in-run through the Trainer (the report is genuinely emitted DURING a training step) ───
def test_chain_fire_rate_emitted_during_trainer_step(spy_sink):
    """O-CHAIN(e) — the report is IN-RUN: a dense training step with aux_chain_weight>0 emits
    the `aux_chain_loss` event through the trainer's injected sink."""
    from mantis.model import arch_from_spec_and_config
    from mantis.train.trainer.core import Trainer, TrainHParams

    spec = lookup("v6_live2_ls")
    arch = arch_from_spec_and_config(spec, {})
    net = build_net(arch)
    config = {
        "schema_version": 1, "run_id": "run5", "seed": 7,
        "identity": {"encoding": "v6_live2_ls", "representation": "grid"},
        "eval": {"random_model_sims": 1, "sealbot_model_sims": 1},
        "selfplay": {"legal_move_radius_schedule": None},
    }
    hp = TrainHParams(fp16=False, lr_schedule="none", aux_chain_weight=0.3, checkpoint_interval=0)
    tr = Trainer(net, config, arch=arch, train_hparams=hp, sink=spy_sink)

    b, planes, hw = 3, int(spec.n_planes), int(spec.board_size)
    n_actions = hw * hw + 1
    states = np.zeros((b, planes, hw, hw), dtype=np.float32)
    policies = np.full((b, n_actions), 1.0 / n_actions, dtype=np.float32)
    outcomes = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    chain_planes = np.zeros((b, 6, hw, hw), dtype=np.float16)
    chain_planes[:] = 0.5  # every row carries chain signal → fire_rate 1.0
    tr.train_step_from_tensors(states, policies, outcomes, chain_planes=chain_planes)

    reports = spy_sink.named("aux_chain_loss")
    assert reports and reports[-1]["fired"] is True and reports[-1]["fire_rate"] > 0.0
