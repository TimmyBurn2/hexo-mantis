"""O-SMOKE — end-to-end launch-path smoke (INTEGRATION tier), on the production regime.

The minted wiring config (Gumbel, its completed target, the playout cap, the aux head) builds its
own net and trainer, runs 2 steps through `run_training_loop` on sparse rows, writes an envelope-v2
checkpoint, resumes from it, and shuts down clean on a simulated signal.
"""
from __future__ import annotations

import signal
from pathlib import Path

import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.config.loader import load_config
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.config.schema import SOFT_POLICY_ARCH_KINDS, RunConfig
from mantis.config.schema.core import derived_visit_capacity
from mantis.encoding import resolve_from_config
from mantis.model import arch_from_spec_and_config, build_net
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.checkpoints import CHECKPOINT_SCHEMA_VERSION, resume_trainer
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers
from mantis.train.loop import run_training_loop
from mantis.train.trainer.core import Trainer

pytestmark = pytest.mark.integration

WIRING = Path(__file__).resolve().parents[2] / "configs" / "smoke_wiring.yaml"


class _Rows:
    """The sink double: keeps every emitted row."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def emit(self, event: dict) -> None:
        self.rows.append(dict(event))


def _sparse_ring(run_config: RunConfig, encoding: str) -> HexgBuffer:
    """A ring at the config's own visit capacity, holding sparse rows with a tail and the playout cap's fast rows."""
    ring = HexgBuffer(64, encoding, derived_visit_capacity(run_config))
    for i in range(16):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        ring.push_graph_position(stones, [(2, 0, 0.5), (1, 1, 0.3)], 1, 30, 2 + i, i % 4 != 0,
                                 1.0 if i % 2 == 0 else -1.0, True, 10 + i, tail_mass=0.2)
    ring.seed_sampler(run_config.seed)
    return ring


def test_launch_path_smoke(tmp_path: Path) -> None:
    """Build → run 2 steps → write envelope-v2 ckpt → resume → clean shutdown on a signal."""
    run_config = load_config(WIRING)
    assert run_config.selfplay.search.kind == "gumbel" and run_config.model.aux_soft_policy is not None
    config = run_config.model_dump()
    spec = resolve_from_config(config)
    arch = arch_from_spec_and_config(spec, config)
    assert type(arch).__name__ in SOFT_POLICY_ARCH_KINDS, f"the wiring config must build the aux head; got {arch}"
    sink = _Rows()
    tr = Trainer(build_net(arch), config, arch=arch, checkpoint_dir=tmp_path, sink=sink)

    ring = _sparse_ring(run_config, spec.name)
    train = run_config.train
    state = ShutdownState()
    seen = {"n": 0}

    def one_step():
        run_declared_train_step(
            tr, ring, spec, batch_size=train.batch_size, augment=train.augment,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=train.microbatch_caps.max_edges,
                                                     max_nodes=train.microbatch_caps.max_nodes),
            sample_threads_provider=lambda: 1,
        )
        seen["n"] += 1
        if seen["n"] >= 2:
            state.shutdown_save = True  # request the final save; the loop observes it

    run_training_loop(trainer=tr, shutdown_state=state, step_fn=one_step, max_steps=10)
    assert tr.step == 2, "the loop must have driven exactly 2 training steps"
    aux = [row.get("aux_soft_policy_loss") for row in sink.rows if row["event"] == "trainer_step"]
    assert len(aux) == 2 and all(a is not None and torch.isfinite(torch.tensor(a)) and a > 0 for a in aux), (
        f"every step must train the aux head on its soft target; got {aux}"
    )

    # the loop wrote a FINAL envelope-v2 checkpoint on shutdown_save
    ckpts = list(tmp_path.glob("*.ckpt"))
    assert ckpts, "run_training_loop must write an envelope-v2 checkpoint on shutdown_save"
    ckpt = ckpts[0]
    prefix = f"{run_config.run_id}_00000002_"
    assert ckpt.name.startswith(prefix), f"unexpected v2 filename {ckpt.name}"
    payload = torch.load(ckpt, weights_only=True)
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 2
    assert payload["kind"] == "full"

    # resume from it (build_net(metadata.arch) + restore optim/scaler/step)
    tr2 = resume_trainer(Trainer, ckpt, fallback_config=config, declared_encoding=None)
    assert tr2.loaded_from_full_checkpoint is True
    assert tr2.step == 2
    assert len(tr2.optimizer.param_groups) == 2

    # clean shutdown on a SIMULATED signal (save-then-exit choreography)
    orig_int = signal.getsignal(signal.SIGINT)
    try:
        state2 = ShutdownState()
        install_signal_handlers(state2)
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)  # simulate one SIGINT
        assert state2.shutdown_save is True and state2.running is False
        # a 0-step loop over the shutdown-flagged state saves once and returns clean.
        final = run_training_loop(trainer=tr2, shutdown_state=state2)
        assert final.shutdown_save is True
        assert len(list(tmp_path.glob(f"{prefix}*.ckpt"))) >= 1
    finally:
        signal.signal(signal.SIGINT, orig_int)
