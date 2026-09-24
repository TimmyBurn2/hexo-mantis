"""AUDIT-1 F-06 / R332(d) — the EMA lever has an arming key, and every config states its posture.

THE DEFECT. `train/ema.py::resolve_ema_config` read FOUR names — a nested `ema` block and the
flat `ema_enabled` / `ema_decay` / `ema_update_every` — off a `RunConfig` that is
`extra="forbid"` and had none of them. So:

* EMA was OFF on every run that has ever been launched;
* NO CONFIG COULD TURN IT ON — a config carrying `ema_enabled: true` fails to load at all,
  because the schema forbids the key;
* nothing said so. The module's docstring calls it an "anti-colony lever (kept)", and a
  disabled lever and an absent one produce identical runs.

That is R1's silently-disabled class verbatim — the class the rule was written for.

WHAT THE ROW IS. `train.ema` is a REQUIRED block with three REQUIRED members. Required, not
optional: unlike `identity.arch_kind` (v13) and `identity.warm_start` (v14), which enter
production configs only at run6's mint, this is a POSTURE every run already has and was not
stating. Every committed config mints `enabled: false` EXPLICITLY, so the OFF is a minted value
rather than a code-side silence, and turning it on later changes one boolean.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import _microbatch_harness as H
from mantis.config.census import discovered_config_paths, production_configs
from mantis.config.loader import load_config, parse_config_yaml
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.config.schema import RunConfig
from mantis.model import state_dict_param_hash
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.ema import MissingEmaConfigError, resolve_ema_config
from mantis.train.trainer.core import Trainer

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = [_REPO / rel for rel in discovered_config_paths(_REPO)]


def test_there_are_configs_to_check() -> None:
    """Vacuity guard: the sweep covers the production census, which raises when it is empty."""
    assert set(production_configs(_REPO)) <= set(_CONFIGS)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_every_committed_config_states_its_ema_posture_explicitly(path: Path) -> None:
    """The posture is IN the file, as text. Read through the parser rather than the schema, so
    this sees what the operator wrote and not what pydantic could have supplied."""
    raw = parse_config_yaml(path)
    ema = raw["train"]["ema"]
    assert set(ema) == {"enabled", "decay", "update_every"}, (
        f"{path.name}: train.ema carries {sorted(ema)}; every member is required"
    )
    assert ema["enabled"] is False, (
        f"{path.name}: EMA is armed in a committed config. Arming it is a PREREG lever "
        "(R332(d)), not a config edit"
    )


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_the_lever_is_readable_from_every_committed_config(path: Path) -> None:
    """The load-bearing row: the resolver reaches the value, from a real config, through the
    real loader. Before the key existed this could not be written at all."""
    enabled, decay, update_every = resolve_ema_config(load_config(path).model_dump())
    assert enabled is False
    assert 0.0 <= decay < 1.0 and update_every >= 1


def test_a_config_CAN_arm_the_lever() -> None:
    """The other half, and the one that was impossible before: a config that says `true` both
    LOADS and reaches the resolver as True. A schema that forbade the key made every "EMA is
    kept" claim unfalsifiable."""
    dump = load_config(_CONFIGS[0]).model_dump()
    dump["train"]["ema"] = {"enabled": True, "decay": 0.99, "update_every": 5}
    RunConfig.model_validate(dump)  # the schema ACCEPTS an armed posture
    assert resolve_ema_config(dump) == (True, 0.99, 5)


def test_an_absent_block_RAISES_instead_of_resolving_to_off() -> None:
    """The planted break for the row. `resolve_ema_config` used to answer `(False, …)` for a
    config with no block — the same answer as a declared OFF — which is precisely why nobody
    could see that no config had one."""
    dump = load_config(_CONFIGS[0]).model_dump()
    del dump["train"]["ema"]
    with pytest.raises(MissingEmaConfigError, match="train.ema is absent"):
        resolve_ema_config(dump)


def test_the_schema_REFUSES_the_flat_keys_the_dead_reader_looked_for() -> None:
    """`extra="forbid"` is what made the old read unreachable, and it still holds: the flat
    names cannot come back as a second arming surface beside the block."""
    dump = load_config(_CONFIGS[0]).model_dump()
    dump["train"]["ema_enabled"] = True
    with pytest.raises(ValueError, match="ema_enabled"):
        RunConfig.model_validate(dump)


def test_the_trainer_builds_an_ema_model_only_when_the_config_arms_it() -> None:
    """End of the chain: the arming key moves the object it names. Without this the key could
    be read, registered and consumed by a line that does nothing."""
    from mantis.encoding import lookup
    from mantis.model import build_net, select_arch
    from mantis.train.ema import EmaModel

    arch = select_arch(lookup("gnn_axis_v1"), {}, arch_kind="GnnArch")
    net = build_net(arch)
    enabled, decay, _every = resolve_ema_config(
        {"train": {"ema": {"enabled": True, "decay": 0.5, "update_every": 1}}},
    )
    assert enabled
    ema = EmaModel(net, decay=decay)
    assert ema.decay == pytest.approx(0.5)
    seeded = {n: t.clone() for n, t in ema.state_dict().items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update_parameters(net)
    shadow = ema.state_dict()
    for name, _ in net.named_parameters():
        assert torch.allclose(shadow[name], seeded[name] + 0.5), f"{name}: the shadow did not mix at decay 0.5"


def _one_real_step(trainer: Trainer) -> None:
    buf = H.uniform_graph_buffer()
    buf.seed_sampler(H.SEED)
    replay = H.ReplayWireBuffer(buf, 8)
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=8, augment=False, recency_weight=0.0,
        caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
        sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)
    assert trainer.step == 1, "premise: the drive took exactly one optimizer step"


def test_an_ARMED_trainer_moves_its_shadow_on_a_real_step(tmp_path: Path) -> None:
    """Prove the armed trainer builds the shadow and a real training step updates it."""
    trainer = H.ema_graph_trainer(tmp_path, update_every=1)
    assert trainer.ema_model is not None, "an armed config built no EMA"
    seeded = state_dict_param_hash(trainer.ema_model.state_dict())
    _one_real_step(trainer)
    assert state_dict_param_hash(trainer.ema_model.state_dict()) != seeded, "the step never updated the shadow"


def test_a_DISARMED_trainer_builds_no_shadow_and_deploys_the_learner(tmp_path: Path) -> None:
    """Prove the declared OFF builds no EMA and the deploy weights stay the learner's after a step."""
    trainer = H.tiny_graph_trainer(tmp_path)
    assert resolve_ema_config(H.graph_config())[0] is False, "premise: the fixture declares EMA off"
    assert trainer.ema_model is None, "a disarmed config built an EMA"
    _one_real_step(trainer)
    assert trainer.ema_model is None
    learner = state_dict_param_hash(trainer.model.state_dict())
    assert state_dict_param_hash(trainer.inference_state_dict()) == learner
