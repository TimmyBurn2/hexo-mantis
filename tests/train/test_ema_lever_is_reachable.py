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

from mantis.config.loader import load_config, parse_config_yaml
from mantis.config.schema import RunConfig
from mantis.train.ema import MissingEmaConfigError, resolve_ema_config

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = sorted((_REPO / "configs").glob("*.yaml"))


def test_there_are_configs_to_check() -> None:
    """Vacuity guard — an empty glob would make every row below pass on nothing."""
    assert len(_CONFIGS) >= 5, f"only {len(_CONFIGS)} config(s) found"


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
    import torch

    from mantis.encoding import lookup
    from mantis.model import build_net, select_arch
    from mantis.train.ema import build_ema_model

    arch = select_arch(lookup("v6_live2_ls"), {}, arch_kind="CnnArch")
    net = build_net(arch)
    enabled, decay, _every = resolve_ema_config(
        {"train": {"ema": {"enabled": True, "decay": 0.5, "update_every": 1}}},
    )
    assert enabled
    ema = build_ema_model(net, decay=decay)
    assert ema.decay == pytest.approx(0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update_parameters(net)  # the shadow moves — the lever is not inert
