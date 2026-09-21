"""`model.gnn` (contract v35): the trunk's shape is a mint fact with one reader, refused by arch first, and a stamp or a resume that claims another shape is refused by name."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.census import production_configs
from mantis.config.loader import discover_configs
from mantis.config.resolve.arch_scope import ArchScopedKeyOutsideItsArchError
from mantis.config.resolve.gnn_widths import MissingGnnWidthsError, resolve_gnn_widths
from mantis.encoding import lookup
from mantis.model import GnnArchV2, arch_from_spec_and_config, build_net, declared_gnn_widths
from mantis.train.checkpoints import (
    CheckpointStampError,
    ResumeIdentityMismatchError,
    _refuse_identity_drift,
    resume_trainer,
    save_checkpoint,
)
from mantis.train.trainer.core import Trainer, build_param_groups

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = discover_configs(_REPO / "configs")
_PRODUCTION = production_configs(_REPO)


@pytest.mark.parametrize("path", _CONFIGS, ids=[p.name for p in _CONFIGS])
def test_every_shipped_config_states_its_shape_and_the_two_readers_agree(path: Path) -> None:
    """The resolver (the refusing front door) and the build's parse read the same block."""
    dump = load_config(path).model_dump()
    spec = resolve_gnn_widths(dump)
    assert declared_gnn_widths(dump) == {"hidden": spec.hidden, "num_layers": spec.num_layers}
    arch = arch_from_spec_and_config(lookup(dump["identity"]["encoding"]), dump)
    assert (arch.hidden, arch.num_layers) == (spec.hidden, spec.num_layers)


@pytest.mark.parametrize("path", _PRODUCTION, ids=[p.name for p in _PRODUCTION])
def test_the_shape_is_a_mint_row_the_build_honours(path: Path) -> None:
    """A production config re-minted to 6 × 192 builds a 6 × 192 net of its own declared kind — the shape is a row, not a constant."""
    dump = load_config(path).model_dump()
    dump["model"]["gnn"] = {"hidden": 192, "num_layers": 6}
    arch = arch_from_spec_and_config(lookup(dump["identity"]["encoding"]), dump)
    assert type(arch).__name__ == dump["identity"]["arch_kind"] and (arch.hidden, arch.num_layers) == (192, 6)
    net = build_net(arch)
    assert net.representation.output_dim == 6 * 192


def test_the_resolver_refuses_a_foreign_arch_before_it_reads_the_block() -> None:
    dump = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    dump["identity"]["representation"] = "grid"
    with pytest.raises(ArchScopedKeyOutsideItsArchError, match="ARCH-SCOPED"):
        resolve_gnn_widths(dump)


def test_absence_is_a_named_error_at_every_level() -> None:
    dump = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    with pytest.raises(MissingGnnWidthsError, match="no `model` section"):
        resolve_gnn_widths({k: v for k, v in dump.items() if k != "model"})
    dump["model"]["gnn"] = {"hidden": 128}
    with pytest.raises(MissingGnnWidthsError, match="num_layers"):
        resolve_gnn_widths(dump)


def test_the_build_refuses_a_declared_representation_the_spec_does_not_have() -> None:
    dump = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    dump["identity"]["representation"] = "grid"
    with pytest.raises(ValueError, match="identity.representation='grid'"):
        arch_from_spec_and_config(lookup(dump["identity"]["encoding"]), dump)


def _tiny(dump: dict) -> GnnArchV2:
    spec = lookup(dump["identity"]["encoding"])
    return GnnArchV2(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                     hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)


def test_the_writer_refuses_a_stamp_whose_config_claims_another_shape(tmp_path: Path) -> None:
    """A 4 × 128 config over an 8 × 1 net is a provenance lie; the same save with the truth lands."""
    dump = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    arch = _tiny(dump)
    kwargs = {"encoding_name": dump["identity"]["encoding"], "run_id": "shape", "arch": arch}
    with pytest.raises(CheckpointStampError, match="model.gnn.hidden: stamped arch=8, config=128"):
        save_checkpoint(model=build_net(arch), optimizer=None, scaler=None, scheduler=None, step=0,
                        config=dump, metadata_kwargs=kwargs, checkpoint_dir=tmp_path, kind="weights")
    dump["model"]["gnn"] = {"hidden": 8, "num_layers": 1}
    assert save_checkpoint(model=build_net(arch), optimizer=None, scaler=None, scheduler=None, step=0,
                           config=dump, metadata_kwargs=kwargs, checkpoint_dir=tmp_path, kind="weights").exists()


def test_a_resume_whose_launch_moves_the_shape_is_refused_by_name(tmp_path: Path) -> None:
    dump = load_config(_REPO / "configs" / "dev_example.yaml").model_dump()
    arch = _tiny(dump)
    dump["model"]["gnn"] = {"hidden": 8, "num_layers": 1}
    net = build_net(arch)
    opt = torch.optim.AdamW(build_param_groups(net, 1e-4), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-5)
    path = save_checkpoint(model=net, optimizer=opt, scaler=None, scheduler=sched, step=1, config=dump,
                           metadata_kwargs={"encoding_name": dump["identity"]["encoding"], "run_id": "shape", "arch": arch},
                           checkpoint_dir=tmp_path, kind="full")
    launch = {"model": {"gnn": {"hidden": 192, "num_layers": 6}}}
    with pytest.raises(ResumeIdentityMismatchError, match="model.gnn.hidden: checkpoint=8, resume=192"):
        resume_trainer(Trainer, path, config_overrides=launch)
    trainer = resume_trainer(Trainer, path)
    assert (trainer.arch.hidden, trainer.arch.num_layers) == (8, 1)


def test_the_shape_refusal_does_not_wait_on_an_identity_block(tmp_path: Path) -> None:
    """A stamp with no `identity` block (pre-identity artefact) still refuses a launch that moves the widths: the shape check needs only the stamped arch."""
    arch = _tiny(load_config(_REPO / "configs" / "dev_example.yaml").model_dump())
    effective = {"identity": {"representation": "graph"}, "model": {"gnn": {"hidden": 192, "num_layers": 6}}}
    with pytest.raises(ResumeIdentityMismatchError, match="model.gnn.hidden: checkpoint=8, resume=192"):
        _refuse_identity_drift(tmp_path / "x.ckpt", {}, effective, arch)
