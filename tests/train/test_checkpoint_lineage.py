"""A stamp's lineage: written and read back, absent reading empty, kept by the strip and a resume, inherited by a warm start, refused when malformed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import (
    CheckpointStampError,
    load_checkpoint,
    resume_trainer,
    save_checkpoint,
    stamped_lineage,
    strip_and_restamp,
)

TAG = "six-0000000-gen1"


def _save(directory: Path, net: torch.nn.Module, opt_sched: Any, config: dict[str, Any], meta: dict[str, Any], *,
          lineage: Any = None, step: int = 10) -> Path:
    opt, scaler, sched = opt_sched
    mk = dict(meta) if lineage is None else {**meta, "lineage": lineage}
    return save_checkpoint(model=net, optimizer=opt, scaler=scaler, scheduler=sched, step=step, config=config,
                           metadata_kwargs=mk, checkpoint_dir=directory)


def test_the_lineage_round_trips_and_an_absent_one_reads_empty(tmp_path, tiny_net, optim_scaler_sched, valid_config,
                                                               metadata_kwargs):
    tagged = _save(tmp_path, tiny_net, optim_scaler_sched, valid_config, metadata_kwargs, lineage=[TAG])
    plain = _save(tmp_path, tiny_net, optim_scaler_sched, valid_config, metadata_kwargs, step=20)
    assert load_checkpoint(tagged, declared_encoding=None).metadata.lineage == (TAG,)
    assert load_checkpoint(plain, declared_encoding=None).metadata.lineage == ()
    assert stamped_lineage(tagged) == (TAG,)
    assert stamped_lineage(plain) == ()


@pytest.mark.parametrize("bad", ["six", [""], [3]])
def test_a_malformed_lineage_is_refused(tmp_path, tiny_net, optim_scaler_sched, valid_config, metadata_kwargs, bad):
    with pytest.raises(CheckpointStampError, match="lineage"):
        _save(tmp_path, tiny_net, optim_scaler_sched, valid_config, metadata_kwargs, lineage=bad)


def test_the_strip_and_a_resume_keep_the_lineage(tmp_path, tiny_net, optim_scaler_sched, valid_config, metadata_kwargs):
    from mantis.train.trainer.core import Trainer

    src = _save(tmp_path, tiny_net, optim_scaler_sched, valid_config, metadata_kwargs, lineage=[TAG])
    stripped = strip_and_restamp(src, new_encoding="gnn_axis_v1", run_id="runc", checkpoint_dir=tmp_path / "strip")
    assert load_checkpoint(stripped, declared_encoding=None).metadata.lineage == (TAG,)
    trainer = resume_trainer(Trainer, src, fallback_config=valid_config, declared_encoding=None)
    assert trainer.lineage == (TAG,)
    trainer.step += 1
    assert load_checkpoint(trainer.save_checkpoint(), declared_encoding=None).metadata.lineage == (TAG,)


def test_a_warm_start_inherits_its_source_lineage_and_a_plain_launch_has_none(tmp_path, valid_config, metadata_kwargs):
    from mantis.encoding import lookup
    from mantis.model import arch_from_spec_and_config, build_net
    from mantis.train.orchestrator import init_trainer

    arch = arch_from_spec_and_config(lookup(valid_config["identity"]["encoding"]), valid_config)
    source = build_net(arch)
    src = save_checkpoint(model=source, optimizer=None, scaler=None, scheduler=None, step=0, config=valid_config,
                          metadata_kwargs={**metadata_kwargs, "arch": arch, "lineage": [TAG]},
                          checkpoint_dir=tmp_path / "src", kind="weights")
    warm = {**valid_config, "identity": {**valid_config["identity"], "warm_start": {
        "checkpoint": str(src), "net_hash": net_param_hash(source), "reinit": []}}}
    trainer = init_trainer(config=warm, device=torch.device("cpu"), checkpoint_dir=str(tmp_path / "warm"))
    assert trainer.lineage == (TAG,)
    assert load_checkpoint(trainer.save_checkpoint(), declared_encoding=None).metadata.lineage == (TAG,)
    plain = init_trainer(config=valid_config, device=torch.device("cpu"), checkpoint_dir=str(tmp_path / "plain"))
    assert plain.lineage == ()
