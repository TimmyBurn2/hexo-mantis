"""A net stamped for one encoding never serves or trains under another, shapes agreeing, each refusal beside its control; the strip is the one way across."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

import _microbatch_harness as H
from mantis.encoding import lookup
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import (
    DeclaredEncodingMismatchError,
    load_checkpoint,
    load_legacy_weights,
    strip_and_restamp,
)
from mantis.train.orchestrator import init_trainer
from mantis.train.trainer.core import Trainer
from mantis.train.warmstart import BcWarmStart, apply_bc_warm_start

R8, PRUNED = "gnn_axis_r8", "gnn_axis_r8_pruned"
_REPO = Path(__file__).resolve().parents[2]
_CPU = torch.device("cpu")


def _r8_checkpoint(tmp_path: Path) -> Path:
    """A full checkpoint the tiny trainer saves under a config naming `gnn_axis_r8`."""
    torch.manual_seed(H.SEED)
    arch = H.tiny_graph_arch()
    config = H.graph_config()
    config["identity"]["encoding"] = R8
    trainer = Trainer(build_net(arch), config, arch=arch, checkpoint_dir=tmp_path / "ckpt", device=_CPU,
                      train_hparams=H.graph_hparams(), sink=H.SpySink())
    return trainer.save_checkpoint(None)


def test_a_stamp_is_read_under_its_own_encoding_and_refused_under_the_other(tmp_path: Path) -> None:
    path = _r8_checkpoint(tmp_path)
    assert load_checkpoint(path, declared_encoding=R8).metadata.encoding_name == R8
    with pytest.raises(DeclaredEncodingMismatchError):
        load_checkpoint(path, declared_encoding=PRUNED)


def test_a_resume_under_another_encoding_is_refused(tmp_path: Path) -> None:
    """The launch config builds the buffer, pool and evals, so its encoding must be the stamp's."""
    path = _r8_checkpoint(tmp_path)
    launch = load_checkpoint(path, declared_encoding=R8).config
    assert init_trainer(config=launch, device=_CPU, checkpoint_path=str(path)).config["identity"]["encoding"] == R8
    launch["identity"]["encoding"] = PRUNED
    with pytest.raises(DeclaredEncodingMismatchError):
        init_trainer(config=launch, device=_CPU, checkpoint_path=str(path))


def test_a_warm_start_across_encodings_is_refused(tmp_path: Path) -> None:
    path = _r8_checkpoint(tmp_path)
    ck = load_checkpoint(path, declared_encoding=R8)
    assert ck.metadata.arch is not None
    parent = build_net(ck.metadata.arch)
    parent.load_state_dict(ck.model_state)
    declared = BcWarmStart(checkpoint=path, net_hash=net_param_hash(parent), reinit=())
    apply_bc_warm_start(build_net(ck.metadata.arch), declared, spec=lookup(R8))
    with pytest.raises(DeclaredEncodingMismatchError):
        apply_bc_warm_start(build_net(ck.metadata.arch), declared, spec=lookup(PRUNED))


def test_a_legacy_stamp_disagreeing_with_the_declared_encoding_is_refused(tmp_path: Path) -> None:
    raw = torch.load(_r8_checkpoint(tmp_path), weights_only=True, map_location="cpu")
    legacy = tmp_path / "legacy.pt"
    torch.save({"model_state": raw["model_state"], "metadata": raw["metadata"]}, legacy)
    assert load_legacy_weights(legacy, declared_encoding=R8).metadata.encoding_name == R8
    with pytest.raises(DeclaredEncodingMismatchError):
        load_legacy_weights(legacy, declared_encoding=PRUNED)


def test_the_weights_only_strip_is_the_one_way_across(tmp_path: Path) -> None:
    path = _r8_checkpoint(tmp_path)
    stripped = strip_and_restamp(path, new_encoding=PRUNED, run_id="crossing", checkpoint_dir=tmp_path / "strip")
    ck = load_checkpoint(stripped, declared_encoding=PRUNED)
    assert ck.metadata.encoding_name == PRUNED
    source = load_checkpoint(path, declared_encoding=R8).model_state
    assert all(torch.equal(ck.model_state[k], source[k]) for k in source)
    with pytest.raises(DeclaredEncodingMismatchError):
        load_checkpoint(stripped, declared_encoding=R8)


def test_a_frontier_cell_refuses_a_net_stamped_for_another_encoding(tmp_path: Path) -> None:
    """The rung and S cells snapshot the net they play; the snapshot is where the stamp meets the cell's config."""
    spec = importlib.util.spec_from_file_location("strength_frontier_crossing_t", _REPO / "tools/strength_frontier.py")
    assert spec is not None and spec.loader is not None
    frontier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontier)
    path = _r8_checkpoint(tmp_path)
    assert frontier._snapshot_from_checkpoint(path, tmp_path / "r8.snap", encoding=R8)["step"] == 0
    with pytest.raises(DeclaredEncodingMismatchError):
        frontier._snapshot_from_checkpoint(path, tmp_path / "pruned.snap", encoding=PRUNED)
