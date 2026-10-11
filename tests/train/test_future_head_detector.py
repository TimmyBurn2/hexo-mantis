"""A head-bearing net is never served or trained by a path that expects none, nor a head-less one by a path that expects it."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import _microbatch_harness as H
from mantis.model import GnnNetV2SoftPolicy, GnnNetV2SoftPolicyFuture
from mantis.train.checkpoints import DeclaredFutureHeadMismatchError, future_head_declared, load_checkpoint

_REPO = Path(__file__).resolve().parents[2]


def _stamps(tmp_path: Path) -> dict[bool, Path]:
    """A saved future-kind net and a saved soft-policy net, keyed by whether the stamp carries the head."""
    return {True: H.future_graph_trainer(tmp_path / "future", sink=H.SpySink()).save_checkpoint(None),
            False: H.soft_policy_graph_trainer(tmp_path / "soft", sink=H.SpySink()).save_checkpoint(None)}


def _bench_server() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_server_tool", _REPO / "tools" / "bench_server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("carries", [True, False])
def test_the_loader_refuses_the_other_declaration_and_adopts_on_none(tmp_path: Path, carries: bool) -> None:
    """PLANTED BREAK for a detector that checks one direction only: both refusals are asserted, each beside its control."""
    path = _stamps(tmp_path)[carries]
    assert load_checkpoint(path, declared_encoding=None, declared_future_head=carries).metadata.arch is not None
    assert load_checkpoint(path, declared_encoding=None, declared_future_head=None).metadata.arch is not None
    with pytest.raises(DeclaredFutureHeadMismatchError, match="expects no" if carries else "expects future"):
        load_checkpoint(path, declared_encoding=None, declared_future_head=not carries)


def test_a_config_declares_the_head_its_arch_kind_carries() -> None:
    assert future_head_declared({"identity": {"arch_kind": "GnnArchV2SoftPolicyFuture"}})
    assert not future_head_declared({"identity": {"arch_kind": "GnnArchV2SoftPolicy"}})
    assert not future_head_declared({"identity": {"encoding": "gnn_axis_v1"}}), "a config predating the row has none"


def test_a_config_bearing_serving_path_serves_only_the_kind_its_config_names(tmp_path: Path) -> None:
    """bench_server builds the served net from a stamp under a config: the matching kind serves, the other refuses."""
    stamps, bench = _stamps(tmp_path), _bench_server()
    future = H.graph_config()
    future["identity"]["arch_kind"] = "GnnArchV2SoftPolicyFuture"
    soft = H.graph_config()
    soft["identity"]["arch_kind"] = "GnnArchV2SoftPolicy"
    assert type(bench._load_net(future, stamps[True])[0]) is GnnNetV2SoftPolicyFuture
    assert type(bench._load_net(soft, stamps[False])[0]) is GnnNetV2SoftPolicy
    with pytest.raises(DeclaredFutureHeadMismatchError):
        bench._load_net(soft, stamps[True])
    with pytest.raises(DeclaredFutureHeadMismatchError):
        bench._load_net(future, stamps[False])
