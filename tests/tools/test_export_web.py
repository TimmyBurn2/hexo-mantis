"""`tools/export_web.py` (contract #13, gate 20): tiny random-init stamps of every carried kind through the whole export, written by the one checkpoint writer into temp directories."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from mantis import _engine
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.tactics import resolve_deploy_tactics
from mantis.encoding import lookup
from mantis.model import ARCH_KINDS, build_net, gnn_widths_block
from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import deploy_state, load_checkpoint, save_checkpoint
from mantis.util.git import head_sha
from _toolpath import load_module_by_path

REPO = Path(__file__).resolve().parents[2]
export_web = load_module_by_path("export_web", REPO / "tools" / "export_web.py")

pytestmark = pytest.mark.xdist_group("export_web")

POSITIONS = 16
CASES = [(kind, "gnn_axis_r8") for kind in export_web.CARRIED_KINDS] + [("GnnArchV2SoftPolicyFuture", "gnn_axis_r8_pruned")]
FUTURE = ("GnnArchV2SoftPolicyFuture", "gnn_axis_r8")


def _base_config(kind: str) -> dict[str, Any]:
    """A production config of the census that serves a carried row, of `kind` where one exists."""
    dumps = [load_config(path).model_dump() for path in production_configs(REPO)]
    graph = [d for d in dumps if d["identity"]["encoding"] in export_web.CARRIED_ENCODINGS]
    return copy.deepcopy(next((d for d in graph if d["identity"]["arch_kind"] == kind), graph[0]))


def _tiny_stamp(out: Path, kind: str, encoding: str, **config_changes: Any) -> Path:
    """A random-init stamp of `kind` under `encoding`, its config a census config with the stamp's identity and widths."""
    torch.manual_seed(0)
    spec = lookup(encoding)
    assert spec.node_feat_dim is not None and spec.edge_feat_dim is not None
    arch = ARCH_KINDS[kind](in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=16, num_layers=2,
                            policy_hidden=8, value_hidden=8)
    config = _base_config(kind)
    config["identity"].update(encoding=encoding, arch_kind=kind)
    config["model"]["gnn"] = gnn_widths_block(arch)
    # A head a kind does not carry is a dead loss row in its config (LAW-08), so the stamp drops it.
    if "SoftPolicy" not in kind:
        config["model"]["aux_soft_policy"] = None
    if not kind.endswith("Future"):
        config["model"]["aux_future_occupancy"] = None
    for section, value in config_changes.items():
        config[section].update(value)
    return save_checkpoint(model=build_net(arch), optimizer=None, scaler=None, scheduler=None, step=0, config=config,
                           kind="weights", metadata_kwargs={"encoding_name": encoding, "run_id": "tiny", "arch": arch},
                           checkpoint_dir=out)


@pytest.fixture(scope="module")
def exports(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], tuple[Path, Path, dict]]:
    """Each carried (kind, encoding): its stamp, its out directory and its manifest."""
    done = {}
    for kind, encoding in CASES:
        root = tmp_path_factory.mktemp(f"{kind}-{encoding}")
        stamp = _tiny_stamp(root / "stamp", kind, encoding)
        done[(kind, encoding)] = (stamp, root / "out", export_web.export(stamp, root / "out", n_positions=POSITIONS))
    return done


@pytest.fixture(scope="module")
def future_cases(exports) -> dict[int, list]:
    """The witness's B 8 cases for the Future-kind stamp."""
    stamp = exports[FUTURE][0]
    ck = load_checkpoint(stamp, declared_encoding=None, declared_future_head=None)
    assert ck.metadata.arch is not None and isinstance(ck.config, dict)
    net = build_net(ck.metadata.arch)
    net.load_state_dict(deploy_state(ck)[0])
    net.eval()
    return export_web.witness_cases(net, lookup(ck.metadata.encoding_name), ck.config, ck.metadata.encoding_name,
                                    export_web.golden_positions(POSITIONS), (8,))


def test_every_carried_kind_and_row_exports_and_holds_the_fp32_lines_at_one_and_default_threads(exports) -> None:
    """The export re-expresses bit for bit (it refuses otherwise) and ORT CPU holds the fp32 lines on every batch."""
    for (kind, encoding), (_stamp, _out, manifest) in exports.items():
        rows = [r for r in manifest["witness"]["rows"] if r["line"] == "fp32"]
        assert {(r["threads"], r["B"]) for r in rows} == {(t, b) for t in (1, "default") for b in (1, 8, 16, 5)}
        assert all(r["pass"] and r["argmax_compared"] > 0 for r in rows), (kind, encoding, rows)
        assert manifest["stamp"]["arch_kind"] == kind and manifest["encoding"]["name"] == encoding
        assert manifest["files"][0]["passes_its_line"] is True


def test_no_written_graph_holds_a_scatternd_a_gathernd_or_an_int64_batch_index(exports) -> None:
    """The batch is int32 so the graph reads it as int32: every batch-sized index a Gather or a scatter reads is int32."""
    for stamp, out, manifest in exports.values():
        for row in manifest["files"][:2]:
            assert row["ops"]["ScatterElements"] > 0, row["name"]
            assert export_web.graph_refusal(row["ops"], onnx.load(str(out / row["name"]))) is None, row["name"]
        assert stamp.stem in manifest["files"][0]["name"]


def test_an_unnarrowed_graph_is_refused_and_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PLANTED BREAK: skip the int32 narrowing and torch's own int64 casts must refuse the graph."""
    stamp = _tiny_stamp(tmp_path / "stamp", "GnnArchV2", "gnn_axis_r8")
    monkeypatch.setattr(export_web, "narrow_index_casts", lambda _path: None)
    with pytest.raises(export_web.ExportRefused) as refused:
        export_web.export(stamp, tmp_path / "out", n_positions=POSITIONS)
    assert refused.value.field == "graph" and not list((tmp_path / "out").glob("*.onnx"))


def test_a_widened_index_is_refused_and_the_narrowing_takes_it_through_a_reshape(tmp_path: Path) -> None:
    """index_select's own lowering: Reshape(int32) -> Cast(int64) -> Gather; refused as written, int32 once narrowed."""
    nodes = [helper.make_node("Reshape", ["idx", "flat"], ["idx_flat"]),
             helper.make_node("Cast", ["idx_flat"], ["idx64"], to=TensorProto.INT64),
             helper.make_node("Gather", ["data", "idx64"], ["out"])]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N", 4]),
                                           helper.make_tensor_value_info("idx", TensorProto.INT32, ["E"])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["E", 4])],
                              [numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    assert "ScatterND" in (export_web.graph_refusal({"ScatterND": 1}, model) or "")
    assert "GatherND" in (export_web.graph_refusal({"GatherND": 1}, model) or "")
    assert "idx64" in (export_web.graph_refusal({"Gather": 1}, model) or ""), "the widened index must be named"
    path = tmp_path / "g.onnx"
    onnx.save(model, str(path))
    export_web.narrow_index_casts(path)
    narrowed = onnx.load(str(path))
    assert export_web.graph_refusal({"Gather": 1}, narrowed) is None
    assert [n.op_type for n in narrowed.graph.node] == ["Reshape", "Gather"]


def test_the_argmax_rule_reds_a_flip_the_value_lines_cannot_see() -> None:
    """PLANTED BREAK: a top-2 swap beyond the gap moves no prob or value past fp16's line, and the argmax rule alone reds it."""
    ref_logits = np.array([0.0, 0.03, -9.0, -9.0], dtype=np.float32)
    ref = {"logits": ref_logits, "probs": np.zeros(4, np.float32), "value": np.zeros(1, np.float32)}
    out = {"logits": ref_logits[[1, 0, 2, 3]], "probs": ref["probs"], "value": ref["value"]}
    row = export_web.gaps(out, ref, np.zeros(4, np.int32), export_web.LINES["fp16"])
    assert (row["argmax_agreed"], row["argmax_compared"]) == (0, 1)
    assert not export_web.passes(row, export_web.LINES["fp16"])
    assert export_web.passes({**row, "argmax_agreed": 1}, export_web.LINES["fp16"])


def test_the_manifest_binds_the_stamp_the_files_and_the_resolved_block(exports) -> None:
    stamp, out, manifest = exports[FUTURE]
    ck = load_checkpoint(stamp, declared_encoding=None, declared_future_head=None)
    assert ck.metadata.arch is not None and isinstance(ck.config, dict)
    net = build_net(ck.metadata.arch)
    net.load_state_dict(deploy_state(ck)[0])
    assert manifest["format"] == export_web.FORMAT
    assert manifest["stamp"]["sha256"] == hashlib.sha256(stamp.read_bytes()).hexdigest()
    assert manifest["stamp"]["lineage"] == list(ck.metadata.lineage)
    assert manifest["stamp"]["net_param_hash"] == net_param_hash(net)
    assert manifest["exporter"]["commit"] == head_sha(REPO)
    assert manifest["encoding"]["registry_sha"] == _engine.registry_sha_hex()
    assert manifest["deploy"]["resolved"]["tactics"] == resolve_deploy_tactics(ck.config)
    for row in manifest["files"]:
        path = out / row["name"]
        assert row["name"].startswith(stamp.stem), "every file is named by the stamp's stem"
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() and row["bytes"] == path.stat().st_size
    weights = next(r for r in manifest["files"] if r["name"].endswith(".weights.bin"))
    assert sum(t["rows"] * t["cols"] for t in weights["tensors"]) * 4 == weights["bytes"]
    device = json.loads((out / f"{stamp.stem}.witness.json").read_text(encoding="utf-8"))
    assert len(device["positions"]) == len(export_web.OPENINGS) + export_web.DEVICE_WITNESS_GOLDENS
    assert len(device["probs"]) == len(device["positions"]) == len(device["value"])


def test_the_fp16_file_stores_only_what_fp16_holds_finite(exports) -> None:
    """A constant like finfo(f32).min would turn infinite in fp16, so it stays fp32: no stored value changes finiteness."""
    stamp, out, _manifest = exports[FUTURE]
    source = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(out / f"{stamp.stem}.fp32.onnx")).graph.initializer
              if i.data_type == TensorProto.FLOAT}
    halves = {i.name.removesuffix("__fp16"): numpy_helper.to_array(i)
              for i in onnx.load(str(out / f"{stamp.stem}.fp16.onnx")).graph.initializer if i.data_type == TensorProto.FLOAT16}
    with np.errstate(over="ignore"):
        kept = [n for n, a in source.items() if np.isfinite(a).all() and not np.isfinite(a.astype(np.float16)).all()]
    assert kept, "the guard has nothing to keep: the fp32 graph should hold finfo(f32).min"
    assert not any(n in halves for n in kept), "a value fp16 cannot hold finite was stored in fp16"
    assert halves and all(np.array_equal(np.isfinite(h), np.isfinite(source[n])) for n, h in halves.items())


def _broken(src: Path, dst: Path, name: str, change: Callable[[np.ndarray], np.ndarray]) -> Path:
    model = onnx.load(str(src))
    init = next(i for i in model.graph.initializer if i.name == name)
    init.CopyFrom(numpy_helper.from_array(change(numpy_helper.to_array(init)), name))
    onnx.save(model, str(dst))
    return dst


def _first_bin(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[0] += np.float32(4.0)
    return out


@pytest.mark.parametrize(("name", "change", "metric"), [
    ("policy.2.bias", lambda a: a + np.float32(1.0), "logits"),
    ("policy.2.weight", lambda a: a * np.float32(50.0), "probs"),
    ("fc2.bias", _first_bin, "value"),
])
def test_a_perturbed_written_graph_misses_the_line_it_moves(exports, future_cases, tmp_path: Path, name: str,
                                                            change: Callable[[np.ndarray], np.ndarray],
                                                            metric: str) -> None:
    """PLANTED BREAKS: a bias shift moves only the logits (softmax ignores it), a weight scale the probs, a bin the value."""
    stamp, out, _manifest = exports[FUTURE]
    broken = _broken(out / f"{stamp.stem}.fp32.onnx", tmp_path / "broken.onnx", name, change)
    for dtype in ("fp32", "fp16"):
        line = export_web.LINES[dtype]
        bound = {"logits": line.logit, "probs": line.prob, "value": line.value}[metric]
        rows = export_web.witness(broken, 1, future_cases, dtype)
        if bound is None:
            assert all(r["logits"] > 0.5 and r["pass"] for r in rows), rows  # reported, not gated, in fp16's line
            continue
        assert all(not r["pass"] and r[metric] > bound for r in rows), (dtype, rows)
    assert all(r["pass"] for r in export_web.witness(out / f"{stamp.stem}.fp32.onnx", 1, future_cases, "fp32"))


def test_a_re_expression_that_is_not_bit_identical_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PLANTED BREAK: one ulp of drift in the traced forward refuses the export before anything is written."""
    stamp = _tiny_stamp(tmp_path / "stamp", "GnnArchV2", "gnn_axis_r8")
    forward = export_web.WebServedGraph.forward

    def drifted(self, *args):  # noqa: ANN001, ANN202 — the traced signature is the original's
        logits, probs, value = forward(self, *args)
        return logits, probs, torch.nextafter(value, torch.full_like(value, 2.0))

    monkeypatch.setattr(export_web.WebServedGraph, "forward", drifted)
    with pytest.raises(export_web.ExportRefused) as refused:
        export_web.export(stamp, tmp_path / "out", n_positions=POSITIONS)
    assert refused.value.field == "re-expression"
    assert not (tmp_path / "out").exists()


def test_a_foreign_batch_and_an_unchecked_registry_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PLANTED BREAKS: an engine batch whose x is not the served one, and a registry the import handshake never checked."""
    stamp = _tiny_stamp(tmp_path / "stamp", "GnnArchV2", "gnn_axis_r8")
    inputs = export_web.web_inputs

    def foreign(*args: Any) -> dict[str, np.ndarray]:
        arrays = inputs(*args)
        arrays["x"] = arrays["x"] + np.float32(1.0)
        return arrays

    with monkeypatch.context() as m:
        m.setattr(export_web, "web_inputs", foreign)
        with pytest.raises(export_web.ExportRefused) as refused:
            export_web.export(stamp, tmp_path / "out", n_positions=POSITIONS)
        assert refused.value.field == "batch"
    with monkeypatch.context() as m:
        m.setattr(export_web, "handshake_ran", lambda: False)
        with pytest.raises(export_web.ExportRefused) as refused:
            export_web.export(stamp, tmp_path / "out", n_positions=POSITIONS)
        assert refused.value.field == "registry"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("kind", "encoding", "search", "field"), [
    ("GnnArch", "gnn_axis_r8", "puct", "arch_kind"),
    ("GnnArchV9", "gnn_axis_r8", "puct", "arch_kind"),
    ("GnnArchV2", "hexcanvas_r8", "puct", "encoding"),
    ("GnnArchV2", "gnn_axis_v1", "puct", "encoding"),
    ("GnnArchV2", "gnn_axis_r8", "gumbel", "deploy.search.kind"),
])
def test_the_detector_refuses_what_it_does_not_carry_by_name(kind: str, encoding: str, search: str, field: str) -> None:
    with pytest.raises(export_web.ExportRefused) as refused:
        export_web.detect(kind, encoding, search)
    assert refused.value.field == field


def test_a_gumbel_deploy_block_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    stamp = _tiny_stamp(tmp_path / "stamp", "GnnArchV2", "gnn_axis_r8",
                        deploy={"search": {"kind": "gumbel", "tactics": None}})
    with pytest.raises(export_web.ExportRefused) as refused:
        export_web.export(stamp, tmp_path / "out", n_positions=POSITIONS)
    assert refused.value.field == "deploy.search.kind"
    assert not (tmp_path / "out").exists()
