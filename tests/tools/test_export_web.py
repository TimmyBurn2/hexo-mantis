"""`tools/export_web.py` (contract #13, gate 20): tiny random-init stamps of every carried kind through the whole export.

Each stamp is written by the one checkpoint writer into a temp directory, so no weights live in the tree. The real
stamp's witness is a release step recorded in its manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper

from mantis import _engine
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import ARCH_KINDS, build_net, gnn_widths_block
from mantis.train.checkpoints import save_checkpoint
from _toolpath import load_module_by_path

REPO = Path(__file__).resolve().parents[2]
export_web = load_module_by_path("export_web", REPO / "tools" / "export_web.py")

pytestmark = pytest.mark.xdist_group("export_web")

POSITIONS = 16
CASES = [(kind, "gnn_axis_r8") for kind in export_web.CARRIED_KINDS] + [("GnnArchV2SoftPolicyFuture", "gnn_axis_r8_pruned")]


def _tiny_stamp(out: Path, kind: str, encoding: str, **config_changes: Any) -> Path:
    """A random-init stamp of `kind` under `encoding`, its config run14's with the stamp's identity and widths."""
    torch.manual_seed(0)
    spec = lookup(encoding)
    assert spec.node_feat_dim is not None and spec.edge_feat_dim is not None
    arch = ARCH_KINDS[kind](in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=16,
                            num_layers=2, policy_hidden=8, value_hidden=8)
    config = load_config(REPO / "configs" / "run14.yaml").model_dump()
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


def test_every_carried_kind_and_row_exports_and_holds_the_fp32_lines_at_one_and_default_threads(exports) -> None:
    """The export re-expresses bit for bit (it refuses otherwise) and ORT CPU holds the fp32 lines on every batch."""
    for (kind, encoding), (_stamp, _out, manifest) in exports.items():
        rows = [r for r in manifest["witness"]["rows"] if r["line"] == "fp32"]
        assert {(r["threads"], r["B"]) for r in rows} == {(t, b) for t in (1, "default") for b in (1, 8, 16, 5)}
        assert all(r["pass"] for r in rows), (kind, encoding, [r for r in rows if not r["pass"]])
        assert manifest["stamp"]["arch_kind"] == kind and manifest["encoding"]["name"] == encoding


def test_no_written_graph_holds_a_scatternd(exports) -> None:
    """ScatterND(add) races under threaded ORT kernels; every scatter-sum is ScatterElements."""
    for _stamp, _out, manifest in exports.values():
        for row in manifest["files"]:
            if "ops" in row:
                assert "ScatterND" not in row["ops"] and row["ops"]["ScatterElements"] > 0, row["name"]


def test_the_manifest_binds_the_stamp_the_files_and_the_resolved_block(exports) -> None:
    stamp, out, manifest = exports[("GnnArchV2SoftPolicyFuture", "gnn_axis_r8")]
    assert manifest["format"] == export_web.FORMAT
    assert manifest["stamp"]["sha256"] == hashlib.sha256(stamp.read_bytes()).hexdigest()
    assert manifest["encoding"]["registry_sha"] == _engine.registry_sha_hex()
    assert manifest["deploy"]["resolved"]["tactics"]["audit"]["mode"] == "hold"
    for row in manifest["files"]:
        path = out / row["name"]
        assert row["name"].startswith(stamp.stem), "every file is named by the stamp's stem"
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() and row["bytes"] == path.stat().st_size
    weights = next(r for r in manifest["files"] if r["name"].endswith(".weights.bin"))
    assert sum(t["rows"] * t["cols"] for t in weights["tensors"]) * 4 == weights["bytes"]
    device = json.loads((out / f"{stamp.stem}.witness.json").read_text(encoding="utf-8"))
    assert len(device["positions"]) == len(export_web.OPENINGS) + export_web.DEVICE_WITNESS_GOLDENS
    assert len(device["probs"]) == len(device["positions"]) == len(device["value"])


def _cases(stamp: Path) -> tuple[Any, dict[int, list[tuple[dict, dict]]]]:
    """The witness's B 8 inputs and served references for `stamp`, rebuilt from the module's own pieces."""
    from mantis.train.checkpoints import deploy_state, load_checkpoint

    ck = load_checkpoint(stamp, declared_encoding=None, declared_future_head=None)
    spec = lookup(ck.metadata.encoding_name)
    assert ck.metadata.arch is not None and spec.node_feat_dim is not None
    net = build_net(ck.metadata.arch)
    net.load_state_dict(deploy_state(ck)[0])
    net.eval()
    positions = export_web.golden_positions(POSITIONS)
    with torch.inference_mode():
        refs = export_web.served_reference(net, spec, ck.config, positions, 8)
    inputs = [export_web.web_inputs(positions[i:i + 8], ck.metadata.encoding_name, spec.node_feat_dim)
              for i in range(0, POSITIONS, 8)]
    return net, {8: list(zip(inputs, refs, strict=True))}


def test_a_perturbed_written_graph_misses_the_fp32_line(exports, tmp_path: Path) -> None:
    """PLANTED BREAK: the written ONNX's last policy bias +1.0 moves every logit by 1, so the witness must red it."""
    stamp, out, _manifest = exports[("GnnArchV2SoftPolicyFuture", "gnn_axis_r8")]
    model = onnx.load(str(out / f"{stamp.stem}.fp32.onnx"))
    bias = next(i for i in model.graph.initializer if i.name.startswith("policy.2.") and i.name.endswith("bias"))
    bias.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(bias) + np.float32(1.0), bias.name))
    broken = tmp_path / "broken.onnx"
    onnx.save(model, str(broken))
    _net, cases = _cases(stamp)
    rows = export_web.witness(broken, 1, cases, "fp32")
    assert rows and not any(r["pass"] for r in rows) and all(r["logits"] > 0.5 for r in rows), rows
    assert all(r["pass"] for r in export_web.witness(out / f"{stamp.stem}.fp32.onnx", 1, cases, "fp32"))


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
