"""R330(e) — a config-less call site resolves an ARTIFACT's arch from its STAMP, never from a
config table, and there is ONE function that answers: `stamped_arch_kind`.

THE DEFECT THIS CLOSES, and the planted break that shows the row bites. Before R330(e) the three
call sites that hold no run config — `load_legacy_weights` (a v1 envelope's embedded config),
`strip_and_restamp` (passes `{}`) and `pretrain.validate` (passes `{}`) — all called
`arch_from_spec_and_config`, i.e. resolved the INCUMBENT for the representation. For a V2-stamped
source, `strip_and_restamp` therefore rebuilt `GnnArch` and re-stamped the stripped artifact as
V1 — V2's weights under V1's provenance, the class LAW-12 exists for. The strip test below reds
under that code (`GnnArchV2` is a SIBLING of `GnnArch`, so the `type(...) is` check cannot be
satisfied by the old resolution) and is green under the stamp read.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import CnnArch, GnnArch, GnnArchV2, RepresentationMismatch, build_net, select_arch
from mantis.train.checkpoints import (
    CheckpointStampError,
    load_checkpoint,
    load_legacy_weights,
    save_checkpoint,
    stamped_arch_kind,
    strip_and_restamp,
)
from mantis.train.pretrain.validate import validate

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
_TINY_GRAPH = dict(hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)


def _graph_config_dump() -> dict:
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_config(path)
        if cfg.identity.representation == "graph":
            return cfg.model_dump()
    raise AssertionError("no shipped graph config")


def _v2_net(spec):
    arch = GnnArchV2(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), **_TINY_GRAPH)
    net = build_net(arch)
    return net, arch


# ── the function ─────────────────────────────────────────────────────────────────────────
def test_a_v2_stamp_names_its_kind_and_a_pre_discriminator_stamp_is_the_incumbent_era():
    assert stamped_arch_kind({"arch": {"arch_kind": "GnnArchV2"}}, representation="graph") == "GnnArchV2"
    assert stamped_arch_kind({"arch": {"arch_kind": "GnnArch"}}, representation="graph") == "GnnArch"
    # a v2 stamp from before B1's discriminator, a v1 envelope, a bare anchor: no arch_kind
    assert stamped_arch_kind({"arch": {"representation": "graph"}}, representation="graph") == "GnnArch"
    assert stamped_arch_kind({"encoding_name": "v6_live2_ls"}, representation="grid") == "CnnArch"
    assert stamped_arch_kind(None, representation="grid") == "CnnArch"


def test_an_unknown_stamped_kind_or_representation_is_refused_not_nearest_fitted():
    with pytest.raises(RepresentationMismatch, match="does not know"):
        stamped_arch_kind({"arch": {"arch_kind": "GnnArchV9"}}, representation="graph")
    with pytest.raises(RepresentationMismatch, match="expected 'grid' or 'graph'"):
        stamped_arch_kind({}, representation="dense")


# ── strip_and_restamp: the source's kind crosses the strip ───────────────────────────────
def test_strip_and_restamp_keeps_a_V2_source_V2(tmp_path):
    """THE PLANTED BREAK. Resolve the arch from the incumbent table instead of the stamp and the
    stripped artifact is stamped `GnnArch`: this row reds."""
    spec = lookup("gnn_axis_v1")
    net, arch = _v2_net(spec)
    src = save_checkpoint(
        model=net, optimizer=None, scaler=None, scheduler=None, step=7,
        config=_graph_config_dump(),
        metadata_kwargs={"encoding_name": "gnn_axis_v1", "run_id": "r330e", "arch": arch},
        checkpoint_dir=tmp_path, kind="weights",
    )
    out = strip_and_restamp(src, new_encoding="gnn_axis_v1", run_id="r330e-strip",
                            checkpoint_dir=tmp_path)
    ck = load_checkpoint(out)
    assert type(ck.metadata.arch) is GnnArchV2, type(ck.metadata.arch).__name__
    assert not isinstance(ck.metadata.arch, GnnArch)
    # kind AND widths: the stamp is the arch, not a table's default-width rebuild of its kind
    assert ck.metadata.arch == arch
    # and the stripped weights rebuild through the stamped arch, not through a table
    rebuilt = build_net(ck.metadata.arch)
    rebuilt.load_state_dict(ck.model_state)


# ── load_legacy_weights: a v1 envelope resolves to the incumbent-era kind, contradictions refuse ─
def _legacy_envelope(tmp_path: Path, config: dict) -> Path:
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), **_TINY_GRAPH)
    net = build_net(arch)
    path = tmp_path / "legacy_full_v1.pt"
    torch.save({
        "step": 3, "model_state": net.state_dict(), "optimizer_state": {}, "scaler_state": {},
        "metadata": {"encoding_name": "gnn_axis_v1"}, "config": config,
    }, path)
    return path


def test_a_legacy_envelope_resolves_to_the_incumbent_era_kind_from_its_stamp(tmp_path):
    ck = load_legacy_weights(_legacy_envelope(tmp_path, _graph_config_dump()))
    assert type(ck.metadata.arch) is GnnArch


def test_a_legacy_envelope_whose_embedded_row_contradicts_its_stamp_is_refused(tmp_path):
    """Two records of one artifact disagree; neither wins. The embedded config says V2, the stamp
    is pre-discriminator and therefore V1 — a config-led resolution would rebuild V2 over V1
    weights, and a stamp-led one would silently drop the row. Both are wrong; refuse."""
    config = _graph_config_dump()
    config["identity"]["arch_kind"] = "GnnArchV2"
    with pytest.raises(CheckpointStampError, match="identity.arch_kind='GnnArchV2'"):
        load_legacy_weights(_legacy_envelope(tmp_path, config))


# ── pretrain.validate: the third config-less site ─────────────────────────────────────────
def _pretrain_payload(tmp_path: Path, metadata: dict) -> Path:
    spec = lookup("v6_live2_ls")
    cfg = {"encoding": "v6_live2_ls", "filters": 8, "res_blocks": 1}
    arch = select_arch(spec, cfg, arch_kind="CnnArch")
    assert isinstance(arch, CnnArch)
    net = build_net(arch)
    path = tmp_path / "pretrain_00000010.pt"
    torch.save({"step": -10, "model_state": net.state_dict(), "config": cfg, "metadata": metadata}, path)
    return path


def test_the_pretrain_validator_reads_the_artifacts_stamp(tmp_path):
    validate(_pretrain_payload(tmp_path, {"encoding_name": "v6_live2_ls"}), torch.device("cpu"))
    with pytest.raises(RepresentationMismatch, match="does not know"):
        validate(_pretrain_payload(tmp_path, {"encoding_name": "v6_live2_ls",
                                              "arch": {"arch_kind": "Bogus"}}), torch.device("cpu"))
