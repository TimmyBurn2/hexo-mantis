"""Tiny random-init stamps for the web package's witnesses: gate 20 exports them in its tests, and gate 21 runs this file to export one for the node tests: `python tests/tools/_web_stamps.py OUT_DIR`."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

from _toolpath import load_module_by_path
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import ARCH_KINDS, build_net, gnn_widths_block
from mantis.train.checkpoints import save_checkpoint

REPO = Path(__file__).resolve().parents[2]
export_web = load_module_by_path("export_web", REPO / "tools" / "export_web.py")


def base_config(kind: str) -> dict[str, Any]:
    """A production config of the census that serves a carried row, of `kind` where one exists."""
    dumps = [load_config(path).model_dump() for path in production_configs(REPO)]
    graph = [d for d in dumps if d["identity"]["encoding"] in export_web.CARRIED_ENCODINGS]
    return copy.deepcopy(next((d for d in graph if d["identity"]["arch_kind"] == kind), graph[0]))


def tiny_stamp(out: Path, kind: str, encoding: str, *, layers: int = 2, **config_changes: Any) -> Path:
    """A random-init stamp of `kind` under `encoding`, its config a census config with the stamp's identity and widths."""
    torch.manual_seed(0)
    spec = lookup(encoding)
    assert spec.node_feat_dim is not None and spec.edge_feat_dim is not None
    arch = ARCH_KINDS[kind](in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=16, num_layers=layers,
                            policy_hidden=8, value_hidden=8)
    config = base_config(kind)
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


def main() -> int:
    """Export the Future-kind tiny stamp on `gnn_axis_r8` into `OUT/net/`, beside the `index.json` the engine reads."""
    out = Path(sys.argv[1])
    stamp = tiny_stamp(out / "stamp", "GnnArchV2SoftPolicyFuture", "gnn_axis_r8")
    manifest = export_web.export(stamp, out / "net", n_positions=16)
    (out / "net" / "index.json").write_text(json.dumps({"manifest": f"{stamp.name.removesuffix('.ckpt')}.manifest.json"}),
                                           encoding="utf-8")
    print(f"exported {manifest['stamp']['file']} into {out / 'net'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
