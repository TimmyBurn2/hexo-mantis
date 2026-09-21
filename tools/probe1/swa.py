"""Reading 7's prerequisite: the uniform weight-average of a checkpoint span, written through the ONE writer with its derivation beside it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from mantis.config.loader import load_config
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import load_checkpoint, save_checkpoint


def average_checkpoints(paths: list[Path], *, config_path: Path, run_id: str, out_dir: Path) -> dict[str, Any]:
    """Equal-weight average of `paths`' floating tensors, stamped weights-only under `run_id` at the last step, `.derivation.json` beside it; Raises: ValueError on a shape mismatch."""
    if not paths:
        raise ValueError("no source checkpoints")
    sources, arch, shapes = [], None, None
    acc: dict[str, torch.Tensor] = {}
    for p in paths:
        ck = load_checkpoint(p)
        if ck.metadata.arch is None:
            raise ValueError(f"{p.name}: the stamp resolves no arch")
        if arch is None:
            arch = ck.metadata.arch
        state = {k: v.detach().to(torch.float64) if v.is_floating_point() else v for k, v in ck.model_state.items()}
        if shapes is None:
            shapes = {k: tuple(v.shape) for k, v in state.items()}
        elif {k: tuple(v.shape) for k, v in state.items()} != shapes:
            raise ValueError(f"{p.name}: state shapes differ from {paths[0].name}")
        for k, v in state.items():
            acc[k] = v.clone() if k not in acc else (acc[k] + v if v.is_floating_point() else v)
        net = build_net(ck.metadata.arch)
        net.load_state_dict(ck.model_state)
        sources.append({"file": p.name, "step": int(ck.metadata.step), "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "net_hash": net_param_hash(net)})
    assert arch is not None
    n = len(paths)
    averaged = {k: (v / n).to(torch.float32) if v.is_floating_point() else v for k, v in acc.items()}
    net = build_net(arch)
    net.load_state_dict(averaged)
    config = load_config(config_path).model_dump()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = save_checkpoint(model=net, optimizer=None, scaler=None, scheduler=None, step=sources[-1]["step"], config=config,
                              kind="weights", checkpoint_dir=out_dir,
                              metadata_kwargs={"encoding_name": config["identity"]["encoding"], "run_id": run_id, "arch": arch,
                                               "corpus_sha256": config.get("corpus_sha256")})
    record = {"kind": "uniform weight average (SWA over stamped checkpoints)", "n": n, "sources": sources,
              "config": str(config_path), "checkpoint": written.name, "net_hash": net_param_hash(net),
              "file_sha256": hashlib.sha256(written.read_bytes()).hexdigest()}
    written.with_name(written.name + ".derivation.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


__all__ = ["average_checkpoints"]
