"""snapshot.py — the ONLY parent-side torch touch in mantis.eval (design §a.3).

`write_model_snapshot` never runs a forward, never calls `.cuda()`, takes NO device
argument — the parent-side write path only ever serializes CPU tensors from a model
already resident wherever the caller built it (isolation law: models arrive only through
`run_evaluation`'s protocol args and are IMMEDIATELY serialized-and-dropped). The
worker-side (child-process) counterpart, `load_model_snapshot`, is the one place a snapshot
is rebuilt into a live module — `torch.load(weights_only=True)` + `build_net` from the
snapshot's plain-dict arch (never a live-module sniff).

Snapshots are SPOOL FILES, never checkpoints: no envelope version, no checkpoint stamp —
the ONE checkpoint loader (`mantis.train.checkpoints`) stays the only checkpoint reader
(LAW-12 one-loader carve-out).
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import torch

from mantis.model import CnnArch, GnnArch, build_net

_ARCH_TYPES: dict[str, type] = {"CnnArch": CnnArch, "GnnArch": GnnArch}


def _arch_to_plain_dict(arch: Any) -> dict[str, Any]:
    type_tag = type(arch).__name__
    if type_tag not in _ARCH_TYPES:
        raise TypeError(f"write_model_snapshot: unsupported arch type {type_tag!r}")
    payload = dataclasses.asdict(arch)
    payload["__arch_type__"] = type_tag
    return payload


def _plain_dict_to_arch(payload: dict[str, Any]) -> Any:
    payload = dict(payload)
    type_tag = payload.pop("__arch_type__")
    arch_cls = _ARCH_TYPES[type_tag]
    return arch_cls(**payload)


def write_model_snapshot(model: torch.nn.Module, path: str | Path) -> str:
    """Snapshot `model`'s CPU weights + its declared `.arch` to `path`; return the sha256
    of the written file. NO forward, NO `.cuda()`, NO device argument."""
    arch = getattr(model, "arch", None)
    if arch is None:
        raise AttributeError(
            "write_model_snapshot: model carries no declared '.arch' attribute (the "
            "arch-travels-with-the-model convention this snapshot relies on)"
        )
    base = getattr(model, "_orig_mod", model)
    state_dict = {key: value.detach().cpu() for key, value in base.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        "arch": _arch_to_plain_dict(arch),
        "encoding": getattr(model, "encoding", None),
        "representation": getattr(arch, "representation", None),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)

    sha = hashlib.sha256()
    with open(target, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_model_snapshot(path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Worker-side (child-process) load: rebuild the IDENTICAL net from the snapshot's
    plain-dict arch via `build_net`, load its weights, move to `device`."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = _plain_dict_to_arch(payload["arch"])
    model = build_net(arch)
    model.load_state_dict(payload["state_dict"])
    model.arch = arch
    model.to(device)
    model.eval()
    return model


__all__ = ["load_model_snapshot", "write_model_snapshot"]
