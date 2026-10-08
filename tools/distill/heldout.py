"""Held-out reads against the teacher: KL(teacher ‖ net) on its policy and the value against its value, every net on the same production samples."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from mantis.util.hashing import sha256_file
from mantis.util.loadpkg import load_tools_package


def _nets_module() -> Any:
    load_tools_package("probe1")
    return importlib.import_module("probe1.nets")


def read_kl(ckpts: list[Path], heldout_ring: Path, *, batches: int, batch_size: int, threads: int,
            seed: int) -> dict[str, Any]:
    """Per net: mean KL(teacher ‖ net), policy CE, the teacher's entropy, value MSE and two-hot CE to the teacher's value."""
    nets_mod = _nets_module()
    nets = [nets_mod.load_net(p) for p in ckpts]
    encoding = nets[0].spec.name
    buffer, rows = nets_mod.open_ring(heldout_ring, seed=seed, encoding=encoding)
    read = nets_mod.read_ring(nets, buffer, batches=batches, batch_size=batch_size, threads=threads, rows=True)
    out: dict[str, Any] = {"heldout_ring": str(heldout_ring), "heldout_sha256": sha256_file(heldout_ring),
                           "heldout_rows": rows, "batches": batches, "batch_size": batch_size, "seed": seed, "nets": []}
    for net in nets:
        r = {k.split(":", 1)[1]: v for k, v in read["rows"].items() if k.startswith(net.net_hash + ":")}
        kl = r["kl_target_prior"]
        out["nets"].append({
            "checkpoint": net.path.name, "step": net.step, "net_hash": net.net_hash,
            "kl_teacher_net": float(kl.mean()), "kl_se": float(kl.std(ddof=1) / np.sqrt(len(kl))),
            "policy_ce": float(r["policy_ce"].mean()), "teacher_entropy": float((r["policy_ce"] - kl).mean()),
            "value_mse": float(((r["ev"] - r["z"]) ** 2).mean()), "value_ce": float(r["value_ce"].mean()),
            "rows_read": int(len(kl)),
        })
    return out


def _segment_argmax(values: Any, offsets: Any) -> np.ndarray:
    """Per graph, the local index of the largest value over its legal segment."""
    v, off = values.detach().float().cpu().numpy().reshape(-1), offsets.cpu().numpy()
    return np.array([int(np.argmax(v[off[g]:off[g + 1]])) for g in range(len(off) - 1)], dtype=np.int64)


def read_baseline(ckpt: Path, ring: Path, *, batches: int, batch_size: int, threads: int, seed: int) -> dict[str, Any]:
    """A net against a teacher-policy ring whose z is the game's: top-1 agreement with the teacher, KL, and its value's correlation with z."""
    import torch

    nets_mod = _nets_module()
    net = nets_mod.load_net(ckpt)
    buffer, rows = nets_mod.open_ring(ring, seed=seed, encoding=net.spec.name)
    agree: list[np.ndarray] = []
    cols: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for _ in range(batches):
            parts = nets_mod.sample_parts(buffer, net.config, net.spec, batch_size=batch_size, threads=threads)
            for make in parts["parts"]:
                inputs = make()
                logits, _v, _bins = net.model.forward_batch(  # pyright: ignore[reportCallIssue]
                    inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index, inputs.stone_mask,
                    node_offsets=inputs.node_offsets)
                agree.append(_segment_argmax(logits, inputs.legal_offsets)
                             == _segment_argmax(inputs.policy_target, inputs.legal_offsets))
                for key, arr in nets_mod.read_rows(net.model, inputs).items():
                    cols.setdefault(key, []).append(arr)
    r = {k: np.concatenate(v) for k, v in cols.items()}
    decided = r["valid"] & (r["z"] != 0)
    return {"checkpoint": ckpt.name, "net_hash": net.net_hash, "ring": str(ring), "ring_rows": rows,
            "rows_read": int(len(r["z"])), "top1_agreement_with_teacher": float(np.concatenate(agree).mean()),
            "kl_teacher_net": float(r["kl_target_prior"].mean()),
            "value_v_z_pearson": float(np.corrcoef(r["ev"][decided], r["z"][decided])[0, 1]),
            "value_v_z_sign_agreement": float((np.sign(r["ev"][decided]) == np.sign(r["z"][decided])).mean())}
