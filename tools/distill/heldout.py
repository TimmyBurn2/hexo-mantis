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
