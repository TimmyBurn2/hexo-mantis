"""One supervised arm on a frozen ring through the production trainer: the base recipe's config with the arm's shape, seeded, saved at the panel's steps."""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from mantis._engine import HexgBuffer, registry_sha_hex
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.config.resolve.sample_threads import resolve_sample_threads
from mantis.diagnostics.ring_reader import _read_header
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import deploy_state, load_checkpoint
from mantis.train.coordinator.dispatch import (
    resolve_step_spec,
    run_declared_eval_step,
    run_declared_train_step,
)
from mantis.train.orchestrator import init_trainer
from mantis.util.determinism import seed_everything
from mantis.util.hashing import sha256_file


@dataclass(frozen=True)
class ArmSpec:
    """What makes one arm: its shape, start, ring, value mask and the teachers its labels came from."""

    run_id: str
    hidden: int
    layers: int
    ring: Path
    warm_start: Path | None
    value_mask_p: float
    lineage: tuple[str, ...]


@dataclass(frozen=True)
class Recipe:
    """What every arm shares: one cosine from `lr` to `eta_min` over `steps`, saves, seed."""

    base_config: Path
    steps: int
    save_at: tuple[int, ...]
    lr: float
    eta_min: float
    seed: int


class _JsonlSink:
    """The trainer's event sink as one JSON line per event."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def emit(self, event: Any) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(event), default=str) + "\n")


def warm_start_row(path: Path, encoding: str) -> dict[str, Any]:
    """`identity.warm_start` for `path`: its deploy net's hash, nothing re-initialised."""
    ck = load_checkpoint(path, declared_encoding=encoding)
    if ck.metadata.arch is None:
        raise ValueError(f"{path.name}: the stamp resolves no arch")
    state, _ = deploy_state(ck)
    net = build_net(ck.metadata.arch)
    net.load_state_dict(state)
    return {"checkpoint": str(path), "net_hash": net_param_hash(net), "reinit": []}


def arm_config(arm: ArmSpec, recipe: Recipe) -> dict[str, Any]:
    """The base recipe with the arm's shape, start, one cosine and value mask; no held-out witness, no warm start unless declared."""
    dump = load_config(recipe.base_config).model_dump()
    encoding = dump["identity"]["encoding"]
    dump["run_id"] = arm.run_id
    dump["seed"] = recipe.seed
    dump["model"]["gnn"] = {"hidden": arm.hidden, "num_layers": arm.layers}
    dump["identity"]["warm_start"] = None if arm.warm_start is None else warm_start_row(arm.warm_start, encoding)
    dump["train"].update(lr=recipe.lr, eta_min=recipe.eta_min, scheduler_t_max=recipe.steps, lr_cycle=None,
                         value_mask_redraw_p=arm.value_mask_p, heldout_gap=None)
    return dump


def open_ring(path: Path, encoding: str, seed: int) -> tuple[HexgBuffer, int]:
    """A frozen ring in a buffer sized to it, its sampler seeded; Raises: ValueError — the ring loads no record."""
    header, _ = _read_header(memoryview(path.read_bytes()[:4096]))
    buf = HexgBuffer(max(header.size, 8), encoding, header.max_visits)
    rows = int(buf.load_from_path(str(path)))
    if rows < 1:
        raise ValueError(f"{path.name}: loaded no records")
    buf.seed_sampler(seed)
    return buf, rows


def train_arm(arm: ArmSpec, recipe: Recipe, out: Path, *, device: str, heldout_ring: Path | None,
              heldout_every: int, heldout_batches: int, sample_threads: int | None,
              caps_override: tuple[int, int] | None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train `arm` for `recipe.steps` production steps and save at `recipe.save_at`; Raises: FileExistsError — `out` exists; RuntimeError — the arm skips more steps than it takes."""
    out.mkdir(parents=True, exist_ok=False)
    dump = arm_config(arm, recipe)
    if caps_override is not None:
        dump["train"]["microbatch_caps"] = {"max_edges": caps_override[0], "max_nodes": caps_override[1]}
    (out / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")
    config = load_config(out / "config.yaml")
    encoding = config.identity.encoding
    knobs = resolve_coordinator_knobs(config.train)
    threads = sample_threads or resolve_sample_threads(dump)
    seed_everything(recipe.seed)
    trainer = init_trainer(config=dump, device=torch.device(device), checkpoint_dir=str(out / "checkpoints"),
                           sink=_JsonlSink(out / "events.jsonl"))
    trainer.lineage = tuple(dict.fromkeys(trainer.lineage + arm.lineage))
    buf, rows = open_ring(arm.ring, encoding, recipe.seed)
    spec, caps = resolve_step_spec(dump), (lambda: resolve_microbatch_caps(dump))
    batch, augment = int(knobs.batch_size), bool(knobs.augment)
    held = None if heldout_ring is None else open_ring(heldout_ring, encoding, recipe.seed)[0]
    tree = Path(__file__).resolve().parents[2]
    record: dict[str, Any] = {
        "run_id": arm.run_id, "hidden": arm.hidden, "layers": arm.layers, "encoding": encoding,
        "lineage": list(trainer.lineage), "ring": str(arm.ring), "ring_sha256": sha256_file(arm.ring), "ring_rows": rows,
        "warm_start": dump["identity"]["warm_start"], "value_mask_p": arm.value_mask_p, "seed": recipe.seed,
        "steps": recipe.steps, "lr": recipe.lr, "eta_min": recipe.eta_min, "batch_size": batch, "augment": augment,
        "sample_threads": threads, "microbatch_caps": dump["train"]["microbatch_caps"],
        "config_sha256": sha256_file(out / "config.yaml"), "registry_sha256": registry_sha_hex(),
        "tree_commit": subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True,
                                      check=False).stdout.strip(),
        "net_hash_start": net_param_hash(trainer._base_model()), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }

    def read_heldout() -> dict[str, float] | None:
        if held is None:
            return None
        held.seed_sampler(recipe.seed)
        sums = {"policy_loss": 0.0, "value_loss": 0.0}
        for _ in range(heldout_batches):
            got = run_declared_eval_step(trainer, held, spec, batch_size=batch, caps_provider=caps,
                                         sample_threads_provider=lambda: threads)
            for k in sums:
                sums[k] += float(got[k]) / heldout_batches
        return sums

    curve: list[dict[str, Any]] = [{"step": 0, "heldout": read_heldout()}]
    saves: dict[int, dict[str, str]] = {}
    t0 = time.time()
    attempts = 0
    while trainer.step < recipe.steps:
        attempts += 1
        if attempts > recipe.steps + 100:
            raise RuntimeError(f"{trainer.skipped_steps} skipped steps: the arm is not training")
        before = trainer.step
        info = run_declared_train_step(trainer, buf, spec, batch_size=batch, augment=augment, caps_provider=caps,
                                       sample_threads_provider=lambda: threads)
        if trainer.step == before:
            continue
        if trainer.step % 250 == 0:
            row: dict[str, Any] = {"step": trainer.step, "loss": float(info["loss"]),
                                   "policy": float(info.get("policy_loss", float("nan"))),
                                   "value": float(info.get("value_loss", float("nan"))),
                                   "lr": float(trainer.optimizer.param_groups[0]["lr"]), "t": round(time.time() - t0, 1)}
            if trainer.step % heldout_every == 0:
                row["heldout"] = read_heldout()
            curve.append(row)
            log(json.dumps(row))
        if trainer.step in recipe.save_at:
            saves[trainer.step] = {"path": str(trainer.save_checkpoint(None)),
                                   "net_hash": net_param_hash(trainer._base_model())}
    wall = time.time() - t0
    record.update(saves=saves, skipped_steps=trainer.skipped_steps, train_wall_s=round(wall, 1),
                  s_per_step=round(wall / max(trainer.step, 1), 4))
    (out / "curve.json").write_text(json.dumps(curve), encoding="utf-8")
    (out / "arm.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record
