"""One supervised arm on a frozen ring through the production trainer: the base recipe's config with the arm's shape, seeded, saved at the panel's steps."""
from __future__ import annotations

import importlib
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
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.model.widen import widen_net
from mantis.train.checkpoints import (
    CheckpointStampError,
    deploy_state,
    load_checkpoint,
    resume_trainer,
)
from mantis.train.coordinator.dispatch import (
    resolve_step_spec,
    run_declared_eval_step,
    run_declared_train_step,
)
from mantis.train.orchestrator import build_fresh_trainer
from mantis.train.trainer.core import Trainer
from mantis.util.determinism import seed_everything
from mantis.util.hashing import sha256_file
from mantis.util.loadpkg import load_tools_package

from .rings import PROVENANCE_SUFFIX, ring_lineage


@dataclass(frozen=True)
class ArmSpec:
    """What makes one arm: its shape, start, ring and value mask (the teachers come with the ring's provenance)."""

    run_id: str
    hidden: int
    layers: int
    ring: Path
    warm_start: Path | None
    value_mask_p: float


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
    """`identity.warm_start` for `path`: its deploy net's hash, nothing re-initialised; Raises: ValueError — the stamp resolves no arch."""
    ck = load_checkpoint(path, declared_encoding=encoding)
    if ck.metadata.arch is None:
        raise ValueError(f"{path.name}: the stamp resolves no arch")
    state, _ = deploy_state(ck)
    net = build_net(ck.metadata.arch)
    net.load_state_dict(state)
    return {"checkpoint": str(path), "net_hash": net_param_hash(net), "reinit": []}


def arm_config(arm: ArmSpec, recipe: Recipe) -> dict[str, Any]:
    """The base recipe with the arm's shape, start, one cosine and value mask; no held-out witness, no periodic save, no warm start unless declared."""
    dump = load_config(recipe.base_config).model_dump()
    encoding = dump["identity"]["encoding"]
    dump["run_id"] = arm.run_id
    dump["seed"] = recipe.seed
    dump["model"]["gnn"] = {"hidden": arm.hidden, "num_layers": arm.layers}
    dump["identity"]["warm_start"] = None if arm.warm_start is None else warm_start_row(arm.warm_start, encoding)
    dump["train"].update(lr=recipe.lr, eta_min=recipe.eta_min, scheduler_t_max=recipe.steps, lr_cycle=None,
                         value_mask_redraw_p=arm.value_mask_p, heldout_gap=None, checkpoint_interval=0)
    return dump


def widen_checkpoint(parent: Path, base_config: Path, out: Path, *, run_id: str, seed: int) -> dict[str, Any]:
    """The parent's deploy net doubled in trunk width (`widen_net`), stamped at step 0 by the trainer's one writer under `base_config` at the doubled width, its lineage the parent's plus `widened-from-<parent>`. Raises: FileExistsError — `out` exists; CheckpointStampError — the parent's stamp resolves no arch; WidenError — a tensor the doubling has no rule for."""
    ck = load_checkpoint(parent, declared_encoding=None)
    if ck.metadata.arch is None:
        raise CheckpointStampError(f"{parent.name}: the stamp resolves no arch")
    state, source = deploy_state(ck)
    net = build_net(ck.metadata.arch)
    net.load_state_dict(state)
    wide = widen_net(net, seed=seed)
    out.mkdir(parents=True, exist_ok=False)
    dump = load_config(base_config).model_dump()
    dump["run_id"] = run_id
    dump["model"]["gnn"] = {"hidden": wide.arch.hidden, "num_layers": wide.arch.num_layers}  # pyright: ignore
    dump["identity"]["warm_start"] = None
    trainer = build_fresh_trainer(config=dump, device=torch.device("cpu"), checkpoint_dir=str(out))
    trainer._base_model().load_state_dict(wide.state_dict(), strict=True)
    trainer.lineage = (*ck.metadata.lineage, f"widened-from-{parent.stem}")
    path = trainer.save_checkpoint(None)
    return {"parent": str(parent), "parent_sha256": sha256_file(parent), "parent_state": source,
            "parent_net_hash": net_param_hash(net), "widened": str(path), "widened_sha256": sha256_file(path),
            "widened_net_hash": net_param_hash(wide), "hidden": [net.arch.hidden, wide.arch.hidden],  # pyright: ignore
            "seed": seed, "lineage": list(trainer.lineage)}


def open_ring(path: Path, encoding: str, seed: int) -> tuple[HexgBuffer, int]:
    """A frozen ring in a buffer sized to it, its sampler seeded (the probe package's reader); Raises: ValueError — the ring loads no record."""
    load_tools_package("probe1")
    return importlib.import_module("probe1.nets").open_ring(path, seed=seed, encoding=encoding)


def resume_seed(seed: int, step: int) -> int:
    """The sampler and torch seed a resume at `step` draws from: an arm stopped at the same steps replays the same stream."""
    return seed + step


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _step_of(ckpt: Path) -> int:
    return int(ckpt.stem.split("_")[-2])


def train_arm(arm: ArmSpec, recipe: Recipe, out: Path, *, device: str, heldout_ring: Path | None,
              heldout_every: int, heldout_batches: int, sample_threads: int | None,
              caps_override: tuple[int, int] | None, save_every: int = 0, gpu_mem_mib: int | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train `arm` for `recipe.steps` production steps and save at `recipe.save_at`, every `save_every` steps beside them; an unfinished `out` resumes from its newest save. Raises: FileExistsError — `out` holds a finished arm, or an unfinished one with no save; FileNotFoundError, ValueError — the ring's provenance or a ring that loads nothing; RuntimeError — the arm skips more steps than it takes."""
    lineage = ring_lineage(arm.ring)
    progress = out / "progress.jsonl"
    if gpu_mem_mib is not None and device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_mem_mib * 2**20 / total))
    if out.exists():
        if (out / "arm.json").exists():
            raise FileExistsError(f"{out}: the arm is finished")
        ckpts = sorted((out / "checkpoints").glob("*.ckpt"), key=_step_of)
        if not ckpts:
            raise FileExistsError(f"{out}: an unfinished arm with no save to resume from")
        dump = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        config = load_config(out / "config.yaml")
        encoding = config.identity.encoding
        seed = resume_seed(recipe.seed, _step_of(ckpts[-1]))
        seed_everything(seed)
        trainer = resume_trainer(Trainer, ckpts[-1], declared_encoding=encoding, device=torch.device(device),
                                 sink=_JsonlSink(out / "events.jsonl"))
        _append(progress, {"event": "resume", "step": trainer.step, "from": str(ckpts[-1]), "seed": seed,
                           "t": time.time()})
        log(f"resumed at {trainer.step} from {ckpts[-1].name}, seed {seed}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        dump = arm_config(arm, recipe)
        if caps_override is not None:
            dump["train"]["microbatch_caps"] = {"max_edges": caps_override[0], "max_nodes": caps_override[1]}
        (out / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")
        config = load_config(out / "config.yaml")
        encoding = config.identity.encoding
        seed = recipe.seed
        seed_everything(seed)
        trainer = build_fresh_trainer(config=dump, device=torch.device(device), checkpoint_dir=str(out / "checkpoints"),
                                      sink=_JsonlSink(out / "events.jsonl"))
        trainer.lineage = tuple(dict.fromkeys(trainer.lineage + lineage))
        _append(progress, {"event": "start", "step": 0, "seed": seed, "t": time.time()})
    knobs = resolve_coordinator_knobs(config.train)
    threads = sample_threads or resolve_sample_threads(dump)
    buf, rows = open_ring(arm.ring, encoding, seed)
    step_spec, caps = resolve_step_spec(dump), (lambda: resolve_microbatch_caps(dump))
    batch, augment = int(knobs.batch_size), bool(knobs.augment)
    held = None if heldout_ring is None else open_ring(heldout_ring, encoding, recipe.seed)[0]
    tree = Path(__file__).resolve().parents[2]
    if not (out / "start.json").exists():
        (out / "start.json").write_text(json.dumps({
            "run_id": arm.run_id, "hidden": arm.hidden, "layers": arm.layers, "encoding": encoding,
            "lineage": list(trainer.lineage), "ring": str(arm.ring), "ring_sha256": sha256_file(arm.ring),
            "ring_rows": rows, "ring_provenance": str(arm.ring) + PROVENANCE_SUFFIX,
            "warm_start": dump["identity"]["warm_start"], "value_mask_p": arm.value_mask_p, "seed": recipe.seed,
            "steps": recipe.steps, "lr": recipe.lr, "eta_min": recipe.eta_min, "batch_size": batch, "augment": augment,
            "sample_threads": threads, "microbatch_caps": dump["train"]["microbatch_caps"], "save_every": save_every,
            "gpu_mem_mib": gpu_mem_mib, "config_sha256": sha256_file(out / "config.yaml"),
            "registry_sha256": registry_sha_hex(),
            "tree_commit": subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                                          text=True, check=False).stdout.strip(),
            "net_hash_start": net_param_hash(trainer._base_model()), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }, indent=1, default=str), encoding="utf-8")

    def read_heldout() -> dict[str, float] | None:
        if held is None:
            return None
        held.seed_sampler(recipe.seed)
        sums = {"policy_loss": 0.0, "value_loss": 0.0}
        for _ in range(heldout_batches):
            got = run_declared_eval_step(trainer, held, step_spec, batch_size=batch, caps_provider=caps,
                                         sample_threads_provider=lambda: threads)
            for k in sums:
                sums[k] += float(got[k]) / heldout_batches
        return sums

    curve = out / "curve.jsonl"
    if trainer.step == 0:
        _append(curve, {"step": 0, "heldout": read_heldout()})
    t0, start_step = time.time(), trainer.step
    attempts = 0
    while trainer.step < recipe.steps:
        attempts += 1
        if attempts > recipe.steps - start_step + 100:
            raise RuntimeError(f"{trainer.skipped_steps} skipped steps: the arm is not training")
        before = trainer.step
        info = run_declared_train_step(trainer, buf, step_spec, batch_size=batch, augment=augment, caps_provider=caps,
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
            _append(curve, row)
            log(json.dumps(row))
        named = trainer.step in recipe.save_at
        if named or (save_every and trainer.step % save_every == 0):
            path = trainer.save_checkpoint(None)
            _append(progress, {"event": "save", "step": trainer.step, "path": str(path), "named": named,
                               "net_hash": net_param_hash(trainer._base_model()), "t": time.time(),
                               "segment_s": round(time.time() - t0, 1), "segment_steps": trainer.step - start_step})
            for old in (out / "checkpoints").glob("*.ckpt"):
                if old != path and _step_of(old) not in recipe.save_at:
                    old.unlink()
    events = _rows(progress)
    saves = {e["step"]: {"path": e["path"], "net_hash": e["net_hash"]} for e in events
             if e["event"] == "save" and e["named"]}
    wall, segment = 0.0, 0.0
    for e in events:
        if e["event"] == "save":
            segment = e["segment_s"]
        else:
            wall, segment = wall + segment, 0.0
    wall += segment
    record = json.loads((out / "start.json").read_text(encoding="utf-8"))
    record.update(saves=saves, skipped_steps=trainer.skipped_steps, train_wall_s=round(wall, 1),
                  s_per_step=round(wall / max(trainer.step, 1), 4),
                  resumes=[e for e in events if e["event"] == "resume"])
    (out / "arm.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record
