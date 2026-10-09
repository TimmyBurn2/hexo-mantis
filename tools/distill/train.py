# >300 justify (R8): an arm's whole life is one unit — its recipe, its start (fresh, warm, or a widened stamp), the rolling
# saves it writes and the resume that reads them back, and the record assembled from those saves; split, the resume would
# read a save layout and a progress log that another file defines.
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
import torch.nn as nn
import yaml

from mantis._engine import HexgBuffer, registry_sha_hex
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.config.resolve.sample_threads import resolve_sample_threads
from mantis.model import ModelArch, build_net
from mantis.model.identity import net_param_hash
from mantis.model.widen import widen_arch, widen_net
from mantis.train.bundle_receipts import CHECKPOINT_NAME_RE, stamped_checkpoints
from mantis.train.checkpoints import (
    CheckpointStampError,
    deploy_state,
    load_checkpoint,
    resume_trainer,
    stamped_lineage,
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


def deploy_net(path: Path, encoding: str | None) -> tuple[nn.Module, ModelArch, str]:
    """A stamped checkpoint's deploy net (the EMA shadow when stamped, else the learner), its declared arch and which state it is; Raises: CheckpointStampError — the stamp resolves no arch; DeclaredEncodingMismatchError — it names another encoding than `encoding`."""
    ck = load_checkpoint(path, declared_encoding=encoding)
    if ck.metadata.arch is None:
        raise CheckpointStampError(f"{path.name}: the stamp resolves no arch")
    state, source = deploy_state(ck)
    net = build_net(ck.metadata.arch)
    net.load_state_dict(state)
    return net, ck.metadata.arch, source


def warm_start_row(path: Path, encoding: str) -> dict[str, Any]:
    """`identity.warm_start` for `path`: its deploy net's hash, nothing re-initialised; Raises: as `deploy_net`."""
    return {"checkpoint": str(path), "net_hash": net_param_hash(deploy_net(path, encoding)[0]), "reinit": []}


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
    """The parent's deploy net doubled (`widen_net`) and stamped at step 0 by the trainer's one writer under `base_config` at the doubled width, EMA off, lineage the parent's plus `widened-from-<parent>`; Raises: FileExistsError — `out` exists; CheckpointStampError, DeclaredEncodingMismatchError — as `deploy_net`; WidenError — a tensor without a rule; RuntimeError — the stamp's deploy net is not the doubled net."""
    dump = load_config(base_config).model_dump()
    encoding = dump["identity"]["encoding"]
    net, arch, source = deploy_net(parent, encoding)
    wide, wide_arch = widen_net(net, arch, seed=seed), widen_arch(arch)
    out.mkdir(parents=True, exist_ok=False)
    dump["run_id"] = run_id
    dump["model"]["gnn"] = {"hidden": wide_arch.hidden, "num_layers": wide_arch.num_layers}
    dump["identity"]["warm_start"] = None
    dump["train"]["ema"] = {**dump["train"]["ema"], "enabled": False}
    trainer = build_fresh_trainer(config=dump, device=torch.device("cpu"), checkpoint_dir=str(out))
    trainer._base_model().load_state_dict(wide.state_dict(), strict=True)
    trainer.lineage = (*stamped_lineage(parent), f"widened-from-{parent.stem}")
    path = trainer.save_checkpoint(None)
    if net_param_hash(deploy_net(path, encoding)[0]) != net_param_hash(wide):
        raise RuntimeError(f"{path.name}: the stamp's deploy net is not the doubled net")
    return {"parent": str(parent), "parent_sha256": sha256_file(parent), "parent_state": source,
            "parent_net_hash": net_param_hash(net), "widened": str(path), "widened_sha256": sha256_file(path),
            "widened_net_hash": net_param_hash(wide), "hidden": [arch.hidden, wide_arch.hidden], "seed": seed,
            "lineage": list(trainer.lineage)}


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
    match = CHECKPOINT_NAME_RE.match(ckpt.name)
    if match is None:
        raise ValueError(f"{ckpt.name}: not a stamped checkpoint name")
    return int(match["step"])


def _segments(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each start or resume's last save event: the wall, steps and skipped steps the segment kept."""
    out: list[dict[str, Any]] = []
    for e in events:
        if e["event"] in ("start", "resume"):
            out.append({})
        elif e["event"] == "save" and "segment_s" in e and out:
            out[-1] = e
    return [e for e in out if e]


def train_arm(arm: ArmSpec, recipe: Recipe, out: Path, *, device: str, heldout_ring: Path | None,
              heldout_every: int, heldout_batches: int, sample_threads: int | None,
              caps_override: tuple[int, int] | None, save_every: int = 0, gpu_mem_mib: int | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train `arm` for `recipe.steps` steps, saving at `recipe.save_at` and every `save_every` steps; an unfinished `out` resumes from its newest save (one with none is moved aside and restarts); Raises: FileExistsError — `out` holds a finished arm; ValueError — a rerun whose arm or recipe differs from the stopped one's, a ring that loads nothing or a provenance for other bytes; FileNotFoundError — no provenance; CheckpointStampError, DeclaredEncodingMismatchError, ResumeIdentityMismatchError, ResumeTargetSemanticsError — the save refuses to resume; RuntimeError — the arm skips more steps than it takes."""
    lineage = ring_lineage(arm.ring)
    if gpu_mem_mib is not None and device.startswith("cuda"):
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_mem_mib * 2**20 / total), index)
    if (out / "arm.json").exists():
        raise FileExistsError(f"{out}: the arm is finished")
    saved = stamped_checkpoints(out / "checkpoints")
    if out.exists() and not saved:
        aside = out.with_name(f"{out.name}.unsaved.{int(time.time())}")
        out.rename(aside)
        log(f"{out} held no save: moved aside to {aside.name}, the arm restarts at step 0")
    dump = arm_config(arm, recipe)
    if caps_override is not None:
        dump["train"]["microbatch_caps"] = {"max_edges": caps_override[0], "max_nodes": caps_override[1]}
    progress, curve = out / "progress.jsonl", out / "curve.jsonl"
    plan = {"save_at": list(recipe.save_at), "save_every": save_every, "ring": str(arm.ring)}
    if saved:
        stored = (out / "config.yaml").read_text(encoding="utf-8")
        begun = json.loads((out / "start.json").read_text(encoding="utf-8"))
        if stored != yaml.safe_dump(dump, sort_keys=False) or {k: begun.get(k) for k in plan} != plan:
            raise ValueError(f"{out}: this run's arm or recipe differs from the one the stopped arm started with")
        encoding = load_config(out / "config.yaml").identity.encoding
        step = _step_of(saved[-1])
        seed = resume_seed(recipe.seed, step)
        seed_everything(seed)
        trainer = resume_trainer(Trainer, saved[-1], declared_encoding=encoding, device=torch.device(device),
                                 sink=_JsonlSink(out / "events.jsonl"))
        if not any(e["event"] == "save" and e["step"] == step for e in _rows(progress)):
            _append(progress, {"event": "save", "step": step, "path": str(saved[-1]),
                               "named": step in recipe.save_at, "net_hash": net_param_hash(trainer._base_model()),
                               "t": time.time(), "backfilled": True})
        _append(progress, {"event": "resume", "step": step, "from": str(saved[-1]), "seed": seed, "t": time.time(),
                           "gpu_mem_mib": gpu_mem_mib, "device": device})
        curve.write_text("".join(json.dumps(r) + "\n" for r in _rows(curve) if r["step"] <= step), encoding="utf-8")
        log(f"resumed at {step} from {saved[-1].name}, seed {seed}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")
        encoding = load_config(out / "config.yaml").identity.encoding
        seed = recipe.seed
        seed_everything(seed)
        trainer = build_fresh_trainer(config=dump, device=torch.device(device), checkpoint_dir=str(out / "checkpoints"),
                                      sink=_JsonlSink(out / "events.jsonl"))
        trainer.lineage = tuple(dict.fromkeys(trainer.lineage + lineage))
        _append(progress, {"event": "start", "step": 0, "seed": seed, "t": time.time(), "gpu_mem_mib": gpu_mem_mib,
                           "device": device})
    knobs = resolve_coordinator_knobs(load_config(out / "config.yaml").train)
    threads = sample_threads or resolve_sample_threads(dump)
    buf, rows = open_ring(arm.ring, encoding, seed)
    step_spec, caps = resolve_step_spec(dump), (lambda: resolve_microbatch_caps(dump))
    batch, augment = int(knobs.batch_size), bool(knobs.augment)
    held = None if heldout_ring is None else open_ring(heldout_ring, encoding, recipe.seed)[0]
    tree = Path(__file__).resolve().parents[2]
    if not (out / "start.json").exists():
        (out / "start.json").write_text(json.dumps({
            "run_id": arm.run_id, "hidden": arm.hidden, "layers": arm.layers, "encoding": encoding,
            "lineage": list(trainer.lineage), "ring_sha256": sha256_file(arm.ring), "ring_rows": rows,
            "ring_provenance": str(arm.ring) + PROVENANCE_SUFFIX, "warm_start": dump["identity"]["warm_start"],
            "value_mask_p": arm.value_mask_p, "seed": recipe.seed, "steps": recipe.steps, "lr": recipe.lr,
            "eta_min": recipe.eta_min, "batch_size": batch, "augment": augment, "sample_threads": threads,
            "microbatch_caps": dump["train"]["microbatch_caps"], "gpu_mem_mib": gpu_mem_mib, **plan,
            "config_sha256": sha256_file(out / "config.yaml"), "registry_sha256": registry_sha_hex(),
            "tree_commit": subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                                          text=True, check=False).stdout.strip(),
            "net_hash_start": net_param_hash(trainer._base_model()), "device": device,
            "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
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
                               "segment_s": round(time.time() - t0, 1), "segment_steps": trainer.step - start_step,
                               "skipped_steps": trainer.skipped_steps})
            for old in stamped_checkpoints(out / "checkpoints"):
                if old != path and _step_of(old) not in recipe.save_at:
                    old.unlink()
    events = _rows(progress)
    hashes = {e["step"]: e["net_hash"] for e in events if e["event"] == "save"}
    saves = {_step_of(p): {"path": str(p), "net_hash": hashes.get(_step_of(p)) or net_param_hash(deploy_net(p, encoding)[0])}
             for p in stamped_checkpoints(out / "checkpoints") if _step_of(p) in recipe.save_at}
    kept = _segments(events)
    wall, steps = sum(e["segment_s"] for e in kept), sum(e["segment_steps"] for e in kept)
    record = json.loads((out / "start.json").read_text(encoding="utf-8"))
    record.update(saves=saves, skipped_steps=sum(e["skipped_steps"] for e in kept), train_wall_s=round(wall, 1),
                  trained_steps=steps, s_per_step=round(wall / max(steps, 1), 4),
                  resumes=[e for e in events if e["event"] == "resume"])
    (out / "arm.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record
