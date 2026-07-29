"""Bootstrap pretrain CLI (WP10 §a.7 IMPROVE of `bootstrap/pretrain_cli.py`).

The `python -m mantis.train.pretrain` entry: argparse surface + config resolution + corpus load +
model build + train/save/validate orchestration. Well under the soft cap (the old CLI was 558 L)
because the KILLED-branch flags are gone.

Ratified WP10 amendments over a pure relocation:
  * **Personal/hardcoded config paths → explicit params.** The old CLI hardcoded
    `configs/model.yaml` / `configs/training.yaml` / `configs/corpus.yaml`; those files do not
    exist in the new repo. Training knobs are now explicit CLI flags (R-TRAINCONFIG-SCHEMA: a
    training knob is an explicit construction param, not a WP8 config key) assembled into a plain
    config dict; the corpus NPZ path is `--corpus-npz` or the registry `resolve_corpus_path`.
  * **KILLED-branch flags removed** — `--gpool-sites` / `--head-no-gpool` / `--pool-type` /
    `--pool-attn-dropout` / `--canvas-realness` / `--gpool-bias-active` / `--policy-only-bias`
    (v8 / pma / pma_global / gpool-bias / canvas_realness are all KILLED — F-04/F-05, v8 never
    crosses). The new `build_net` CNN does not carry those knobs.
  * **`pretrain_legacy` raw-JSON corpus fallback KILLED** — 0 config consumers (grep-verified);
    a missing NPZ is now a loud error, not a silent raw-JSON re-scan.
  * Model construction via `build_net(arch_from_spec_and_config(...))` (WP9 authority); events via
    the injected `EventSink`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from mantis.encoding import all_specs as _all_specs
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import resolve_corpus_path as _resolve_corpus_path
from mantis.encoding import resolve_from_checkpoint as _resolve_encoding_from_ckpt
from mantis.encoding.registry import EncodingRegistryError as _EncodingRegistryError
from mantis.encoding.resolvers import MissingEncodingError
from mantis.model import arch_from_spec_and_config, build_net, compile_model
from mantis.train.emit import NullEventSink
from mantis.train.pretrain.dataset import (
    AugmentedBootstrapDataset,
    make_augmented_collate,
)
from mantis.train.pretrain.freeze import _apply_finetune_freeze
from mantis.train.pretrain.trainer import BootstrapTrainer
from mantis.train.pretrain.validate import validate
from mantis.util.device import best_device

_LOG = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap pretrain pipeline (mantis)")
    parser.add_argument("--epochs", type=int, default=5, help="Full passes over the dataset")
    parser.add_argument("--steps", type=int, default=None,
                        help="Hard step budget (overrides epochs; for smoke runs)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile even on CUDA")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a full pretrain checkpoint (restarts the cosine schedule)")
    parser.add_argument("--lr-peak", type=float, default=None,
                        help="Peak LR (cosine restart peak; default 2e-3)")
    parser.add_argument("--inference-out", type=str, default=None,
                        help="Override the bare inference-weights output path")
    parser.add_argument("--eta-min", type=float, default=None,
                        help="Override CosineAnnealingLR eta_min (default 1e-5)")
    _registered = tuple(s.name for s in _all_specs())
    parser.add_argument("--encoding", choices=_registered, default=None,
                        help="Encoding (registry-routed). Registered: " + ", ".join(_registered))
    parser.add_argument("--filters", type=int, default=None, help="Trunk channel count")
    parser.add_argument("--res-blocks", type=int, default=None, help="Trunk depth")
    parser.add_argument("--corpus-npz", type=str, default=None,
                        help="Corpus NPZ path (default: registry resolve_corpus_path)")
    parser.add_argument("--freeze-trunk-entry", action="store_true",
                        help="Freeze trunk.input_conv + trunk.input_gn (staged fine-tune)")
    parser.add_argument("--unfreeze-blocks", type=str, default=None,
                        help="CSV trunk.tower block indices to keep trainable (others freeze)")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--aux-weight", type=float, default=0.15,
                        help="Opponent-reply auxiliary loss weight")
    parser.add_argument("--aux-chain-weight", type=float, default=0.0,
                        help="Q13 chain-length aux head weight (0 disables it)")
    parser.add_argument("--lr", type=float, default=0.002, help="Peak/base LR")
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser


def _resolve_encoding_name(args: argparse.Namespace) -> str:
    """Resolve the encoding: --encoding, else auto-detect from the --resume checkpoint.

    There is no third branch. Pretraining with an unstated encoding silently produced a
    v6 model until R45 (LAW-11, LAW-05); an absent encoding now raises.

    Raises:
        MissingEncodingError: if neither `--encoding` nor `--resume` was given. R45 names
            the convention by ERROR CLASS, and this is that class — so it raises the class
            error rather than `SystemExit`, even though the surrounding function uses
            `SystemExit` for its argument-shaped failures. `pretrain()` converts it to a
            clean CLI message at the boundary, so the operator still sees a message rather
            than a traceback.
    """
    if args.encoding is not None:
        return args.encoding
    if args.resume is not None:
        try:
            spec = _resolve_encoding_from_ckpt(args.resume)
        except _EncodingRegistryError as e:
            raise SystemExit(
                f"--resume {args.resume!r}: could not resolve encoding (no metadata). Pass "
                f"--encoding explicitly. Underlying error: {e}"
            ) from e
        _LOG.info("auto_detected_encoding_from_resume_ckpt name=%s resume=%s", spec.name, args.resume)
        return spec.name
    raise MissingEncodingError(
        "no encoding specified: pass --encoding <name>, or --resume <ckpt> to inherit it "
        "from the checkpoint's metadata. There is no default (LAW-11, R45) — pretraining "
        "silently defaulted to v6 before this was closed."
    )


def pretrain(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args(argv)

    # The class error is the authority (R45); the CLI boundary is the only place it is
    # turned into a message, so an operator who forgot a flag gets one line, not a stack.
    try:
        encoding = _resolve_encoding_name(args)
    except MissingEncodingError as e:
        raise SystemExit(str(e)) from e
    spec = _lookup_encoding(encoding)  # loud raise on an unregistered name

    # Plain config dict (training knobs are explicit params — R-TRAINCONFIG-SCHEMA; no yaml files).
    config: dict = {
        "encoding": encoding,
        "in_channels": int(spec.n_planes),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
    }
    if args.filters is not None:
        config["filters"] = int(args.filters)
    if args.res_blocks is not None:
        config["res_blocks"] = int(args.res_blocks)

    device = best_device()
    _LOG.info("pretrain_device device=%s encoding=%s", device, encoding)

    # ── Corpus (mmap'd NPZ; the raw-JSON fallback is KILLED — 0 config consumers) ──
    npz_path = Path(args.corpus_npz) if args.corpus_npz is not None else Path(_resolve_corpus_path(spec))
    if not npz_path.exists():
        raise SystemExit(
            f"corpus NPZ not found: {npz_path}. Build it (export the corpus NPZ for encoding "
            f"{encoding!r}) or pass --corpus-npz. The legacy raw-JSON corpus fallback is removed "
            "(0 config consumers)."
        )
    data = np.load(npz_path, mmap_mode="r")
    states, policies, outcomes, weights = (
        data["states"], data["policies"], data["outcomes"], data["weights"],
    )
    if len(outcomes) == 0:
        raise SystemExit(f"empty corpus at {npz_path}.")
    _LOG.info("dataset_built n_positions=%d", int(len(outcomes)))

    dataset = AugmentedBootstrapDataset(states, policies, outcomes)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(np.asarray(weights)).double(),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_augmented_collate(augment=True, encoding=encoding),
    )

    # ── Model — the WP9 construction authority ──
    arch = arch_from_spec_and_config(spec, config)
    model = build_net(arch)
    if device.type == "cuda" and not args.no_compile:
        model = compile_model(model, mode="default")

    checkpoint_dir = Path(args.checkpoint_dir)
    step_budget = args.steps
    total_pretrain_steps = step_budget if step_budget is not None else args.epochs * len(loader)
    config["pretrain_total_steps"] = total_pretrain_steps
    if args.eta_min is not None:
        config["pretrain_eta_min"] = float(args.eta_min)

    trainer = BootstrapTrainer(
        model, config, device, checkpoint_dir, arch=arch, sink=NullEventSink(),
    )

    if args.resume:
        _resume_into(trainer, args, total_pretrain_steps)

    if args.freeze_trunk_entry or args.unfreeze_blocks is not None:
        unfreeze_set: set | None = None
        if args.unfreeze_blocks is not None:
            unfreeze_set = {int(s) for s in args.unfreeze_blocks.split(",") if s.strip()}
        report = _apply_finetune_freeze(
            getattr(trainer.model, "_orig_mod", trainer.model),
            freeze_trunk_entry=args.freeze_trunk_entry,
            unfreeze_blocks=unfreeze_set,
        )
        _LOG.info("finetune_freeze_applied %s", report)

    trainer.step = -total_pretrain_steps
    start_step = trainer.step
    chain_weight = float(args.aux_chain_weight)
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(
            loader,
            label_smoothing=float(args.label_smoothing),
            aux_weight=float(args.aux_weight),
            chain_weight=chain_weight,
            step_budget=step_budget,
            start_step=start_step,
        )
        _LOG.info("epoch_complete epoch=%d %s", epoch, {k: round(v, 4) for k, v in metrics.items()})
        if step_budget is not None and (trainer.step - start_step) >= step_budget:
            break

    inf_out = Path(args.inference_out) if args.inference_out else None
    ckpt_path = trainer.save_checkpoint(inf_out=inf_out)
    _LOG.info("pretrain_checkpoint path=%s", str(ckpt_path))
    validate(ckpt_path, device)


def _resume_into(trainer: BootstrapTrainer, args: argparse.Namespace, total_steps: int) -> None:
    """Resume model/optimizer/scaler from a full pretrain checkpoint; restart the cosine schedule
    across the new window at the requested peak LR (weights-only source → optimizer/scaler reset)."""
    resume_path = Path(args.resume)
    resume_ckpt = torch.load(resume_path, map_location=trainer.device, weights_only=True)
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    weights_only_src = isinstance(resume_ckpt, dict) and "model_state" not in resume_ckpt
    if weights_only_src:
        base.load_state_dict(resume_ckpt)
    else:
        base.load_state_dict(resume_ckpt["model_state"])
        trainer.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if resume_ckpt.get("scaler_state") is not None:
            trainer.scaler.load_state_dict(resume_ckpt["scaler_state"])
    new_peak = float(args.lr_peak) if args.lr_peak is not None else float(trainer.config.get("lr", 0.002))
    new_eta_min = float(args.eta_min) if args.eta_min is not None else 1e-5
    for g in trainer.optimizer.param_groups:
        g["lr"] = new_peak
        g["initial_lr"] = new_peak
    trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=max(1, total_steps), eta_min=new_eta_min,
    )
    _LOG.info("resume_complete new_peak_lr=%s cosine_t_max=%s weights_only=%s",
              new_peak, total_steps, weights_only_src)


if __name__ == "__main__":
    pretrain()
