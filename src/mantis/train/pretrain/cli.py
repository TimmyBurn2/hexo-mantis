# >300 justify (R8). ONE entry point over ONE linear act — parse, resolve the encoding, read
# the training terms from the config, load the corpus, build the net, train, save, validate.
# Splitting it forks the argv namespace and the assembled config dict into two modules that
# must agree about every term, which is the duplicate-authority shape F-816-25 was filed for.
"""Bootstrap pretrain CLI (WP10 §a.7 IMPROVE of `bootstrap/pretrain_cli.py`).

The `python -m mantis.train.pretrain` entry: argparse surface + config resolution + corpus load +
model build + train/save/validate orchestration.

Ratified WP10 amendments over a pure relocation:
  * **Personal/hardcoded config paths → explicit params.** The old CLI hardcoded
    `configs/model.yaml` / `configs/training.yaml` / `configs/corpus.yaml`; those files do not
    exist in the new repo. The corpus NPZ path is `--corpus-npz` or the registry
    `resolve_corpus_path`.
  * **KILLED-branch flags removed** — `--gpool-sites` / `--head-no-gpool` / `--pool-type` /
    `--pool-attn-dropout` / `--canvas-realness` / `--gpool-bias-active` / `--policy-only-bias`
    (v8 / pma / pma_global / gpool-bias / canvas_realness are all KILLED — F-04/F-05, v8 never
    crosses). The new `build_net` CNN does not carry those knobs.
  * **`pretrain_legacy` raw-JSON corpus fallback KILLED** — 0 config consumers (grep-verified);
    a missing NPZ is now a loud error, not a silent raw-JSON re-scan.
  * Model construction via `build_net(arch_from_spec_and_config(...))` (WP9 authority); events via
    the injected `EventSink`.

F-816-25 / R296(b): the five flags that shadowed minted `train.*` keys are GONE and `--config`
is REQUIRED in their place — see `training_terms`, the one read path. R-TRAINCONFIG-SCHEMA, the
old ground for a CLI-side training knob, is DEAD: that schema extension landed (WPSC SC-A1) and
all six are live `train.*` leaves today.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from mantis.config import TrainConfig, load_config
from mantis.encoding import all_specs as _all_specs
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import resolve_corpus_path as _resolve_corpus_path
from mantis.encoding import resolve_from_checkpoint as _resolve_encoding_from_ckpt
from mantis.encoding.registry import EncodingRegistryError as _EncodingRegistryError
from mantis.encoding.resolvers import MissingEncodingError
from mantis.model import HexTacToeNet, arch_from_spec_and_config, build_net, compile_model
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

#: The dense arm's label smoothing. The parser's default is `None` so "was it supplied?" is
#: answerable — the graph route refuses flags it would ignore, and it cannot refuse a value it
#: cannot distinguish from a default. THIS is the one default authority for the term.
DEFAULT_LABEL_SMOOTHING = 0.05


def _build_arg_parser() -> argparse.ArgumentParser:
    # `allow_abbrev=False` is LOAD-BEARING, not tidiness (found by this fix's own oracle).
    # With argparse's default prefix matching, the DELETED `--lr` is an unambiguous abbreviation
    # of the surviving `--lr-peak`, so an old command line `--lr 0.002` would silently set the
    # cosine restart peak instead of erroring — a deleted shadow re-entering as a different
    # knob, which is worse than the shadow this fix removed.
    parser = argparse.ArgumentParser(
        description="Bootstrap pretrain pipeline (mantis)", allow_abbrev=False
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Run config (schema-validated). THE authority for lr, weight_decay, "
                             "batch_size, aux_opp_reply_weight, aux_chain_weight and eta_min — "
                             "this CLI states none of them (F-816-25, R296(b)/R79).")
    parser.add_argument("--epochs", type=int, default=5, help="Full passes over the dataset")
    parser.add_argument("--steps", type=int, default=None,
                        help="Hard step budget (overrides epochs; for smoke runs)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile even on CUDA")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a full pretrain checkpoint (restarts the cosine schedule)")
    parser.add_argument("--lr-peak", type=float, default=None,
                        help="Peak LR for a --resume cosine restart; absent = the config's train.lr")
    parser.add_argument("--inference-out", type=str, default=None,
                        help="Override the bare inference-weights output path")
    parser.add_argument("--eta-min", type=float, default=None,
                        help="Override CosineAnnealingLR eta_min; absent = the config's train.eta_min")
    _registered = tuple(s.name for s in _all_specs())
    parser.add_argument("--encoding", choices=_registered, default=None,
                        help="Encoding (registry-routed). Registered: " + ", ".join(_registered))
    parser.add_argument("--filters", type=int, default=None, help="Trunk channel count")
    parser.add_argument("--res-blocks", type=int, default=None, help="Trunk depth")
    parser.add_argument("--corpus-npz", type=str, default=None,
                        help="Corpus NPZ path (default: registry resolve_corpus_path)")
    # ── the held-out stopping rule (R328(d)) — ALL-OR-NONE, like the split that feeds it ──
    parser.add_argument("--heldout-hexg", type=str, default=None,
                        help="held-out .hexg ring; enables the held-out policy-loss stop")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="training steps between held-out evaluations")
    parser.add_argument("--patience", type=int, default=None,
                        help="held-out evaluations without improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=None,
                        help="improvement a held-out reading must beat; REFUSED if below the "
                             "estimator's own measured noise")
    parser.add_argument("--corpus-hexg", type=str, default=None,
                        help="Corpus .hexg ring for the GRAPH route (default: registry "
                             "resolve_corpus_path). Ignored on the dense route.")
    parser.add_argument("--freeze-trunk-entry", action="store_true",
                        help="Freeze trunk.input_conv + trunk.input_gn (staged fine-tune)")
    parser.add_argument("--unfreeze-blocks", type=str, default=None,
                        help="CSV trunk.tower block indices to keep trainable (others freeze)")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help=f"Dense-arm label smoothing (default: {DEFAULT_LABEL_SMOOTHING})")
    return parser


#: The `train.*` leaves this CLI is NOT allowed to have an opinion about (F-816-25, R296(b)).
#: Named as data rather than spelled out in the reader below so the oracle can assert the SET,
#: not a hand-listed copy of it that would stay green while a seventh shadow was added.
SHADOWED_TRAIN_KEYS: tuple[str, ...] = (
    "lr", "weight_decay", "batch_size", "aux_opp_reply_weight", "aux_chain_weight", "eta_min",
)


def training_terms(train_cfg: TrainConfig) -> dict[str, float | int]:
    """The training terms a bootstrap pretrain runs on, read from the minted config.

    THE ONE READ PATH, and that is the whole of F-816-25's fix. Each of these six was a
    code-side literal on the argparse surface — three of them DIVERGENT from `configs/run5.yaml`
    — so a bootstrap pretrain ran on the parser's numbers while the minted ones sat inert. Two
    authorities over one number is R79; here there is one, and it is the config.

    `pretrain_eta_min` is the one renamed key: `BootstrapTrainer` reads that name, and the
    rename happens HERE rather than at the trainer so the mapping is visible at the seam that
    performs it.

    Raises:
        AttributeError: if `train_cfg` lacks a key this reads. Not defended against — a
            validated `TrainConfig` cannot, and a caller passing something else is a defect
            that should surface by name rather than as a silently-defaulted number.
    """
    return {
        "lr": float(train_cfg.lr),
        "weight_decay": float(train_cfg.weight_decay),
        "batch_size": int(train_cfg.batch_size),
        "aux_opp_reply_weight": float(train_cfg.aux_opp_reply_weight),
        "aux_chain_weight": float(train_cfg.aux_chain_weight),
        "pretrain_eta_min": float(train_cfg.eta_min),
    }


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

    run_config = load_config(args.config)
    train_cfg = run_config.train
    config: dict = {
        "encoding": encoding,
        "in_channels": int(spec.n_planes),
        **training_terms(train_cfg),
    }
    if args.filters is not None:
        config["filters"] = int(args.filters)
    if args.res_blocks is not None:
        config["res_blocks"] = int(args.res_blocks)

    device = best_device()
    _LOG.info("pretrain_device device=%s encoding=%s", device, encoding)

    # The GRAPH arm is a REROUTE, not a second pipeline: it hands a loaded `.hexg` ring to the
    # SAME declared train-step seam the self-play loop uses (R325(c)). Everything below this
    # branch — the NPZ reader, the augmented dense collate, `BootstrapTrainer` — is the dense
    # arm and stays dense. `--no-compile` has no subject here: production does not compile the
    # graph net, so this route does not either.
    if getattr(spec, "representation", None) == "graph":
        from mantis.train.pretrain.graph_route import GraphPretrainError, run_graph_pretrain

        ring_path = (Path(args.corpus_hexg) if args.corpus_hexg is not None
                     else Path(_resolve_corpus_path(spec)))
        # ALL-OR-NONE, the same shape `--split-*` takes in the encoder and for the same reason:
        # a patience without a ring, or a ring without a cadence, is a stopping rule nobody
        # declared. Silence is the pre-existing behaviour — budget-bound, no monitor.
        _stop_flags = (args.heldout_hexg, args.eval_every, args.patience, args.min_delta)
        if any(f is not None for f in _stop_flags) and not all(f is not None for f in _stop_flags):
            raise SystemExit(
                "--heldout-hexg, --eval-every, --patience and --min-delta are all-or-none: a "
                "partially specified stopping rule is one nobody declared (R328(d))."
            )
        monitor = None
        if args.heldout_hexg is not None:
            from mantis.config.resolve.coordinator import resolve_coordinator_knobs
            from mantis.config.resolve.microbatch import resolve_microbatch_caps
            from mantis.config.resolve.sample_threads import resolve_sample_threads
            from mantis.train.pretrain.graph_route import load_ring
            from mantis.train.pretrain.heldout import HeldOutMonitor

            _full = run_config.model_dump()
            _ho_buf, _ho_prov = load_ring(Path(args.heldout_hexg), encoding=spec.name)
            if _ho_prov.get("split_part") != "heldout":
                raise SystemExit(
                    f"--heldout-hexg names a ring whose provenance says split_part="
                    f"{_ho_prov.get('split_part')!r}, not 'heldout'. A held-out loss measured "
                    "over the TRAINING ring falls forever and every other check still passes; "
                    "the sidecar is what makes that unrepresentable (R328(d), PB-8)."
                )
            monitor = HeldOutMonitor.build(
                ring=_ho_buf, spec=spec, plies=int(_ho_prov["plies"]),
                batch_size=resolve_coordinator_knobs(train_cfg).batch_size,
                eval_every=args.eval_every, patience=args.patience, min_delta=args.min_delta,
                caps_provider=lambda: resolve_microbatch_caps(_full),
                sample_threads_provider=lambda: resolve_sample_threads(_full),
            )
        try:
            written = run_graph_pretrain(
                spec=spec, full_config=run_config.model_dump(), train_section=train_cfg,
                ring_path=ring_path, checkpoint_dir=Path(args.checkpoint_dir), device=device,
                steps=args.steps, epochs=args.epochs, monitor=monitor,
                dense_arm_flags={
                    "--filters": args.filters, "--res-blocks": args.res_blocks,
                    "--resume": args.resume, "--lr-peak": args.lr_peak,
                    "--eta-min": args.eta_min,
                    "--freeze-trunk-entry": args.freeze_trunk_entry,
                    "--unfreeze-blocks": args.unfreeze_blocks,
                    "--inference-out": args.inference_out,
                    "--label-smoothing": args.label_smoothing,
                },
            )
        except GraphPretrainError as e:
            raise SystemExit(str(e)) from e
        _LOG.info("pretrain_complete route=graph checkpoint=%s", written)
        return

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
        # torch stub gap: WeightedRandomSampler is annotated Sequence[float] but is
        # documented to take (and internally as_tensor()s) a Tensor.
        weights=torch.from_numpy(np.asarray(weights)).double(),  # pyright: ignore[reportArgumentType]
        num_samples=len(dataset),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_augmented_collate(augment=True, encoding=encoding),
    )

    # ── Model — the WP9 construction authority ──
    arch = arch_from_spec_and_config(spec, config)
    model = build_net(arch)
    if not isinstance(model, HexTacToeNet):
        # The bootstrap-pretrain pipeline is dense-only (dense corpus NPZ, dense collate,
        # HexTacToeNet trainer); a graph encoding has no pretrain path through this CLI.
        raise SystemExit(
            f"pretrain CLI: encoding {encoding!r} built a {type(model).__name__} — the "
            "bootstrap-pretrain pipeline is dense-only (no graph corpus route)."
        )
    if device.type == "cuda" and not args.no_compile:
        model = compile_model(model, mode="default")

    checkpoint_dir = Path(args.checkpoint_dir)
    step_budget = args.steps
    total_pretrain_steps = step_budget if step_budget is not None else args.epochs * len(loader)
    config["pretrain_total_steps"] = total_pretrain_steps
    if args.eta_min is not None:
        config["pretrain_eta_min"] = float(args.eta_min)  # an EXPLICIT operator override

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
    chain_weight = float(config["aux_chain_weight"])
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(
            loader,
            label_smoothing=(DEFAULT_LABEL_SMOOTHING if args.label_smoothing is None
                             else float(args.label_smoothing)),
            aux_weight=float(config["aux_opp_reply_weight"]),
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
    new_peak = float(args.lr_peak) if args.lr_peak is not None else float(trainer.config["lr"])
    new_eta_min = (
        float(args.eta_min) if args.eta_min is not None
        else float(trainer.config["pretrain_eta_min"])
    )
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
