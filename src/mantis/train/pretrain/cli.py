# >300 justify (R8). ONE entry point over ONE linear act — parse, resolve the encoding, read the
# training terms from the config, load the corpus, build the net, train, save, validate. A split
# forks the argv namespace and the assembled config dict into two modules that must agree about
# every term, which is the duplicate-authority shape this file was rewritten to end.
"""Bootstrap pretrain CLI: the `python -m mantis.train.pretrain` entry.

Argparse surface, config resolution, corpus load, model build, train/save/validate. Config paths
are explicit parameters rather than hardcoded files, a missing corpus is a loud error rather than
a silent raw-JSON re-scan, and the five flags that shadowed minted `train.*` keys are GONE with
`--config` REQUIRED in their place — see `training_terms`, the one read path.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mantis.config import TrainConfig, load_config
from mantis.encoding import all_specs as _all_specs
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import resolve_corpus_path as _resolve_corpus_path
from mantis.encoding import resolve_from_checkpoint as _resolve_encoding_from_ckpt
from mantis.encoding.registry import EncodingRegistryError as _EncodingRegistryError
from mantis.encoding.resolvers import MissingEncodingError
from mantis.monitor.logging_setup import configure_logging
from mantis.util.device import best_device

_LOG = logging.getLogger(__name__)

#: The dense arm's label smoothing and the ONE default authority for it; the parser's own default
#: is `None` so "was it supplied?" stays answerable, since the graph route refuses flags it would
#: ignore and cannot refuse a value it cannot tell from a default.
DEFAULT_LABEL_SMOOTHING = 0.05


def _build_arg_parser() -> argparse.ArgumentParser:
    # `allow_abbrev=False` is LOAD-BEARING: with prefix matching, the DELETED `--lr` is an
    # unambiguous abbreviation of `--lr-peak`, so `--lr 0.002` would silently set the cosine
    # restart peak instead of erroring.
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
    # the held-out stopping rule — ALL-OR-NONE, like the split that feeds it
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


#: The `train.*` leaves this CLI may not have an opinion about. Named as data so the oracle can
#: assert the SET, not a hand-listed copy that stays green while a seventh shadow is added.
SHADOWED_TRAIN_KEYS: tuple[str, ...] = ("lr", "weight_decay", "batch_size", "eta_min")


def training_terms(train_cfg: TrainConfig) -> dict[str, float | int | bool | str]:
    """The training terms a bootstrap pretrain runs on, read from the minted config.

    THE ONE READ PATH: each term was a code-side literal on the argparse surface, three of them
    DIVERGENT from the shipped config, so a pretrain ran on the parser's numbers while the minted
    ones sat inert.

    Raises:
        AttributeError: `train_cfg` lacks a key this reads — undefended, since a validated
            `TrainConfig` cannot and any other caller is a defect that should surface by name.
    """
    return {
        "lr": float(train_cfg.lr),
        "weight_decay": float(train_cfg.weight_decay),
        "batch_size": int(train_cfg.batch_size),
        "pretrain_eta_min": float(train_cfg.eta_min),
    }


def _resolve_encoding_name(args: argparse.Namespace) -> str:
    """Resolve the encoding: --encoding, else auto-detect from the --resume checkpoint. There is
    no third branch — an unstated encoding silently produced a v6 model until LAW-11.

    Raises:
        MissingEncodingError: neither `--encoding` nor `--resume` was given, raised as the class
            error rather than `SystemExit`, which `pretrain()` converts at the boundary.
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
    # THE one mantis handler, not `basicConfig`: `configure_logging` is idempotent, so a
    # repeated entry cannot double every line.
    configure_logging()
    args = _build_arg_parser().parse_args(argv)

    # The class error is the authority; the CLI boundary is the only place it becomes a message.
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
    # SAME declared train-step seam the self-play loop uses.
    if getattr(spec, "representation", None) == "graph":
        from mantis.train.pretrain.graph_route import GraphPretrainError, run_graph_pretrain

        ring_path = (Path(args.corpus_hexg) if args.corpus_hexg is not None
                     else Path(_resolve_corpus_path(spec)))
        # ALL-OR-NONE: a patience without a ring, or a ring without a cadence, is a stopping
        # rule nobody declared. Silence is the pre-existing behaviour — budget-bound.
        _stop_flags = (args.heldout_hexg, args.eval_every, args.patience, args.min_delta)
        if any(f is not None for f in _stop_flags) and not all(f is not None for f in _stop_flags):
            raise SystemExit(
                "--heldout-hexg, --eval-every, --patience and --min-delta are all-or-none: a "
                "partially specified stopping rule is one nobody declared (R328(d))."
            )
        monitor = None
        if args.heldout_hexg is not None:
            from mantis.config.resolve.coordinator import resolve_coordinator_knobs
            from mantis.config.resolve.fast_policy_weight import resolve_fast_policy_weight
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
                fast_policy_weight_provider=lambda: resolve_fast_policy_weight(_full),
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
    raise SystemExit(
        f"pretrain CLI: encoding {encoding!r} declares representation "
        f"{getattr(spec, 'representation', None)!r}; the only pretrain route is the graph "
        "reroute (the dense NPZ pipeline went with the grid path, R346(f))."
    )

if __name__ == "__main__":
    pretrain()
