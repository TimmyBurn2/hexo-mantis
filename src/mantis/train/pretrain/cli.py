"""Bootstrap pretrain CLI: the `python -m mantis.train.pretrain` entry.

Every `train.*` term comes from the REQUIRED `--config`; the CLI states none of them, and a flag
it does not declare is refused by argparse rather than parsed and ignored.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mantis.config import load_config
from mantis.encoding import all_specs as _all_specs
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import resolve_corpus_path as _resolve_corpus_path
from mantis.encoding.resolvers import MissingEncodingError
from mantis.monitor.logging_setup import configure_logging
from mantis.util.device import best_device

_LOG = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    # `allow_abbrev=False` is LOAD-BEARING: a deleted flag must not parse as a live flag's prefix.
    parser = argparse.ArgumentParser(
        description="Bootstrap pretrain pipeline (mantis)", allow_abbrev=False
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Run config (schema-validated): THE authority for every train.* "
                             "term; this CLI states none of them.")
    parser.add_argument("--epochs", type=int, default=5, help="Full passes over the dataset")
    parser.add_argument("--steps", type=int, default=None,
                        help="Hard step budget (overrides epochs; for smoke runs)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    _registered = tuple(s.name for s in _all_specs())
    parser.add_argument("--encoding", choices=_registered, default=None,
                        help="Encoding (registry-routed). Registered: " + ", ".join(_registered))
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
                        help="Corpus .hexg ring (default: registry resolve_corpus_path)")
    return parser


def _resolve_encoding_name(args: argparse.Namespace) -> str:
    """Resolve the encoding from `--encoding`; there is no second branch and no default (LAW-11).

    Raises:
        MissingEncodingError: `--encoding` was not given, raised as the class error rather than
            `SystemExit`, which `pretrain()` converts at the boundary.
    """
    if args.encoding is not None:
        return args.encoding
    raise MissingEncodingError(
        "no encoding specified: pass --encoding <name>. There is no default (LAW-11, R45) — "
        "pretraining silently defaulted to v6 before this was closed."
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
