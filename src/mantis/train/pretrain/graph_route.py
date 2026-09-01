# >300 justify (R8): the ring's provenance handshake, the step budget it feeds, the dense-arm
# refusal and the loop those three constrain are ONE reroute over ONE artifact. The budget is a
# function of the ring's own ply count and the refusal exists because the dense arm's flags mean
# something different here — split them and a caller can reach the loop with a ring whose
# provenance was checked somewhere else, which is the exact class the sidecar handshake exists
# to close.
"""BC pretrain on the GRAPH arch — a REROUTE through the declared train-step seam.

WHY A REROUTE AND NOT A SECOND TRAINER. `train/pretrain/`'s dense arm is a dense NPZ reader,
a dense collate and a `HexTacToeNet` trainer. Graph-ifying `BootstrapTrainer` would build a
SECOND graph training path beside the one that trains run5, with its own collate, its own
micro-batch split and its own drift surface. The graph training step already exists — it is
`mantis.train.coordinator.dispatch.run_declared_train_step`, the same declared route the
straight self-play loop takes — and the ring the R247 corpus encodes to
(`mantis.data.bootstrap_encode`) is the trainer's own `.hexg`. So this module supplies the two
things the seam needs (a loaded ring and the providers) and calls it. Nothing here reimplements
sampling, collation, micro-batching or the gradient step.

IT IS CAPABILITY, NOT POSTURE (R325(c)). No config selects this route, no resolver arms it, and
no production path reaches it: it runs only when an operator invokes the pretrain CLI with a
`--config` whose declared representation is `graph`. Execution of any pretrain still waits on
the operator's bootstrap posture word and the recipe rows that follow it (R119: no armed value
is authored here — every training term is read from the minted config).

THE RING'S GEOMETRY IS READ FROM ITS PROVENANCE SIDECAR, NEVER GUESSED. `HexgBuffer` takes its
capacity and visit capacity at construction, before a single record is read, and the producer
records both in the sidecar it writes beside the artifact. A guessed capacity smaller than the
corpus silently drops the head of the ring — a wrong training set that every downstream check
would pass. An absent or disagreeing sidecar is therefore a REFUSAL, not a fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.config.resolve.sample_threads import resolve_sample_threads
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import resolve_corpus_sha_pin
from mantis.model import arch_from_spec_and_config, build_net
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.emit import NullEventSink
from mantis.train.trainer.core import Trainer

_LOG = logging.getLogger(__name__)

#: The BC ring carries no time ordering, so it has no "recent window" for the engine's
#: `recent_frac` to mean anything over. Structural, not a tuning choice — and NOT read from
#: `train.recency_weight`, which is the self-play loop's knob over a loop-written ring.
BC_RECENCY_WEIGHT = 0.0


class GraphPretrainError(RuntimeError):
    """The graph BC-pretrain route cannot run as declared: a missing ring, a missing or
    disagreeing provenance sidecar, an empty ring, or a step budget that resolves to zero."""


def read_ring_provenance(ring_path: Path, *, encoding: str) -> dict[str, Any]:
    """Return the provenance sidecar written beside `ring_path`, checked against `encoding`.

    Args:
        ring_path: the `.hexg` artifact.
        encoding: the declared encoding the run resolved.

    Returns:
        The parsed provenance mapping.

    Raises:
        GraphPretrainError: the sidecar is absent, unparseable, missing a geometry key, or
            names a different encoding than the one declared.
    """
    sidecar = ring_path.with_name(ring_path.name + ".provenance.json")
    if not sidecar.is_file():
        raise GraphPretrainError(
            f"{ring_path}: no provenance sidecar at {sidecar}. The ring's capacity and visit "
            "capacity are recorded there and are needed to reconstruct the buffer; a guessed "
            "capacity silently drops the head of the corpus. Re-produce the ring with "
            "`python -m mantis.data.bootstrap_encode`."
        )
    try:
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GraphPretrainError(f"{sidecar}: unparseable provenance sidecar: {e}") from e
    for key in ("encoding", "ring_capacity", "ring_visit_capacity", "plies"):
        if key not in prov:
            raise GraphPretrainError(f"{sidecar}: provenance is missing {key!r}.")
    if prov["encoding"] != encoding:
        raise GraphPretrainError(
            f"{sidecar}: the ring was encoded for {prov['encoding']!r} but the run declares "
            f"{encoding!r}. A ring is bound to its encoding's geometry; these are different "
            "training sets, not the same one under two names."
        )
    return prov


def _assert_launch_pin(ring_path: Path, *, encoding: str) -> None:
    """Refuse a corpus that is not the encoding's launch-pinned bytes, when one is pinned.

    RE-HOMED HERE BY R327(e). The pin registry outlived the dense corpus-mix loader that used
    to read it: a launch pin says two hosts must train on the byte-identical corpus, and BC
    pretrain is the surviving path that trains on one. `resolve_corpus_sha_pin` returns `None`
    when the encoding registers no pin, and `None` means NOT ENFORCED — the documented
    contract, and the reason the stream below is conditional rather than unconditional.

    Distinct from `read_ring_provenance`'s checks, which read the SIDECAR: a sidecar can be
    rewritten beside a swapped ring, so the pin is taken over the artifact's own bytes.

    Raises:
        GraphPretrainError: a pin is registered for `encoding` and the ring does not match it.
    """
    pin = resolve_corpus_sha_pin(_lookup_encoding(encoding))
    if pin is None:
        return
    digest = hashlib.sha256()
    with ring_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != pin:
        raise GraphPretrainError(
            f"{ring_path}: sha256 {actual[:12]}… is not the launch-pinned corpus for "
            f"{encoding!r} ({pin[:12]}…). Both hosts must read the byte-identical launch "
            "corpus; sync it rather than re-exporting one that happens to load."
        )


def load_ring(ring_path: Path, *, encoding: str) -> tuple[Any, dict[str, Any]]:
    """Reconstruct the `.hexg` ring at its recorded geometry and load it.

    Args:
        ring_path: the `.hexg` artifact.
        encoding: the declared encoding.

    Returns:
        `(buffer, provenance)` — the loaded `HexgBuffer` and its sidecar.

    Raises:
        GraphPretrainError: the ring is absent, its sidecar fails `read_ring_provenance`, or
            it loads zero records.
    """
    from mantis._engine import HexgBuffer  # noqa: PLC0415 — extension only available post-build

    if not ring_path.is_file():
        raise GraphPretrainError(
            f"corpus ring not found: {ring_path}. Produce it with "
            "`python -m mantis.data.bootstrap_encode`, or pass --corpus-hexg."
        )
    prov = read_ring_provenance(ring_path, encoding=encoding)
    _assert_launch_pin(ring_path, encoding=encoding)
    buf = HexgBuffer(int(prov["ring_capacity"]), encoding, int(prov["ring_visit_capacity"]))
    loaded = buf.load_from_path(str(ring_path))
    if loaded == 0:
        raise GraphPretrainError(f"{ring_path}: loaded 0 records — an empty corpus is not a run.")
    _LOG.info("bc_ring_loaded path=%s records=%d games=%s", ring_path, loaded, prov.get("games"))
    return buf, prov


def resolve_step_budget(
    ring_size: int, *, batch_size: int, steps: int | None, epochs: int
) -> int:
    """The BC step budget, on the dense arm's own convention.

    The dense arm computes `epochs * len(loader)` over a with-replacement sampler; the graph
    sampler is also with-replacement, so an "epoch" is the same nominal object on both arms and
    is derived the same way. An explicit `--steps` overrides, as it does on the dense arm.

    Raises:
        GraphPretrainError: the budget resolves to zero or fewer steps.
    """
    if steps is not None:
        total = int(steps)
    else:
        total = int(epochs) * max(1, math.ceil(ring_size / max(1, batch_size)))
    if total <= 0:
        raise GraphPretrainError(
            f"step budget resolves to {total}; nothing would be trained. Pass --steps > 0."
        )
    return total


#: Why each of these has no subject on the graph route. Data, not prose, so the refusal message
#: and the oracle read the SAME set — a hand-listed copy in either would drift.
DENSE_ARM_FLAGS: dict[str, str] = {
    "--filters": "a CNN trunk width; the graph arch reads `gnn_hidden` in the config",
    "--res-blocks": "a CNN trunk depth; the graph arch reads `gnn_num_layers` in the config",
    "--resume": "the dense arm's `BootstrapTrainer` resume; this route builds a fresh net",
    "--lr-peak": "a cosine-restart peak for the dense `--resume` path",
    "--eta-min": "read as `pretrain_eta_min` by `BootstrapTrainer`; the graph step takes its "
                 "schedule from the trainer the config builds",
    "--freeze-trunk-entry": "freezes `trunk.input_conv`/`trunk.input_gn`, CNN modules",
    "--unfreeze-blocks": "selects `trunk.tower` block indices, CNN modules",
    "--inference-out": "the dense arm's bare-weights export path",
    "--label-smoothing": "a `BootstrapTrainer.train_epoch` term; the graph loss reads its "
                         "terms from the config",
}


def refuse_dense_arm_flags(supplied: dict[str, Any]) -> None:
    """Refuse any dense-arm CLI flag that the graph route would silently ignore.

    A flag that reads as though it set a width, a schedule or a freeze and in fact sets
    NOTHING is the shadow class F-816-25 was filed for. Every flag here is refused rather than
    ignored; the ones with a real subject on this route (`--config`, `--steps`, `--epochs`,
    `--checkpoint-dir`, `--encoding`, `--corpus-hexg`) are absent from the set. `--no-compile`
    is deliberately NOT here: production does not compile the graph net either, so the flag
    asks for the behaviour this route already has.

    Args:
        supplied: flag name → the parsed value. A value of `None` or `False` is "not supplied".

    Raises:
        GraphPretrainError: any supplied flag has no subject on the graph route.
    """
    unknown = set(supplied) - set(DENSE_ARM_FLAGS)
    if unknown:
        raise GraphPretrainError(
            f"refuse_dense_arm_flags called with {sorted(unknown)}, which are not in "
            "DENSE_ARM_FLAGS. The set is the authority; add the flag and its reason there."
        )
    named = [n for n, v in supplied.items() if v is not None and v is not False]
    if named:
        reasons = "; ".join(f"{n} is {DENSE_ARM_FLAGS[n]}" for n in sorted(named))
        raise GraphPretrainError(
            f"the graph BC route does not read {', '.join(sorted(named))} — {reasons}. "
            "Refused rather than ignored: a flag that silently sets nothing is worse than no "
            "flag at all."
        )


def run_graph_pretrain(
    *, spec: Any, full_config: dict[str, Any], train_section: Any, ring_path: Path,
    checkpoint_dir: Path, device: Any, steps: int | None, epochs: int,
    dense_arm_flags: dict[str, Any], monitor: Any | None = None,
) -> Path:
    """Run a BC pretrain on the graph arch and return the written checkpoint's path.

    Every training term arrives from the minted config through its existing resolver: the
    coordinator knobs author `batch_size` and `augment`, `resolve_microbatch_caps` authors the
    graph micro-batch caps and `resolve_sample_threads` the ring-rebuild width. This function
    authors none of them (R119), and the two providers are passed as CALLABLES for the reason
    the dispatcher's docstring gives — they read graph-only config sections.

    Args:
        spec: the resolved `EncodingSpec` (representation `graph`).
        full_config: `RunConfig.model_dump()` — what the seam and the trainer read.
        train_section: the typed `train` section, for the coordinator-knob resolver.
        ring_path: the `.hexg` corpus ring.
        checkpoint_dir: where the envelope-v2 checkpoint is written.
        device: the torch device.
        steps: explicit step budget, or None to derive from `epochs`.
        epochs: nominal passes, used only when `steps` is None.
        dense_arm_flags: the CLI's dense-arm flag values, refused here if any was supplied.
            REQUIRED and undefaulted — a caller that omits it would silently skip the refusal,
            which is the same shape as the flags it exists to catch.
        monitor: a `heldout.HeldOutMonitor`, or None for no held-out monitoring. `None`
            DEFAULTS here and nowhere else, because the budget alone is still a valid bound
            and the pre-existing behaviour is exactly that; what a caller cannot do is ask for
            a stopping rule and silently get none, since asking means passing one.

    Returns:
        The path of the checkpoint written by `Trainer.save_checkpoint`.

    Raises:
        GraphPretrainError: a dense-arm flag was supplied, the ring or its budget refuses
            (see `refuse_dense_arm_flags`, `load_ring`, `resolve_step_budget`), or the
            held-out estimator's measured noise exceeds the monitor's `min_delta`.
    """
    refuse_dense_arm_flags(dense_arm_flags)
    buf, prov = load_ring(ring_path, encoding=spec.name)
    knobs = resolve_coordinator_knobs(train_section)
    total_steps = resolve_step_budget(
        int(prov["plies"]), batch_size=knobs.batch_size, steps=steps, epochs=epochs
    )

    # The NESTED dump, which is what `Trainer._derive_arch` and `train.orchestrator` pass.
    # The CLI's flat term dict is the DENSE arm's shape and would resolve a different arch the
    # day a `gnn_*` width key is minted.
    arch = arch_from_spec_and_config(spec, full_config)
    model = build_net(arch)
    trainer = Trainer(
        model, full_config, arch=arch, checkpoint_dir=checkpoint_dir, device=device,
        sink=NullEventSink(),
    )

    def _caps() -> Any:
        return resolve_microbatch_caps(full_config)

    def _threads() -> int:
        return resolve_sample_threads(full_config)

    _LOG.info(
        "bc_graph_pretrain_start steps=%d batch_size=%d augment=%s ring_records=%s",
        total_steps, knobs.batch_size, knobs.augment, prov["plies"],
    )
    if monitor is not None:
        # THE ESTIMATOR'S OWN NOISE, MEASURED BEFORE THE FIRST OPTIMIZER STEP. Any difference
        # between two readings here is the sampler's, because nothing moved in between. A
        # patience rule whose `min_delta` sits inside that spread cannot distinguish progress
        # from resampling, and would stop on noise or never stop at all.
        noise = monitor.measure_noise(trainer)
        if monitor.stop.min_delta < noise:
            raise GraphPretrainError(
                f"the held-out estimator's spread on an UNCHANGED model is {noise:.6g}, which "
                f"is WIDER than the configured min_delta {monitor.stop.min_delta:.6g}. The "
                "stopping rule would be reading sampling noise as progress. Raise min_delta "
                "above the measured spread, or widen the held-out pass."
            )
    loss_info: dict[str, float] = {}
    steps_run = 0
    stopped_early = False
    for _ in range(total_steps):
        loss_info = run_declared_train_step(
            trainer, buf, spec,
            batch_size=knobs.batch_size, augment=knobs.augment,
            recency_weight=BC_RECENCY_WEIGHT, recent_buffer=None,
            caps_provider=_caps, sample_threads_provider=_threads,
        )
        steps_run += 1
        if monitor is not None:
            should_stop, _loss = monitor.maybe_evaluate(trainer, step=steps_run)
            if should_stop:
                stopped_early = True
                break
    if monitor is not None:
        # LAW-18: the lever under test reports its own state in-run. A run that hit its
        # ceiling and a run that stopped early must not be distinguishable only by arithmetic
        # on the logs.
        _LOG.info("bc_graph_pretrain_stop steps_run=%d budget=%d stopped_early=%s %s",
                  steps_run, total_steps, stopped_early, monitor.stop.counters())
    path = trainer.save_checkpoint(loss_info)
    _LOG.info("bc_graph_pretrain_saved path=%s steps=%d budget=%d stopped_early=%s",
              path, steps_run, total_steps, stopped_early)
    return path
