# Exceeds the 300-line soft cap (R8): the declared route and both of its arms are ONE
# unit. The route decision, the graph arm and the grid arm have to be read together to
# see the property they exist for — that a graph-only input reaches the graph arm ALONE,
# and reaches it as a provider so it is never even evaluated on the grid route.
"""The DECLARED training-step dispatcher — TD-1 / CARD-TRAINSTEP-ADAPTER (WPTS Phase T, R102).

The straight self-play arm's route from a replay buffer to ONE gradient update. Dispatch is
keyed on the RESOLVED `EncodingSpec.representation` — the operator's declaration, resolved by
THE one authority (`resolve_from_config`) — never on the buffer's runtime class (the old
side's `isinstance(buffer, HexgBuffer)` sniff is exactly what R102 bans). A closed match:
graph routes to `trainer.train_step_from_graph_batch`, grid routes to
`trainer.train_step_from_tensors`, anything else RAISES — an absent or unknown
representation is an ERROR, never a dense default (LAW-11; the `_build_buffer` posture).

The sampling POLICY (batch_size / augment / recency_weight) arrives from
`StepCoordinatorConfig` — the coordinator-authored knobs (CARD-COORD-KNOBS, R78/R80). It
does NOT live on the trainer: a trainer-side `train_step` reading its own config's batch
size would be a second authority beside `cfg.batch_size`, the duplicated-default class R1
exists to kill. That asymmetry is why TD-1's fix is this dispatcher and not a reinstated
`Trainer.train_step`.

A declaration↔object mismatch (graph declared, dense buffer injected — or vice versa) is a
NAMED `RepresentationRouteError`, the `BufferKindMismatch` posture: the route was already
chosen by the declaration, and a buffer that cannot serve it is a wiring error surfaced at
the route, not an `AttributeError` from the middle of a sampling call.

torch / numpy / `graph_collate` imports are lazy inside the arms (same pattern as `step.py`'s
lazy `batch_assembly` import): no top-level `train → selfplay` edge, module import-safe in
torch-free environments.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.encoding.resolvers import resolve_from_config

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphStepInputs:
    """ONE micro-batch's collated tensors + targets — what a `parts` callable returns.

    Defined here rather than in the trainer because THIS is where it is built; the trainer
    consumes it by attribute access and needs no import, which keeps the
    `train.trainer -> train.coordinator` edge from existing at all.
    """

    x: Any
    edge_index: Any
    edge_attr: Any
    legal_index: Any
    stone_mask: Any
    node_offsets: Any
    legal_offsets: Any
    policy_target: Any
    explicit_mask: Any
    tail_mass: Any
    outcomes: Any
    value_valid: Any
    #: R347(b) — the per-row POLICY weight, 1 on a full-search row and
    #: `train.fast_policy_weight` on a fast-arm one. It replaced a plain `is_full_search`
    #: mask and is named for what it is: the fast arm is weighted, not gated.
    policy_row_weight: Any
    n_graphs: int


class RepresentationRouteError(TypeError):
    """The declared representation and the training-step route disagree.

    Raised at DISPATCH (unknown/absent representation; a buffer that cannot serve the
    declared route; a graph run handed a dense `RecentBuffer`; the dense-only mixed arm
    entered under a graph declaration). A `TypeError` subclass for the same reason
    `BufferKindMismatch` is one: a wiring error, not a data error.
    """


def resolve_step_spec(full_config: Any) -> Any:
    """Resolve the coordinator's declared encoding spec through THE one authority.

    A thin veneer over `mantis.encoding.resolvers.resolve_from_config` — the same resolver
    the pool uses (post-WPBRIDGE/TD-4). An undeclared encoding raises
    `MissingEncodingError`; there is no default arm and no second path (LAW-11, R1).
    """
    return resolve_from_config(full_config)


def _dump_train_collate(
    trainer: Any, wire: Any, error: BaseException, *, span: tuple[int, int], n_graphs: int
) -> None:
    """Write the offending training batch before the caller re-raises. Never raises.

    The eval path has had this since R339(c); the training path had nothing, so when
    `F-816-37` fired in the trainer at R340 leg 3 the run halted with no artifact — which is
    the one thing R340(c)'s *"a firing halts with the artifact"* requires it not to do.

    Raises:
        Nothing. `write_collate_dump` swallows its own failures and returns `None`; a
        diagnostic that could replace a named contract failure with its own exception would
        destroy the evidence it exists to keep.
    """
    from mantis.selfplay.collate_dump import write_collate_dump

    dump_dir = Path(getattr(trainer, "checkpoint_dir", "checkpoints")).parent / "collate_dumps"
    context = {
        "path": "train",
        "phase": "train_step",
        "fused_span": [int(span[0]), int(span[1])],
        "n_graphs": int(n_graphs),
        # No `run_id`: the Trainer has no such attribute (it lives in checkpoint metadata),
        # so reaching for one would be a dead field AND an undeclared seam access.
        "step": getattr(trainer, "step", None),
    }
    path = write_collate_dump(wire, dump_dir=str(dump_dir), context=context, error=error)
    if path is None:
        _LOG.error("F-816-37 train-path dump-on-fire FAILED to write under %s", dump_dir)
    else:
        _LOG.error("F-816-37 train-path dump-on-fire wrote %s", path)


def run_declared_train_step(
    trainer: Any,
    buffer: Any,
    spec: Any,
    *,
    batch_size: int,
    augment: bool,
    recency_weight: float,
    recent_buffer: Any | None,
    caps_provider: Callable[[], Any],
    sample_threads_provider: Callable[[], int],
    fast_policy_weight_provider: Callable[[], float],
) -> dict[str, float]:
    """One straight self-play gradient update through the typed route for ``spec``.

    ``caps_provider`` is a ZERO-ARG CALLABLE returning the resolved
    `train.microbatch_caps` — the PROVIDER, never the value, and it is handed to the GRAPH arm
    ALONE. `_grid_step` does not take the parameter, so a grid run structurally cannot read
    the caps: a property of the call graph, checkable from two signatures, rather than a
    convention a later edit can break silently. Python evaluates every argument before the
    call, so passing the resolved VALUE here would read `full_config["train"]` on both
    representations — and four FROZEN grid coordinators construct a `full_config` with no
    `train` key at all (WP12-R F2, DESIGN_DFIX §3.11.1).

    It is REQUIRED and has no default. A default would be a code-side default for a
    config-derived value — the exact class R1 kills — and a caller that forgot it would
    silently get an UNCAPPED step, which is the defect this whole card exists to close.

    ``sample_threads_provider`` rides the SAME shape and for the SAME reason, not merely a
    related one (PERF-TRANCHE-1 B1). It is the ring rebuild's width, DERIVED from the run's
    own keys by `mantis.config.resolve.sample_threads`, and it is a PROVIDER because that
    resolver reads `full_config["selfplay"]` — which the four frozen grid coordinators do not
    have. Passing the resolved VALUE here evaluated it on both representations and raised
    `MissingSampleThreadsInputError` on every grid step; the laziness is what keeps a
    graph-only input out of the grid route, exactly as it does for the caps.

    Required and undefaulted, because a default would be a thread budget nobody derived,
    silently taking cores from the self-play workers on whatever box the run lands on.

    ``fast_policy_weight_provider`` (R347(b)) rides the SAME shape for the SAME reason: it
    resolves `train.fast_policy_weight`, and the four frozen grid coordinators carry no
    `train` section, so the resolved VALUE evaluated here would raise on every grid step. It
    is a PROVIDER rather than a read off `trainer`, because `trainer` is a declared seam
    (`TrainerLike`) and reaching through it for a config leaf would widen that protocol to
    carry a whole hyper-parameter object for one float.
    """
    representation = getattr(spec, "representation", None)
    if representation == "graph":
        return _graph_step(trainer, buffer, spec, batch_size=batch_size, augment=augment,
                           recency_weight=recency_weight, recent_buffer=recent_buffer,
                           caps_provider=caps_provider,
                           sample_threads_provider=sample_threads_provider,
                           fast_policy_weight_provider=fast_policy_weight_provider)
    raise RepresentationRouteError(
        f"declared representation {representation!r} selects no training-step route — an "
        "absent or unknown representation is an ERROR, never a dense default (LAW-11)"
    )


def _build_graph_parts(
    trainer: Any, buffer: Any, spec: Any, *,
    batch_size: int, augment: bool, recency_weight: float, recent_buffer: Any | None,
    caps_provider: Callable[[], Any], sample_threads_provider: Callable[[], int],
    fast_policy_weight_provider: Callable[[], float],
) -> dict[str, Any]:
    """One sampled graph batch, prepared for a step — the kwargs BOTH step routes take.

    EXTRACTED, NOT DUPLICATED (R328(d)). The forward-only held-out evaluation needs exactly
    this preparation and must reach it through the SAME code the training step uses: the
    collate parameterisation here is asserted to be the production one (`semantic="full"`,
    every batch, every part), and a second copy for the eval path would measure the held-out
    loss through a different wire than the training loss. The comparison between the two
    losses is the whole instrument, so they must share their producer.

    THE CAPS ARE READ EXACTLY ONCE, HERE. `tests/train/test_graph_microbatch_authority.py`
    freezes the reader census by `(module, receiver, enclosing function)`; this extraction
    MOVES those two reads from `_graph_step` into this function and adds none. Two reads
    before, two after, one authority throughout — the census expectation moves with them and
    its own planted break (a THIRD read) still reds.

    graph: `sample_graph_batch` → wire payload (ONCE) → `plan_microbatches` →
    per-part `collate_graph_batch` (semantic="full", the trainer's every-batch posture) +
    `stone_mask_from_batch` → `train_step_from_graph_batch`.

    Recency flows IN-ENGINE via `recent_frac=recency_weight` (old-side WP-5b commit-B
    parity); the graph side constructs no dense `RecentBuffer`, so receiving one is
    mis-wiring, refused loud rather than silently ignored.

    THE CAPS ARE RESOLVED HERE, on this route, after the route decision, and the raise is
    never caught: `caps_provider()` is the one invocation, and an absent block reaches the
    caller as `MissingMicrobatchCapsError` with the missing level named (LAW-11). There is no
    fallback arm and no `.get` on this path (WP12-R F2, F2-ABORT-5).

    THE SPLIT IS PRE-COLLATE. The wire is converted to a payload EXACTLY ONCE — the Rust
    getters COPY OUT, so a getter read per micro-batch would copy every array M times — and
    each part is a numpy slice that is collated on demand. The per-part callables are LAZY so
    only one micro-batch's tensors are ever resident; that laziness IS the memory bound.
    """
    if recent_buffer is not None:
        raise RepresentationRouteError(
            "the graph route takes no dense recent_buffer — recency flows in-engine "
            "(sample_graph_batch recent_frac); a RecentBuffer injected on a graph run is "
            "mis-wiring"
        )
    sampler = getattr(buffer, "sample_graph_batch", None)
    if sampler is None:
        raise RepresentationRouteError(
            f"declared representation 'graph' but the injected buffer "
            f"({type(buffer).__name__}) has no sample_graph_batch — the route and the "
            "buffer disagree; build the buffer from the declared identity"
        )
    import numpy as np
    import torch

    from mantis.selfplay.graph_collate import (
        GraphContractError,
        collate_graph_batch,
        graph_wire_from_rust,
        stone_mask_from_batch,
    )
    from mantis.selfplay.graph_wire_split import (
        plan_microbatches,
        slice_graph_wire,
        slice_targets,
    )
    from mantis.train.losses import graph_loss_denominators, graph_policy_row_weights

    # ONE read of each member, into a local. Not a style choice: `train.microbatch_caps` has
    # exactly one authority and `tests/train/test_graph_microbatch_authority.py` freezes the
    # reader census at two reads here and three in the resolver, so a second read anywhere
    # (including a convenience re-read for the event payload) is a census failure by design.
    caps = caps_provider()
    max_edges = caps.max_edges
    max_nodes = caps.max_nodes
    # B1: the rebuild's width, DERIVED from the run's own keys — the cores the self-play
    # workers and the inference-server thread are not already holding. `sample_ring` is
    # 1 386 ms of a 2 769 ms step and 88 % of that is a serial loop over independent items
    # (ledger §10.5 #1, split by PERF-TRANCHE-1 M-2).
    wire, targets = sampler(batch_size, augment=augment, recent_frac=recency_weight,
                            n_threads=sample_threads_provider())
    payload = graph_wire_from_rust(wire)
    plan = plan_microbatches(payload.edge_offsets, payload.node_offsets,
                             max_edges, max_nodes)
    device = trainer.device
    n_graphs = int(payload.n_graphs)
    # R347(b) — ONE evaluation of the weight rule, over the WHOLE batch, so the per-part
    # numerator and the whole-step denominator below cannot be computed from two different
    # vectors. `hp` is the trainer's own resolved hyper-parameters: the schema is the sole
    # default authority and nothing here supplies a fallback (R1).
    policy_row_weight = graph_policy_row_weights(
        np.asarray(targets.is_full_search), float(fast_policy_weight_provider())
    )

    def _make(g0: int, g1: int):
        def _materialise():
            sub = slice_graph_wire(payload, g0, g1)
            tsl = slice_targets(targets, payload.legal_offsets, g0, g1)
            # Parameterization = the production collate call (`inference_server.py`), trainer
            # cadence: semantic="full" EVERY batch (old seam design §6.1 — hot path runs
            # "canary"), and now on every PART, so each micro-batch passes the full
            # structural + semantic contract on its own rather than inheriting the whole
            # batch's verdict.
            try:
                batch = collate_graph_batch(
                    sub,
                    expected_version=1,
                    trunk_size=spec.trunk_size,
                    win_length=spec.win_length,
                    node_feat_dim=spec.node_feat_dim,
                    edge_feat_dim=spec.edge_feat_dim,
                    device=str(device),
                    semantic="full",
                    target_argmax_cells=tsl.target_argmax_cells,
                )
            except GraphContractError as exc:
                # Unconditionally armed: a contract failure is run-fatal, so the dump costs
                # nothing on any path the run survives (R340 leg 3 — F-816-37 fired HERE and
                # left nothing, because the eval path held the only dump).
                try:
                    _dump_train_collate(trainer, sub, exc, span=(g0, g1), n_graphs=n_graphs)
                except Exception:  # noqa: BLE001 — a dump may NEVER replace the contract failure
                    _LOG.exception("F-816-37 train-path dump-on-fire raised")
                raise
            return GraphStepInputs(
                x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                legal_index=batch.legal_node_gather, stone_mask=stone_mask_from_batch(batch),
                node_offsets=batch.node_offsets, legal_offsets=batch.legal_offsets,
                policy_target=torch.from_numpy(
                    np.asarray(tsl.policy_target, dtype=np.float32)).to(device),
                explicit_mask=torch.from_numpy(
                    np.asarray(tsl.explicit_mask, dtype=np.uint8)).to(device),
                tail_mass=torch.from_numpy(
                    np.asarray(tsl.tail_mass, dtype=np.float32)).to(device),
                outcomes=torch.from_numpy(
                    np.asarray(tsl.outcomes, dtype=np.float32)).to(device),
                value_valid=torch.from_numpy(
                    np.asarray(tsl.value_valid, dtype=np.uint8)).to(device),
                policy_row_weight=policy_row_weight[g0:g1].clone().to(device),
                n_graphs=g1 - g0,
            )

        return _materialise

    # The denominators are the WHOLE step's, computed ONCE from the FULL target arrays, so
    # every micro-batch divides by the quantity the un-split batch would have divided by and
    # the parts sum to the un-split loss exactly. They are NOT `1/M` and NOT `B_m/B`: neither
    # denominator is the graph count, and the two are different quantities from each other.
    policy_denominator, value_denominator = graph_loss_denominators(
        policy_row_weight, np.asarray(targets.value_valid), n_graphs)
    return {
        "parts": tuple(_make(g0, g1) for g0, g1 in plan),
        "policy_denominator": policy_denominator,
        "value_denominator": value_denominator,
        "total_edges": int(payload.edge_offsets[-1]),
        "total_nodes": int(payload.node_offsets[-1]),
        "caps_max_edges": max_edges,
        "caps_max_nodes": max_nodes,
        "batch_composition": _batch_composition(buffer),
    }


def _graph_step(
    trainer: Any, buffer: Any, spec: Any, *,
    batch_size: int, augment: bool, recency_weight: float, recent_buffer: Any | None,
    caps_provider: Callable[[], Any], sample_threads_provider: Callable[[], int],
    fast_policy_weight_provider: Callable[[], float],
) -> dict[str, float]:
    """One gradient update from a freshly sampled graph batch."""
    return trainer.train_step_from_graph_batch(**_build_graph_parts(
        trainer, buffer, spec, batch_size=batch_size, augment=augment,
        recency_weight=recency_weight, recent_buffer=recent_buffer,
        caps_provider=caps_provider, sample_threads_provider=sample_threads_provider,
        fast_policy_weight_provider=fast_policy_weight_provider,
    ))


def _batch_composition(buffer: Any) -> dict[str, int]:
    """R345(b)(6) — what the batch was MADE OF, read off the ring that just sampled it.

    Two facts, and each answers a question the loss curve cannot. ROWS PER GAME says whether
    the ring's same-game dedupe is doing anything: with every row tagged `-1` it was inert,
    and a batch could be a dozen positions from one game counted as a dozen samples. AGE, in
    rows back from the newest, says whether the ring is still being fed — a ring whose
    producer has died keeps sampling happily from older and older data and reports nothing.

    Returned rather than emitted here: the sink belongs to the trainer, and reaching into its
    private `_sink` from the dispatcher is the undeclared-seam access the conformance gate
    exists to refuse. The trainer puts these on its own `trainer_step` event, beside the edge
    and node counts that are already there.

    An empty dict on a buffer with no such reading (the dense ring, a test double). Absence of
    an instrument is not a measurement and must not be published as zeros.
    """
    reader = getattr(buffer, "last_batch_composition", None)
    if reader is None:
        return {}
    try:
        return {str(k): int(v) for k, v in dict(reader()).items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def run_declared_eval_step(
    trainer: Any, buffer: Any, spec: Any, *,
    batch_size: int,
    caps_provider: Callable[[], Any],
    sample_threads_provider: Callable[[], int],
    fast_policy_weight_provider: Callable[[], float],
) -> dict[str, float]:
    """One FORWARD-ONLY loss reading over `buffer`, through the declared graph route.

    GRAPH ONLY, and the refusal is the point rather than a gap: BC pretrain is the only
    consumer and it is a graph-arch route, so a grid caller here is asking for an instrument
    that was never built. Answering it with a dense forward would produce a number nobody can
    attribute (LAW-11's shape applied to an evaluation).

    `augment` is fixed FALSE and `recency_weight` fixed 0.0, neither exposed as a knob. An
    augmented held-out batch measures the loss on positions the held-out set does not contain,
    and the BC ring carries no time ordering for a recency window to mean anything over — the
    same reason `graph_route.BC_RECENCY_WEIGHT` is 0.0.

    Raises:
        RepresentationRouteError: `spec` does not declare the graph representation.
    """
    if getattr(spec, "representation", None) != "graph":
        raise RepresentationRouteError(
            f"declared representation {getattr(spec, 'representation', None)!r} selects no "
            "EVALUATION route — the forward-only loss exists on the graph arm only, and a "
            "dense forward here would be a number with no producer behind its name"
        )
    return trainer.eval_step_from_graph_batch(**_build_graph_parts(
        trainer, buffer, spec, batch_size=batch_size, augment=False,
        recency_weight=0.0, recent_buffer=None,
        caps_provider=caps_provider, sample_threads_provider=sample_threads_provider,
        fast_policy_weight_provider=fast_policy_weight_provider,
    ))
