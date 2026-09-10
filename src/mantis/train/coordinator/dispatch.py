# Exceeds the 300-line soft cap (R8): the declared route and both of its arms are ONE unit. The
# route decision, the graph arm and the grid arm have to be read together to see the property they
# exist for — that a graph-only input reaches the graph arm ALONE, as a provider, so it is never
# even evaluated on the grid route.
"""The DECLARED training-step dispatcher: a replay buffer to ONE gradient update.

Dispatch is keyed on the RESOLVED `EncodingSpec.representation` — the operator's declaration,
resolved by THE one authority — never on the buffer's runtime class. A closed match: graph routes
to `train_step_from_graph_batch`, grid to `train_step_from_tensors`, and anything else RAISES.

The sampling POLICY arrives from `StepCoordinatorConfig` and does NOT live on the trainer, which
would be a second authority beside `cfg.batch_size`. A declaration↔object mismatch is a NAMED
`RepresentationRouteError` surfaced at the route rather than an `AttributeError` from inside a
sampling call. torch / numpy / `graph_collate` imports are lazy inside the arms, so there is no
top-level `train → selfplay` edge.
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
    """ONE micro-batch's collated tensors + targets — what a `parts` callable returns. Defined here
    rather than in the trainer because THIS is where it is built, which keeps the
    `train.trainer -> train.coordinator` edge from existing at all."""

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
    #: The per-row POLICY weight, 1 on a full-search row and `train.fast_policy_weight` on a
    #: fast-arm one: the fast arm is weighted, not gated.
    policy_row_weight: Any
    n_graphs: int


class RepresentationRouteError(TypeError):
    """The declared representation and the training-step route disagree: an unknown or absent
    representation, a buffer that cannot serve the declared route, or the dense-only mixed arm
    entered under a graph declaration. A `TypeError` subclass, being a wiring error."""


def resolve_step_spec(full_config: Any) -> Any:
    """Resolve the coordinator's declared encoding spec through THE one authority: an undeclared
    encoding raises, and there is no default arm and no second path."""
    return resolve_from_config(full_config)


def _dump_train_collate(
    trainer: Any, wire: Any, error: BaseException, *, span: tuple[int, int], n_graphs: int
) -> None:
    """Write the offending training batch before the caller re-raises. Never raises.

    The eval path has had this for a while; the training path had nothing, so a contract failure
    in the trainer halted the run with no artifact.

    Raises:
        Nothing. `write_collate_dump` swallows its own failures and returns `None`; a diagnostic
        that could replace a named contract failure with its own exception would destroy the
        evidence it exists to keep.
    """
    from mantis.selfplay.collate_dump import write_collate_dump

    dump_dir = Path(getattr(trainer, "checkpoint_dir", "checkpoints")).parent / "collate_dumps"
    context = {
        "path": "train",
        "phase": "train_step",
        "fused_span": [int(span[0]), int(span[1])],
        "n_graphs": int(n_graphs),
        # No `run_id`: the Trainer has no such attribute (it lives in checkpoint metadata), so
        # reaching for one would be a dead field AND an undeclared seam access.
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

    ``caps_provider``, ``sample_threads_provider`` and ``fast_policy_weight_provider`` are ZERO-ARG
    CALLABLES handed to the GRAPH arm ALONE: `_grid_step` does not take them, so a grid run
    structurally cannot read those keys. Python evaluates every argument before the call, so
    passing a resolved VALUE would read `full_config["train"]` / `["selfplay"]` on both
    representations, and four FROZEN grid coordinators carry neither section.

    All three are REQUIRED and undefaulted: a default would be a code-side default for a
    config-derived value, and a caller that forgot one would silently get an uncapped step, a
    thread budget nobody derived, or a weight read through a seam that must not carry it.
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

    EXTRACTED, NOT DUPLICATED: the forward-only held-out evaluation reaches this preparation
    through the SAME code the training step uses, or the two losses would be measured through
    different wires and their comparison is the whole instrument.

    THE CAPS ARE READ EXACTLY ONCE, HERE, resolved AFTER the route decision, and the raise is never
    caught: an absent block reaches the caller with the missing level named, with no fallback arm.

    THE SPLIT IS PRE-COLLATE. The wire becomes a payload EXACTLY ONCE — the Rust getters COPY OUT —
    and each part is a numpy slice collated on demand by a LAZY callable, so only one micro-batch's
    tensors are ever resident. That laziness IS the memory bound.
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
    # exactly one authority and the reader census is frozen at two reads here, so a second read
    # anywhere — including a convenience re-read for the event payload — is a census failure, and
    # the census's own planted break (a THIRD read) is what proves that check still reds.
    caps = caps_provider()
    max_edges = caps.max_edges
    max_nodes = caps.max_nodes
    # The rebuild's width, DERIVED from the run's own keys: the cores the self-play workers and the
    # inference-server thread are not already holding. `sample_ring` is 1 386 ms of a 2 769 ms step
    # and 88 % of that is a serial loop over independent items.
    wire, targets = sampler(batch_size, augment=augment, recent_frac=recency_weight,
                            n_threads=sample_threads_provider())
    payload = graph_wire_from_rust(wire)
    plan = plan_microbatches(payload.edge_offsets, payload.node_offsets,
                             max_edges, max_nodes)
    device = trainer.device
    n_graphs = int(payload.n_graphs)
    # ONE evaluation of the weight rule, over the WHOLE batch, so the per-part numerator and the
    # whole-step denominator below cannot be computed from two different vectors. `hp` is the
    # trainer's own resolved hyper-parameters and nothing here supplies a fallback.
    policy_row_weight = graph_policy_row_weights(
        np.asarray(targets.is_full_search), float(fast_policy_weight_provider())
    )

    def _make(g0: int, g1: int):
        def _materialise():
            sub = slice_graph_wire(payload, g0, g1)
            tsl = slice_targets(targets, payload.legal_offsets, g0, g1)
            # Parameterization = the production collate call, at trainer cadence:
            # semantic="full" on EVERY batch and on every PART, so each micro-batch passes the
            # full structural and semantic contract on its own rather than inheriting the whole
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
                # Unconditionally armed: a contract failure is run-fatal, so the dump costs nothing
                # on any path the run survives.
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

    # The denominators are the WHOLE step's, computed ONCE from the FULL target arrays, so every
    # micro-batch divides by the quantity the un-split batch would have divided by and the parts
    # sum to the un-split loss exactly. They are NOT `1/M` and NOT `B_m/B`.
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
    """What the batch was MADE OF, read off the ring that just sampled it.

    ROWS PER GAME says whether the ring's same-game dedupe is doing anything; AGE, in rows back
    from the newest, says whether the ring is still being fed. Returned rather than emitted, since
    the sink belongs to the trainer. An empty dict on a buffer with no such reading: absence of an
    instrument is not a measurement and must not be published as zeros.
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

    GRAPH ONLY, and the refusal is the point: BC pretrain is the only consumer and it is a graph
    route. `augment` is fixed FALSE and `recency_weight` fixed 0.0, neither a knob — an augmented
    held-out batch measures positions the held-out set does not contain, and the BC ring carries no
    time ordering for a recency window to mean anything over.

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
