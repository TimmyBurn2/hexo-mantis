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

import math
from typing import Any

from mantis.encoding.resolvers import resolve_from_config


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


def run_declared_train_step(
    trainer: Any,
    buffer: Any,
    spec: Any,
    *,
    batch_size: int,
    augment: bool,
    recency_weight: float,
    recent_buffer: Any | None,
) -> dict[str, float]:
    """One straight self-play gradient update through the typed route for ``spec``."""
    representation = getattr(spec, "representation", None)
    if representation == "graph":
        return _graph_step(trainer, buffer, spec, batch_size=batch_size, augment=augment,
                           recency_weight=recency_weight, recent_buffer=recent_buffer)
    if representation == "grid":
        return _grid_step(trainer, buffer, batch_size=batch_size, augment=augment,
                          recency_weight=recency_weight, recent_buffer=recent_buffer)
    raise RepresentationRouteError(
        f"declared representation {representation!r} selects no training-step route — an "
        "absent or unknown representation is an ERROR, never a dense default (LAW-11)"
    )


def _graph_step(
    trainer: Any, buffer: Any, spec: Any, *,
    batch_size: int, augment: bool, recency_weight: float, recent_buffer: Any | None,
) -> dict[str, float]:
    """graph: `sample_graph_batch` → `collate_graph_batch` (semantic="full", the trainer's
    every-batch posture) → `stone_mask_from_batch` → `train_step_from_graph_batch`.

    Recency flows IN-ENGINE via `recent_frac=recency_weight` (old-side WP-5b commit-B
    parity); the graph side constructs no dense `RecentBuffer`, so receiving one is
    mis-wiring, refused loud rather than silently ignored.
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

    from mantis.selfplay.graph_collate import collate_graph_batch, stone_mask_from_batch

    wire, targets = sampler(batch_size, augment=augment, recent_frac=recency_weight)
    # Parameterization = the production collate call (`inference_server.py`), trainer
    # cadence: semantic="full" every batch (old seam design §6.1 — hot path runs "canary").
    batch = collate_graph_batch(
        wire,
        expected_version=1,
        trunk_size=spec.trunk_size,
        win_length=spec.win_length,
        node_feat_dim=spec.node_feat_dim,
        edge_feat_dim=spec.edge_feat_dim,
        device=str(trainer.device),
        semantic="full",
        target_argmax_cells=targets.target_argmax_cells,
    )
    stone_mask = stone_mask_from_batch(batch)
    device = trainer.device
    policy_target = torch.from_numpy(np.asarray(targets.policy_target, dtype=np.float32)).to(device)
    outcomes = torch.from_numpy(np.asarray(targets.outcomes, dtype=np.float32)).to(device)
    value_valid = torch.from_numpy(np.asarray(targets.value_valid, dtype=np.uint8)).to(device)
    is_full_search = torch.from_numpy(
        np.asarray(targets.is_full_search, dtype=np.uint8)).to(device)
    return trainer.train_step_from_graph_batch(
        x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
        legal_mask=batch.legal_mask, stone_mask=stone_mask,
        node_offsets=batch.node_offsets, legal_offsets=batch.legal_offsets,
        policy_target=policy_target, outcomes=outcomes,
        value_valid=value_valid, is_full_search=is_full_search,
    )


def _grid_step(
    trainer: Any, buffer: Any, *,
    batch_size: int, augment: bool, recency_weight: float, recent_buffer: Any | None,
) -> dict[str, float]:
    """grid: the old-side `train_step` dense body, ported verbatim — recency mix
    (recent draw + aux reshape + zero ply-fill + uniform remainder) when a recent buffer is
    live and weighted, else one uniform `sample_batch_with_pos` draw."""
    sampler = getattr(buffer, "sample_batch_with_pos", None)
    if sampler is None:
        raise RepresentationRouteError(
            f"declared representation 'grid' but the injected buffer "
            f"({type(buffer).__name__}) has no sample_batch_with_pos — the route and the "
            "buffer disagree; build the buffer from the declared identity"
        )
    import numpy as np

    n_recent = 0
    if recent_buffer is not None and recent_buffer.size > 0 and recency_weight > 0.0:
        n_recent = max(1, int(round(batch_size * recency_weight)))
        n_uniform = batch_size - n_recent
        s_r, c_r, p_r, o_r, own_r, wl_r, ifs_r, vv_r = recent_buffer.sample(n_recent)
        # RecentBuffer stores aux flat (n, s*s); reshape to (n, s, s) (old-side WHY note).
        _bs = int(math.isqrt(own_r.shape[1]))
        own_r = own_r.reshape(-1, _bs, _bs)
        wl_r = wl_r.reshape(-1, _bs, _bs)
        # Recent rows lack a ply index; zero-fill (§S181-AUDIT Wave 4 4B-impl-3).
        pos_r = np.zeros(len(s_r), dtype=np.uint16)
        s_u, c_u, p_u, o_u, own_u, wl_u, ifs_u, pos_u, vv_u = sampler(max(1, n_uniform), augment)
        states = np.concatenate([s_r, s_u], axis=0)
        chain_planes = np.concatenate([c_r, c_u], axis=0)
        policies = np.concatenate([p_r, p_u], axis=0)
        outcomes = np.concatenate([o_r, o_u], axis=0)
        ownership = np.concatenate([own_r, own_u], axis=0)
        winning_line = np.concatenate([wl_r, wl_u], axis=0)
        is_full_search = np.concatenate([ifs_r, ifs_u], axis=0)
        position_indices = np.concatenate([pos_r, pos_u], axis=0)
        value_target_valid = np.concatenate([vv_r, vv_u], axis=0)
    else:
        (states, chain_planes, policies, outcomes, ownership, winning_line,
         is_full_search, position_indices, value_target_valid) = sampler(batch_size, augment)

    return trainer.train_step_from_tensors(
        states, policies, outcomes,
        chain_planes=chain_planes, ownership_targets=ownership,
        threat_targets=winning_line, is_full_search=is_full_search,
        n_pretrain=0, n_recent=n_recent,
        position_indices=position_indices, value_target_valid=value_target_valid,
    )
