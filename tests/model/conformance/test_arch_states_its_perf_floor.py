# >300 justify (R8): the floor/overhead instrument, the dispatch census it is checked against,
# and the planted breaks that show either can bite are one unit — a manifest asserted in one file
# against a census computed in another can be satisfied by editing whichever side is cheaper.
"""T7 — every arch `build_net` dispatches STATES ITS FLOOR, and the suite proves it stated one.

WHAT THIS SECTION IS FOR. SEAM_V1_DESIGN §3: "a new arch states its floor before anyone argues
about its speed." The perf rig produced the ×1.88-of-the-silicon-floor reading twice
(PERF-BASELINE, PERF-TRANCHE-1) and had NO producer test either time: the floor was a number in
a ledger, and nothing in the tree would have gone red if the harness that made it had stopped
measuring what its name says. This tier is that missing producer test, generalised per arch.

TWO CLAIMS, and only the first one gates.

  1. **THE MANIFEST.** The set of arch kinds `mantis.model.build.build_net` dispatches and the
     set with a registered floor probe are EQUAL — checked in BOTH directions, as SET EQUALITY
     and never as a cardinality (§2.7). The required side is DERIVED by walking `build.py`'s own
     `isinstance(arch, X)` branches, so adding a third arch to `build_net` makes this tier red
     until that arch states a floor. That is the whole mechanism GnnNetV2 has to satisfy, and it
     is why the derivation is off the dispatch rather than off a typed list beside it: a typed
     list is edited in the same commit as the arch and never notices.

  2. **THE MEASUREMENT** (`slow`) — floor µs and serving overhead per arch, as a TABLE. It
     asserts no magnitude, exactly as T6 does not, and for the same reason: a µs figure is
     host-attested or it is mechanism evidence, and nothing here is written into a tracked path.

WHAT "FLOOR" AND "OVERHEAD" MEAN HERE, because both words are already loaded in this repo.
FLOOR is the arch's own forward, alone: input construction outside the timed region, no seam, no
queue, no collate. OVERHEAD is `served / floor` where `served` is the SAME forward reached
through the arch's serving path — so the ratio is dimensionless and is a RELATION BETWEEN TWO
MEASUREMENTS TAKEN IN THE SAME PROCESS, which is what makes it host-independent and reportable
where a level is not. `PERF_BASELINE_LEDGER`'s 2.805-against-1.494 is this ratio's shape; this
tier commits neither number and compares against neither.

THE TIMER IS T6'S, IMPORTED, NOT REIMPLEMENTED. T6's module docstring argues that the instrument
and the self-tests showing it measures what its name says are one unit — so a second timer here
would be one whose exclusion of input construction nobody asserts. Importing it keeps ONE timer
authority and lets T6's differential remain the thing that makes both tiers non-vacuous.

THE GRAVES (read at HEAD, `docs/registers/falsified.md`). This tier PROPOSES NO OPTIMIZATION and
changes no hot path — F-17/F-18/F-19 (bench-falsified legal-set perf ideas, with F-19's
build-once-per-leaf corollary standing) and F-21 (the borrowed CUDA kernel, with its stated
fallback order) are cited because the first thing an overhead number does is tempt someone, and
they are the fence that temptation clears first.
"""
from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import mantis.model.build as build_module
from mantis.model.arch import CnnArch, GnnArch, GnnArchV2
from mantis.model.build import build_net

from _corpus import ConformanceRefusal, build_board, roster
from test_leaf_forward_throughput_harness import Measurement, measure_forward

BUILD_SOURCE = Path(build_module.__file__)


class ArchDeclaresNoFloor(ConformanceRefusal):
    """`build_net` dispatches an arch kind that has registered no floor probe."""


class FloorProbeForUnknownArch(ConformanceRefusal):
    """A floor probe names an arch kind `build_net` does not dispatch — a probe for nothing."""


class DispatchCensusEmpty(ConformanceRefusal):
    """The walk of `build_net` found no arch branch, so the manifest is checked against nothing."""


class ServedBeneathTheFloor(ConformanceRefusal):
    """The serving measurement came in under the forward it contains — the arms are not nested."""


class OverheadFromOneMeasurement(ConformanceRefusal):
    """Floor and served are the same measurement, so the ratio is 1 by construction."""


@dataclass(frozen=True)
class FloorProbe:
    """One arch kind's two arms. `floor` times the bare forward; `served` times the same
    forward reached through the serving path, so `served` STRICTLY CONTAINS `floor`."""

    arch_kind: str
    floor_arm: Callable[[Any], tuple[Callable[[], Any], Callable[[Any], None]]]
    served_arm: Callable[[Any], tuple[Callable[[], Any], Callable[[Any], None]]]


def arch_kinds_dispatched(source: Path | str) -> frozenset[str]:
    """The arch kinds `build_net` names in its own `isinstance` branches.

    DERIVED FROM THE DISPATCH, which is the point: the alternative — a tuple of arch classes
    typed next to this tier — is written and edited in the same commit as a new arch, so it can
    never be the thing that notices one. Accepts a path or a source string so the planted breaks
    below drive the same walker over a stand-in.
    """
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    tree = ast.parse(text)
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "build_net"):
            continue
        for inner in ast.walk(node):
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
                continue
            if inner.func.id != "isinstance" or len(inner.args) != 2:
                continue
            target = inner.args[1]
            if isinstance(target, ast.Name):
                kinds.add(target.id)
            elif isinstance(target, ast.Tuple):
                kinds.update(e.id for e in target.elts if isinstance(e, ast.Name))
    return frozenset(kinds)


def check_floor_manifest(
    dispatched: frozenset[str], registered: frozenset[str]
) -> frozenset[str]:
    """SET EQUALITY, both directions, and a vacuity refusal in front of it.

    The vacuity refusal is not decoration: a walker that returns the empty set makes
    `dispatched <= registered` true for free, which is the shape of green this suite exists to
    refuse. Cardinality is never compared — two sets of the same size that disagree on a member
    is exactly the case an arch rename produces.
    """
    if not dispatched:
        raise DispatchCensusEmpty(
            f"the walk of {BUILD_SOURCE.name} found no arch branch in `build_net`. An empty "
            "required set satisfies every manifest check by construction, so this is refused "
            "before the comparison rather than reported as a clean manifest."
        )
    missing = sorted(dispatched - registered)
    if missing:
        raise ArchDeclaresNoFloor(
            f"`build_net` dispatches {missing} and no floor probe is registered for them. An "
            "arch states its floor before anyone argues about its speed (SEAM_V1_DESIGN §3); "
            "landing an arch without one is the state this tier makes red."
        )
    stray = sorted(registered - dispatched)
    if stray:
        raise FloorProbeForUnknownArch(
            f"floor probes are registered for {stray}, which `build_net` does not dispatch. A "
            "probe for an arch nothing constructs measures nothing, and it inflates the "
            "manifest so a genuinely missing arch can hide behind the count."
        )
    return dispatched


def serving_overhead(floor: Measurement, served: Measurement) -> float:
    """`served / floor` — dimensionless, both terms measured in THIS process.

    Refuses the two degenerate ways the ratio stops meaning its name: the same measurement on
    both sides (1.0 by construction), and a served arm that came in beneath the forward it is
    supposed to contain (the arms are not nested, so the ratio is not an overhead).
    """
    if floor is served:
        raise OverheadFromOneMeasurement(
            "the floor and served arms are the same Measurement object, so the ratio is 1.0 by "
            "construction and reports nothing about the seam."
        )
    if served.median_ns < floor.median_ns:
        raise ServedBeneathTheFloor(
            f"the served median {served.median_ns} ns is below the floor median "
            f"{floor.median_ns} ns. The served arm is supposed to CONTAIN the forward the floor "
            "arm times; a served arm that is faster is timing a different, smaller thing."
        )
    return served.median_ns / max(floor.median_ns, 1.0)


# --------------------------------------------------------------------------------------- #
# The registered probes — one per arch kind `build_net` dispatches
# --------------------------------------------------------------------------------------- #
def _gnn_probe_arms(spec, arch_cls=GnnArch):
    """Graph arches: floor = `forward_batch` on an already-collated batch; served = the same
    forward reached from the wire, through `collate_graph_batch` and the ragged softmax.

    `arch_cls` is the only thing that differs between the V1 and V2 probes, which is the seam's
    own claim in miniature: the serving path is the arch's, and swapping the arch swaps nothing
    else. A second copy of these arms for V2 would have been a second serving path to keep in
    step, and the first divergence would have been invisible.
    """
    import torch

    from mantis._engine import HexgBuffer
    from mantis.selfplay.graph_collate import collate_graph_batch, segment_softmax

    net = build_net(
        arch_cls(
            in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
            hidden=8, num_layers=1, policy_hidden=8, value_hidden=8,
        )
    ).eval()

    def wire():
        board = build_board(spec.name, [(i, 0) for i in range(6)])
        legal = board.legal_moves()
        buffer = HexgBuffer(2, spec.name, 8)
        buffer.push_graph_position(
            board.get_stones(), [(legal[0][0], legal[0][1], 1.0)],
            board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8,
        )
        return buffer.sample_graph_batch(1, augment=False)[0]

    def collate(w):
        batch = collate_graph_batch(
            w, trunk_size=spec.trunk_size, win_length=spec.win_length,
            node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
        )
        stone_mask = torch.zeros(batch.x.shape[0], dtype=torch.bool)
        stone_mask[: int(batch.n_stones.sum())] = True
        return batch, stone_mask

    def run(batch, stone_mask):
        with torch.no_grad():
            return net.forward_batch(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.legal_node_gather, stone_mask, batch.node_offsets,
            )

    def floor_arm(_spec):
        collated = collate(wire())
        return (lambda: collated), (lambda payload: run(*payload))

    def served_arm(_spec):
        def forward(w) -> None:
            batch, stone_mask = collate(w)
            logits, _value, _bins = run(batch, stone_mask)
            segment_softmax(logits, batch.legal_offsets)

        return wire, forward

    return floor_arm, served_arm


def _cnn_probe_arms(spec):
    """CnnArch: floor = the trunk+heads forward on a prepared plane stack; served = the same
    forward reached from a constructed board through the plane assembly that feeds it.

    The served arm is the PRODUCTION route, not a stand-in: `GameState.from_board(...)`,
    `.to_tensor()`, then the spec's own `kept_plane_indices` slice — which is what
    `selfplay/inference_local.py::_forward_boards` does, and the plane count comes off the bound
    spec there for the same reason it does here.
    """
    import torch

    from mantis._engine import Board
    from mantis.env.game_state import GameState

    net = build_net(
        CnnArch(
            board_size=spec.board_size, in_channels=spec.n_planes,
            filters=8, res_blocks=1, se_reduction_ratio=2,
        )
    ).eval()

    def board():
        b = Board.with_encoding_name(spec.name)
        for i in range(4):
            b.apply_move(i, 0)
        return b

    def planes(b):
        tensor, _centers = GameState.from_board(b).to_tensor()
        if tensor.shape[1] != spec.n_planes:
            tensor = tensor[:, list(spec.kept_plane_indices)]
        return torch.from_numpy(tensor).float()

    def run(x) -> None:
        with torch.no_grad():
            net.forward(x)

    def floor_arm(_spec):
        prepared = planes(board())
        return (lambda: prepared), run

    def served_arm(_spec):
        constructed = board()
        return (lambda: constructed), (lambda b: run(planes(b)))

    return floor_arm, served_arm


def registered_probes() -> dict[str, FloorProbe]:
    """The registry. Keyed by the arch CLASS NAME, which is what the dispatch census reports."""
    return {
        "GnnArch": FloorProbe(
            arch_kind="GnnArch",
            floor_arm=lambda spec: _gnn_probe_arms(spec)[0](spec),
            served_arm=lambda spec: _gnn_probe_arms(spec)[1](spec),
        ),
        "GnnArchV2": FloorProbe(
            arch_kind="GnnArchV2",
            floor_arm=lambda spec: _gnn_probe_arms(spec, GnnArchV2)[0](spec),
            served_arm=lambda spec: _gnn_probe_arms(spec, GnnArchV2)[1](spec),
        ),
        "CnnArch": FloorProbe(
            arch_kind="CnnArch",
            floor_arm=lambda spec: _cnn_probe_arms(spec)[0](spec),
            served_arm=lambda spec: _cnn_probe_arms(spec)[1](spec),
        ),
    }


def specs_for(arch_kind: str) -> tuple[Any, ...]:
    """EVERY registered encoding whose representation the arch kind serves, NAME-SORTED.

    AUDIT-1 F-41. This returned the FIRST match in `all_specs()` order, so the table measured
    `gnn_axis_v1` and never `gnn_axis_r8` — run6's own identity — and a roster REORDER would
    have silently changed the subject of a measurement nothing else in the tree reproduces.
    Name-sorted so the order is a property of the registry's contents, not its declaration order.

    Raises:
        ArchDeclaresNoFloor: no registered encoding carries this arch's representation, so its
            floor cannot be measured on any subject the tree actually ships.
    """
    graph = arch_kind.startswith("GnnArch")
    matches = tuple(sorted((s for s in roster() if bool(s.is_graph) is graph),
                           key=lambda s: s.name))
    if not matches:
        raise ArchDeclaresNoFloor(
            f"{arch_kind}: no registered encoding carries the representation this arch serves, "
            "so its floor cannot be measured on any subject the tree actually ships."
        )
    return matches


# --------------------------------------------------------------------------------------- #
# The manifest and its planted breaks — ALL DEFAULT TIER
# --------------------------------------------------------------------------------------- #
def test_EVERY_arch_build_net_dispatches_HAS_a_registered_floor_probe(derived):
    """Claim 1. Both directions, set equality, against a census derived from the dispatch."""
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    registered = frozenset(registered_probes())
    derived("t7.dispatch.kinds", sorted(dispatched))
    derived("t7.probes.registered", sorted(registered))
    assert check_floor_manifest(dispatched, registered) == dispatched


def test_the_dispatch_census_SEES_a_third_arch_branch(derived):
    """PB-T7a. The manifest is only load-bearing if adding an arch to `build_net` moves the
    required set — otherwise it is a fixed set of literals agreeing with itself. The planted
    name is deliberately one no registry knows: an arch that HAS landed cannot demonstrate the
    refusal, which is exactly what happened when `GnnArchV2` stood here and then registered."""
    planted = (
        "def build_net(arch):\n"
        "    if isinstance(arch, CnnArch):\n        return A()\n"
        "    elif isinstance(arch, GnnArch):\n        return B()\n"
        "    elif isinstance(arch, GnnArchNext):\n        return C()\n"
        "    raise RepresentationMismatch('no')\n"
    )
    seen = arch_kinds_dispatched(planted)
    derived("t7.dispatch.planted", sorted(seen))
    assert "GnnArchNext" in seen
    with pytest.raises(ArchDeclaresNoFloor, match="GnnArchNext"):
        check_floor_manifest(seen, frozenset(registered_probes()))


def test_the_dispatch_census_does_NOT_fire_on_isinstance_OUTSIDE_build_net():
    """Negative control. A census that collects every `isinstance` in the module would report
    arch kinds from helpers and validators, and its green would stop meaning its name."""
    planted = (
        "def _validate(x):\n    if isinstance(x, SomethingElse):\n        return 1\n"
        "def build_net(arch):\n"
        "    if isinstance(arch, CnnArch):\n        return A()\n"
        "    raise RepresentationMismatch('no')\n"
    )
    assert arch_kinds_dispatched(planted) == frozenset({"CnnArch"})


def test_an_EMPTY_dispatch_census_is_REFUSED_rather_than_reported_clean():
    """PB-T7b. `dispatched <= registered` is free when `dispatched` is empty — the exact shape
    of vacuous green this suite refuses everywhere else."""
    with pytest.raises(DispatchCensusEmpty, match="satisfies every manifest check"):
        check_floor_manifest(frozenset(), frozenset(registered_probes()))


def test_a_PROBE_for_an_arch_NOTHING_dispatches_is_refused():
    """PB-T7c. The reverse direction. A stray probe inflates the registered set, so a genuinely
    missing arch could hide behind a matching count — which is why this is set equality."""
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    with pytest.raises(FloorProbeForUnknownArch, match="GhostArch"):
        check_floor_manifest(dispatched, frozenset(registered_probes()) | {"GhostArch"})


def test_the_manifest_does_NOT_fire_on_the_REAL_pair():
    """Negative control for claim 1. A manifest that is red at HEAD measures nothing."""
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    assert check_floor_manifest(dispatched, dispatched) == dispatched


def test_the_CHECKPOINT_LOADERS_arch_registry_matches_the_same_dispatch(derived):
    """PK3, third registry. `train.checkpoints._ARCH_KINDS` is what rehydrates a stamp, and an
    arch missing from it is a checkpoint that cannot come back — the LAW-12 half of the same
    manifest. Checked against THIS census rather than a second walk, so the three registries
    (floor probes, memory envelopes, loader kinds) are all held to one reading of `build_net`.
    """
    from mantis.train.checkpoints import _ARCH_KINDS

    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    derived("t7.loader.arch_kinds", sorted(_ARCH_KINDS))
    assert check_floor_manifest(dispatched, frozenset(_ARCH_KINDS)) == dispatched
    assert all(cls.__name__ == kind for kind, cls in _ARCH_KINDS.items()), (
        "a loader row is keyed by a name that is not its class's — `_arch_to_dict` writes "
        "`type(arch).__name__`, so a mismatched key is a stamp that cannot be read back"
    )


# --------------------------------------------------------------------------------------- #
# The overhead relation and its planted breaks — ALL DEFAULT TIER
# --------------------------------------------------------------------------------------- #
def test_the_OVERHEAD_is_a_RELATION_between_two_measurements_not_a_level(derived):
    """Self-test. A sleep injected into the served arm ALONE must push the ratio above 1. Both
    terms are measured in this process, so no host-dependent bound is committed — the same
    device T6's differential uses, applied to a ratio instead of a difference."""
    sleep_s = 0.005
    floor = measure_forward(
        lambda: None, lambda p: None, repeats=3, warmup=1, device_type="cpu"
    )
    served = measure_forward(
        lambda: None, lambda p: time.sleep(sleep_s), repeats=3, warmup=1, device_type="cpu"
    )
    ratio = serving_overhead(floor, served)
    derived("t7.overhead.relation", ratio)
    assert ratio > 1.0, (
        "work added to the served arm alone did not move the ratio, so `served / floor` is not "
        "reading the two arms separately"
    )


def test_ONE_measurement_on_BOTH_sides_is_refused():
    """PB-T7d. The ratio's degenerate case: hand it the same object and it is 1.0 for free."""
    only = measure_forward(lambda: None, lambda p: None, repeats=2, warmup=1, device_type="cpu")
    with pytest.raises(OverheadFromOneMeasurement, match="1.0 by construction"):
        serving_overhead(only, only)


def test_a_SERVED_arm_BENEATH_the_floor_is_refused():
    """PB-T7e. The served path contains the forward, so a served arm that reads faster is
    timing something smaller — a refusal, never a sub-1.0 overhead nobody questions."""
    cheap = Measurement(median_ns=10.0, iqr_ns=0.0, sync_calls=0, repeats=3)
    dear = Measurement(median_ns=100.0, iqr_ns=0.0, sync_calls=0, repeats=3)
    assert serving_overhead(cheap, dear) > 1.0
    with pytest.raises(ServedBeneathTheFloor, match="timing a different, smaller thing"):
        serving_overhead(dear, cheap)


def test_the_FLOOR_arm_input_FOLLOWS_the_arch_it_was_built_for(derived):
    """The derivation control, the half a manifest cannot give. A probe whose constructed input
    does not move when the arch's declared width moves is a fixed fixture wearing an arch."""
    spec = specs_for("GnnArch")[0]
    narrow = build_net(
        GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=8, num_layers=1,
                policy_hidden=8, value_hidden=8)
    )
    wide = build_net(
        GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=16, num_layers=1,
                policy_hidden=8, value_hidden=8)
    )
    narrow_params = sum(p.numel() for p in narrow.parameters())
    wide_params = sum(p.numel() for p in wide.parameters())
    derived("t7.derivation.narrow_params", narrow_params)
    derived("t7.derivation.wide_params", wide_params)
    assert wide_params > narrow_params, (
        "the declared hidden width did not change the constructed net, so a floor measured "
        "against this arch is not a measurement of the arch"
    )


# --------------------------------------------------------------------------------------- #
# The MEASUREMENT (`slow`) — a table, no magnitude asserted
# --------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_report_the_per_arch_floor_and_serving_overhead(derived):
    """The measurement. One row per registered arch kind: floor median, served median, the
    dimensionless overhead, repeats and device. NOTHING here is compared to a threshold and
    nothing is written to a tracked path.

    COVERAGE, STATED, because a table is read as its own scope: CPU only, at the smallest net
    each arch admits, on EVERY registered encoding the arch serves (AUDIT-1 F-41 — it was one,
    picked by roster order, and it was never r8). The magnitudes are
    therefore mechanism evidence about the SEAM's shape and are not comparable to
    `PERF_BASELINE_LEDGER`'s production-shape readings — which were taken on the box, at
    production width, in bf16.
    """
    rows: list[dict] = []
    for kind, probe in sorted(registered_probes().items()):
        for spec in specs_for(kind):
            floor_build, floor_forward = probe.floor_arm(spec)
            served_build, served_forward = probe.served_arm(spec)
            floor = measure_forward(
                floor_build, floor_forward, repeats=5, warmup=2, device_type="cpu"
            )
            served = measure_forward(
                served_build, served_forward, repeats=5, warmup=2, device_type="cpu"
            )
            rows.append(
                {
                    "arch_kind": kind,
                    "encoding": spec.name,
                    "floor_median_ns": floor.median_ns,
                    "served_median_ns": served.median_ns,
                    "serving_overhead": serving_overhead(floor, served),
                    "repeats": floor.repeats,
                    "device": "cpu",
                }
            )
    derived("t7.measurement.rows", rows)
    assert {(row["arch_kind"], row["encoding"]) for row in rows} == {
        (kind, spec.name) for kind in registered_probes() for spec in specs_for(kind)
    }, (
        "the measured table does not cover every registered arch kind on every encoding it "
        "serves, so an arch could state a floor in the manifest and never be measured against "
        "it — or be measured on only one of the encodings it ships for (AUDIT-1 F-41)"
    )
