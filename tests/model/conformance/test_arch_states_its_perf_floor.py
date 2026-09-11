# >300 justify (R8): the floor/overhead instrument, the dispatch census it is checked against,
# and the planted breaks that show either can bite are one unit — a manifest asserted in one file
# against a census computed in another can be satisfied by editing whichever side is cheaper.
"""T7 — every arch `build_net` dispatches STATES ITS FLOOR, and the suite proves it stated one.

The perf rig produced its x1.88-of-the-silicon-floor reading twice with NO producer test either
time, so nothing would have gone red if the harness had stopped measuring what its name says.

TWO CLAIMS, only the first gating. (1) THE MANIFEST: the arch kinds `build_net` dispatches and
the kinds with a registered floor probe are EQUAL, as SET EQUALITY in both directions, with the
required side DERIVED by walking `build.py`'s own `isinstance` branches — a typed list beside
this tier is edited in the same commit as a new arch and can never notice one. (2) THE
MEASUREMENT (`slow`): a TABLE asserting no magnitude, whose one assertion is the NESTING — that
the served arm did not read faster than the forward it contains — made noise-aware.

FLOOR is the arch's forward alone, input construction outside the timed region; OVERHEAD is
`served / floor` with both terms measured in one process, which is what makes it reportable
where a level is not. The timer is T6's, imported, so there is ONE timer authority.
"""
from __future__ import annotations

import ast
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import mantis.model.build as build_module
from mantis.model.arch import GnnArch, GnnArchV2
from mantis.model.build import build_net

from _corpus import ConformanceRefusal, build_board, roster
from test_leaf_forward_throughput_harness import (
    Measurement,
    measure_forward,
    measure_forward_paired,
)

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
    """The arch kinds `build_net` names in its own `isinstance` branches, DERIVED from the
    dispatch. Accepts a path or a source string so the planted breaks drive the same walker."""
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
    """SET EQUALITY, both directions, with a vacuity refusal in front of it: an empty
    `dispatched` makes the subset test true for free, and two same-sized sets disagreeing on a
    member is exactly what an arch rename produces."""
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
    """`served / floor` — dimensionless, both terms measured in THIS process. Refuses the two
    degenerate readings: one measurement on both sides, and a served arm beneath the forward it
    is supposed to contain."""
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


#: The three readings of one floor/served pair. Names, not booleans, because the third one is
#: not "not inverted": it is the measurement declining to answer.
NESTING_ORDERED = "ordered"
NESTING_INVERTED = "inverted"
NESTING_UNRESOLVED = "unresolved"

#: Instrument parameters for the paired reading — NOT thresholds on any subject. A five-sample
#: median of a wall-clock CPU timing is decided by whichever repeat the scheduler preempted;
#: the median of twenty-five is not.
_NESTING_REPEATS = 25
_NESTING_WARMUP = 5
#: The sleep the mutation self-tests plant to make a REAL inversion — an instrument parameter,
#: no subject's threshold. Measured: at 0.005 s under load ~20 the planted deficit cleared the
#: bar by a factor of 2.9 at worst.
_INVERSION_PLANT_S = 0.02


class ServingNestingUnresolved(UserWarning):
    """The two arms differ by less than the noise the measurement itself carries. A WARNING, not
    a refusal and not a skip — this suite refuses every skip spelling — so the reading is
    REPORTED and the run continues."""


def nesting_noise_ns(floor: Measurement, served: Measurement) -> float:
    """The spread the two readings JOINTLY carry, in ns — the bar an inversion must clear.

    The sum of the two IQRs, UNDIVIDED, measured rather than chosen: `IQR / sqrt(repeats)`
    assumes independent draws, but host load arrives in bursts, so the effective sample count is
    a fraction of the nominal. Driven at load ~20 the sqrt form left the bar at 29.18 ms against
    a 31.58 ms deficit and red a pair nested by construction. A noise bar, not a p-value.
    """
    return floor.iqr_ns + served.iqr_ns


def nesting_verdict(floor: Measurement, served: Measurement) -> str:
    """ORDERED / INVERTED / UNRESOLVED for one pair — and the ASYMMETRY is the whole design.

    `served` containing `floor` is STRUCTURAL, so any `served >= floor` is ORDERED with no
    margin. Only a reading that CONTRADICTS the structure must clear the noise bar, because a
    served arm that stopped containing the forward misses by a whole collate and never by a
    hairline — a hairline deficit is the scheduler.
    """
    deficit = floor.median_ns - served.median_ns
    if deficit <= 0.0:
        return NESTING_ORDERED
    return NESTING_INVERTED if deficit > nesting_noise_ns(floor, served) else NESTING_UNRESOLVED


def _gnn_probe_arms(spec, arch_cls=GnnArch):
    """Graph arches: floor = `forward_batch` on an already-collated batch; served = the same
    forward from the wire, through `collate_graph_batch` and the ragged softmax. `arch_cls` is
    the only difference between the V1 and V2 probes, so there is one serving path to keep in
    step rather than two."""
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
        # Stated, not sniffed: the floor net is built on CPU, so the batch lands there too.
        batch = collate_graph_batch(
            w, trunk_size=spec.trunk_size, win_length=spec.win_length,
            node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim, device="cpu",
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
    }


def specs_for(arch_kind: str) -> tuple[Any, ...]:
    """EVERY registered encoding whose representation the arch kind serves, NAME-SORTED.

    It returned the FIRST match in `all_specs()` order, so the table measured `gnn_axis_v1` and
    never run6's own `gnn_axis_r8`, and a roster REORDER silently changed the subject.

    Raises:
        ArchDeclaresNoFloor: no registered encoding carries this arch's representation.
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


def test_EVERY_arch_build_net_dispatches_HAS_a_registered_floor_probe(derived):
    """Claim 1. Both directions, set equality, against a census derived from the dispatch."""
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    registered = frozenset(registered_probes())
    derived("t7.dispatch.kinds", sorted(dispatched))
    derived("t7.probes.registered", sorted(registered))
    assert check_floor_manifest(dispatched, registered) == dispatched


def test_the_dispatch_census_SEES_a_third_arch_branch(derived):
    """PB-T7a. The manifest is load-bearing only if adding an arch to `build_net` moves the
    required set. The planted name is deliberately one no registry knows: an arch that HAS
    landed cannot demonstrate the refusal, which is what happened when `GnnArchV2` stood here."""
    planted = (
        "def build_net(arch):\n"
        "    if isinstance(arch, GnnArch):\n        return B()\n"
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
        "    if isinstance(arch, GnnArch):\n        return A()\n"
        "    raise RepresentationMismatch('no')\n"
    )
    assert arch_kinds_dispatched(planted) == frozenset({"GnnArch"})


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
    """PK3, third registry. `train.checkpoints._ARCH_KINDS` is what rehydrates a stamp, so an
    arch missing from it is a checkpoint that cannot come back. Checked against THIS census, so
    all three registries are held to one reading of `build_net`."""
    from mantis.train.checkpoints import _ARCH_KINDS

    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    derived("t7.loader.arch_kinds", sorted(_ARCH_KINDS))
    assert check_floor_manifest(dispatched, frozenset(_ARCH_KINDS)) == dispatched
    assert all(cls.__name__ == kind for kind, cls in _ARCH_KINDS.items()), (
        "a loader row is keyed by a name that is not its class's — `_arch_to_dict` writes "
        "`type(arch).__name__`, so a mismatched key is a stamp that cannot be read back"
    )


def test_the_OVERHEAD_is_a_RELATION_between_two_measurements_not_a_level(derived):
    """Self-test. A sleep injected into the served arm ALONE must push the ratio above 1; both
    terms are measured in this process, so no host-dependent bound is committed."""
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


def test_the_NOISE_AWARE_verdict_STILL_REDS_on_a_GENUINELY_inverted_arm(derived):
    """THE MUTATION SELF-TEST for the loosened comparison (LAW-07). The plant is a REAL
    measurement — the floor arm sleeps and the served arm does not — and it must survive the
    noise bar and reach the strict refusal on whatever host runs it."""
    floor, served = measure_forward_paired(
        lambda: None, lambda payload: time.sleep(_INVERSION_PLANT_S),
        lambda: None, lambda payload: None,
        repeats=5, warmup=1, device_type="cpu",
    )
    derived("t7.mutation.inverted_floor_ns", floor.median_ns)
    derived("t7.mutation.inverted_served_ns", served.median_ns)
    derived("t7.mutation.inverted_noise_ns", nesting_noise_ns(floor, served))
    assert nesting_verdict(floor, served) == NESTING_INVERTED, (
        "a served arm measurably faster than the floor arm read as anything but INVERTED — "
        "the noise bar has swallowed the defect it is supposed to let through"
    )
    with pytest.raises(ServedBeneathTheFloor, match="timing a different, smaller thing"):
        serving_overhead(floor, served)


def test_the_NOISE_AWARE_verdict_RESOLVES_a_GENUINELY_nested_pair(derived):
    """Non-vacuity control, the half that stops the loosening becoming an off switch: a verdict
    answering UNRESOLVED to everything would never red and would pass every mutation test above.
    So the ordered direction must RESOLVE, not merely fail to be inverted."""
    floor, served = measure_forward_paired(
        lambda: None, lambda payload: None,
        lambda: None, lambda payload: time.sleep(_INVERSION_PLANT_S),
        repeats=5, warmup=1, device_type="cpu",
    )
    derived("t7.mutation.nested_ratio", served.median_ns / max(floor.median_ns, 1.0))
    assert nesting_verdict(floor, served) == NESTING_ORDERED
    assert serving_overhead(floor, served) > 1.0


def test_a_HAIRLINE_inversion_reads_UNRESOLVED_where_the_STRICT_form_reds(derived):
    """The loosening, shown REAL and shown to be a bar rather than a switch. Both pairs carry the
    SAME spread and only the deficit differs: if the first still red nothing changed, and if the
    second did not, the bar is an off switch."""
    spread = Measurement(median_ns=1000.0, iqr_ns=200.0, sync_calls=0, repeats=25)
    hairline = Measurement(median_ns=990.0, iqr_ns=200.0, sync_calls=0, repeats=25)
    gross = Measurement(median_ns=10.0, iqr_ns=200.0, sync_calls=0, repeats=25)
    derived("t7.hairline.noise_ns", nesting_noise_ns(spread, hairline))
    assert nesting_verdict(spread, hairline) == NESTING_UNRESOLVED
    assert nesting_verdict(spread, gross) == NESTING_INVERTED
    with pytest.raises(ServedBeneathTheFloor):
        serving_overhead(spread, hairline)


def test_the_NOISE_BAR_FOLLOWS_the_SPREAD_of_the_readings_it_qualifies(derived):
    """The bar's one moving part: a steady reading demands a small gap before it calls an
    inversion, a jittery one demands a large gap. A constant bar would commit a magnitude."""
    steady = Measurement(median_ns=1000.0, iqr_ns=10.0, sync_calls=0, repeats=25)
    jittery = Measurement(median_ns=1000.0, iqr_ns=900.0, sync_calls=0, repeats=25)
    derived("t7.noise.steady_ns", nesting_noise_ns(steady, steady))
    derived("t7.noise.jittery_ns", nesting_noise_ns(jittery, jittery))
    assert nesting_noise_ns(steady, steady) < nesting_noise_ns(jittery, jittery)
    beneath = Measurement(median_ns=600.0, iqr_ns=10.0, sync_calls=0, repeats=25)
    assert nesting_verdict(steady, beneath) == NESTING_INVERTED, (
        "a 400 ns deficit between two readings that each wobble by 10 ns read as noise"
    )
    wobbly_beneath = Measurement(median_ns=600.0, iqr_ns=900.0, sync_calls=0, repeats=25)
    assert nesting_verdict(jittery, wobbly_beneath) == NESTING_UNRESOLVED, (
        "the same 400 ns deficit resolved between two readings that each wobble by 900 ns"
    )


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


@pytest.mark.slow
def test_report_the_per_arch_floor_and_serving_overhead(derived):
    """The measurement: one row per registered arch kind — floor median, served median, the
    dimensionless overhead, repeats and device. No magnitude is compared to a threshold and
    nothing is written to a tracked path; the one comparison is the NESTING.

    THE TWO ARMS ARE TIMED ALTERNATELY: run in sequence they occupy different windows of
    whatever else the box is doing, and at load ~19 the same arm read 163.588 ms and 0.309 ms on
    two runs of one pair. COVERAGE: CPU only, smallest net each arch admits, every registered
    encoding the arch serves — mechanism evidence about the SEAM, not comparable to the ledger's
    production-shape readings taken on the box in bf16.
    """
    rows: list[dict] = []
    readings: dict[tuple[str, str], tuple[Measurement, Measurement]] = {}
    for kind, probe in sorted(registered_probes().items()):
        for spec in specs_for(kind):
            floor_build, floor_forward = probe.floor_arm(spec)
            served_build, served_forward = probe.served_arm(spec)
            floor, served = measure_forward_paired(
                floor_build, floor_forward, served_build, served_forward,
                repeats=_NESTING_REPEATS, warmup=_NESTING_WARMUP, device_type="cpu",
            )
            verdict = nesting_verdict(floor, served)
            readings[(kind, spec.name)] = (floor, served)
            rows.append(
                {
                    "arch_kind": kind,
                    "encoding": spec.name,
                    "floor_median_ns": floor.median_ns,
                    "served_median_ns": served.median_ns,
                    "floor_iqr_ns": floor.iqr_ns,
                    "served_iqr_ns": served.iqr_ns,
                    "noise_ns": nesting_noise_ns(floor, served),
                    "nesting": verdict,
                    "serving_overhead": (
                        serving_overhead(floor, served) if verdict == NESTING_ORDERED
                        else served.median_ns / max(floor.median_ns, 1.0)
                    ),
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
    # THE TABLE LANDS BEFORE THE REFUSAL, deliberately: a refusal that takes the evidence with
    # it leaves whoever reads the run with a row name and no numbers to act on.
    unresolved = [row for row in rows if row["nesting"] == NESTING_UNRESOLVED]
    derived("t7.measurement.unresolved", [(r["arch_kind"], r["encoding"]) for r in unresolved])
    if unresolved:
        warnings.warn(
            "the floor/served nesting is UNRESOLVED on "
            f"{[(r['arch_kind'], r['encoding']) for r in unresolved]}: each median deficit is "
            "smaller than the noise the reading carries, so the measurement supports neither "
            f"ordering. Rows: {unresolved}. Load average {os.getloadavg()} over "
            f"{os.cpu_count()} CPU(s) — the number that explains it, reported rather than "
            "guessed at. This is a host too busy to measure on and not a verdict on the arms; "
            "re-run it on a quiet box before reading anything into it.",
            ServingNestingUnresolved,
            stacklevel=2,
        )
    for row in rows:
        if row["nesting"] == NESTING_INVERTED:
            serving_overhead(*readings[(row["arch_kind"], row["encoding"])])
