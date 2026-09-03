# >300 justify (R8): the term type that makes the unit structural, the three probes that emit
# terms, and the breaks that show a missing/undeclared/underived term is caught are one unit —
# a unit rule enforced in one file against terms produced in another is enforced on nothing.
"""T8 — every arch STATES ITS MEMORY ENVELOPE: trainer, eval and serving, in the mint's units.

WHY THIS SECTION EXISTS, in the history that ordered it. The memory partition has been
re-derived BY HAND at three sittings, and it has been wrong at least twice in ways the tree
could not see: the `eval_child` term went 0.881 → 1.1855 → 3.5293 GiB across three measurements
(`src/mantis/eval/child_memory.py`, module docstring) and F-816-12 is the case where "one member
of a partition moved by a large factor and its partner kept the value fitted before the move"
(`tests/train/test_graph_microbatch_bound.py`, the `_SIZING_BUDGET_GIB` derivation comment).
Both are the same defect: the partition's terms belong to nothing that changes with the model,
so nothing re-derives them when the model changes. SEAM_V1_DESIGN §3 makes them the arch's:
per-arch trainer / eval / serving terms, emitted in the partition machinery's units, so the mint
GENERALIZES instead of being re-derived per era.

THE UNIT IS A TYPE, NOT A COMMENT. `_SIZING_BUDGET_BYTES = int(_SIZING_BUDGET_GIB * 1024 ** 3)`
is the partition's own conversion, and a term handed over as a bare float is one GiB/bytes slip
away from being 2^30 wrong in a comparison that will still look reasonable. So a term here is a
`MemoryTerm`, constructible only through `from_bytes` / `from_gib`, and an envelope that emits a
bare number is REFUSED rather than coerced. That is the whole of "in the partition machinery's
units" made mechanical.

THE BASIS IS DECLARED, for the same reason T6 labels its synthetic ladder block. A CUDA peak and
a CPU resident-storage sum are different quantities; both are useful, neither is the other, and
an undeclared basis is how the smaller one gets read as the larger. A term without a basis is
refused. **Nothing in this tier compares a term to a budget** — the budget is the operator's mint
and lives in `configs/`; this tier proves the terms EXIST, are typed, are complete against the
dispatch, and MOVE WITH THE ARCH.

WHAT IS NOT CLAIMED, stated rather than implied. The magnitudes this tier reports on CI are
CPU resident-storage sums at the smallest net each arch admits. They are NOT the mint's numbers,
they do not replace a sitting's measurement, and no mint decision may be taken from them. What
generalizes is the SHAPE: three named terms per arch, in one unit, derived from the arch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from mantis.model.arch import CnnArch, GnnArch, GnnArchV2
from mantis.model.build import build_net

from _corpus import ConformanceRefusal, roster
from test_arch_states_its_perf_floor import BUILD_SOURCE, arch_kinds_dispatched

#: The partition's own conversion, taken from the one place that performs it
#: (`tests/train/test_graph_microbatch_bound.py`: `int(_SIZING_BUDGET_GIB * 1024 ** 3)`).
#: Written once, here, so `MemoryTerm` has exactly one way to cross between the two spellings.
_BYTES_PER_GIB = 1024 ** 3

BASIS_CUDA_PEAK = "cuda_peak"
BASIS_CPU_RESIDENT = "cpu_resident"
_BASES = frozenset({BASIS_CUDA_PEAK, BASIS_CPU_RESIDENT})

#: The three terms the partition is made of, as the mint names them. Trainer and serving are the
#: run's own two device tenants; `eval` is the eval CHILD, which `child_memory.py` documents as a
#: separate process that puts a SECOND model on the card during the gate block.
REQUIRED_TERMS: tuple[str, ...] = ("trainer", "eval", "serving")


class ArchDeclaresNoMemoryEnvelope(ConformanceRefusal):
    """`build_net` dispatches an arch kind that has registered no memory envelope."""


class EnvelopeForUnknownArch(ConformanceRefusal):
    """An envelope names an arch kind `build_net` does not dispatch."""


class EnvelopeTermMissing(ConformanceRefusal):
    """An envelope is missing one of the partition's three terms, or carries a fourth."""


class EnvelopeTermNotInPartitionUnits(ConformanceRefusal):
    """A term arrived as a bare number rather than a `MemoryTerm` — the unit is unstated."""


class MemoryBasisUnstated(ConformanceRefusal):
    """A term declares no measurement basis, so a CPU sum could be read as a device peak."""


class EnvelopeTermNotDerived(ConformanceRefusal):
    """A term did not move when the arch's declared width moved — a constant wearing a term."""


@dataclass(frozen=True)
class MemoryTerm:
    """One partition term. Bytes, always, plus the basis the bytes were measured on.

    The constructors are the point: `from_gib` performs the partition's own conversion so the
    GiB spelling never reaches a comparison, and there is no way to build a term that has not
    stated which quantity it measured.
    """

    nbytes: int
    basis: str

    @staticmethod
    def from_bytes(nbytes: int, basis: str) -> MemoryTerm:
        return MemoryTerm(nbytes=int(nbytes), basis=require_basis(basis))

    @staticmethod
    def from_gib(gib: float, basis: str) -> MemoryTerm:
        return MemoryTerm(nbytes=int(gib * _BYTES_PER_GIB), basis=require_basis(basis))

    @property
    def gib(self) -> float:
        return self.nbytes / _BYTES_PER_GIB


def require_basis(basis: str) -> str:
    if basis not in _BASES:
        raise MemoryBasisUnstated(
            f"memory basis {basis!r} is not one of {sorted(_BASES)}. A CUDA peak and a CPU "
            "resident-storage sum are different quantities; an undeclared basis is how the "
            "smaller one gets read as the larger."
        )
    return basis


@dataclass(frozen=True)
class MemoryEnvelope:
    """One arch kind's three terms, as callables so nothing is measured at import."""

    arch_kind: str
    terms: dict[str, Callable[[Any], MemoryTerm]]


def resident_bytes(tensors) -> int:
    """Distinct-storage byte sum over an iterable of tensors.

    DISTINCT STORAGE, not `numel * element_size` per tensor: a view and its base share one
    allocation, and summing both double-counts an allocation that was made once. Optimizer
    state, parameters and gradients all alias in ways that make the naive sum wrong upward,
    which is the direction that would quietly widen a budget.
    """
    seen: dict[int, int] = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
    return sum(seen.values())


def check_envelope_manifest(
    dispatched: frozenset[str], registered: frozenset[str]
) -> frozenset[str]:
    """Set equality against the dispatch, both directions — T7's rule, this tier's subject.

    The census itself is T7's, imported rather than re-walked: two walkers over one dispatch is
    two authorities, and the one that is not looked at is the one that goes stale.
    """
    if not dispatched:
        raise ArchDeclaresNoMemoryEnvelope(
            f"the walk of {BUILD_SOURCE.name} found no arch branch, so the required set is "
            "empty and every manifest check passes for free."
        )
    missing = sorted(dispatched - registered)
    if missing:
        raise ArchDeclaresNoMemoryEnvelope(
            f"`build_net` dispatches {missing} and no memory envelope is registered for them. "
            "The partition's terms have been re-derived by hand at three sittings and were "
            "wrong twice; an arch that states no envelope puts the next mint back there."
        )
    stray = sorted(registered - dispatched)
    if stray:
        raise EnvelopeForUnknownArch(
            f"memory envelopes are registered for {stray}, which `build_net` does not dispatch."
        )
    return dispatched


def check_terms(arch_kind: str, terms: dict[str, Any]) -> dict[str, MemoryTerm]:
    """Exactly the partition's three terms, each a `MemoryTerm`. Set equality, not a count."""
    observed = frozenset(terms)
    required = frozenset(REQUIRED_TERMS)
    if observed != required:
        raise EnvelopeTermMissing(
            f"{arch_kind}: the envelope emits {sorted(observed)}; the partition is made of "
            f"{sorted(required)}. Missing: {sorted(required - observed)}; unknown: "
            f"{sorted(observed - required)}. A term the mint does not know about is a term "
            "nothing sums, and a missing one is the F-816-12 shape."
        )
    for name, value in terms.items():
        if not isinstance(value, MemoryTerm):
            raise EnvelopeTermNotInPartitionUnits(
                f"{arch_kind}.{name} arrived as {type(value).__name__}, not a MemoryTerm. The "
                "partition converts with `int(gib * 1024 ** 3)`; a bare number is one slip away "
                "from being 2^30 wrong in a comparison that still looks reasonable."
            )
    return dict(terms)


def check_term_moves_with_arch(
    arch_kind: str, narrow: dict[str, MemoryTerm], wide: dict[str, MemoryTerm]
) -> dict[str, int]:
    """Every term strictly grows when the arch's declared width grows.

    THE HALF A MANIFEST CANNOT GIVE. A registry proves an envelope was written; only this proves
    it is a function of the arch rather than three constants that happen to be typed correctly.
    All three terms carry the parameters, so all three must move — a term that does not is
    reported by name rather than folded into an aggregate.
    """
    flat = {
        name: wide[name].nbytes - narrow[name].nbytes
        for name in REQUIRED_TERMS
        if wide[name].nbytes <= narrow[name].nbytes
    }
    if flat:
        raise EnvelopeTermNotDerived(
            f"{arch_kind}: terms {sorted(flat)} did not grow when the declared width grew "
            f"(deltas {flat}). Every term carries the parameters, so a term that is flat under a "
            "width change is a constant wearing a term — exactly the partition member that kept "
            "its pre-move value while its partner moved."
        )
    return {name: wide[name].nbytes - narrow[name].nbytes for name in REQUIRED_TERMS}


# --------------------------------------------------------------------------------------- #
# The registered envelopes — one per arch kind `build_net` dispatches
# --------------------------------------------------------------------------------------- #
def _net_tensors(net) -> list[torch.Tensor]:
    return [*net.parameters(), *net.buffers()]


def _serving_term(net, sample: torch.Tensor) -> MemoryTerm:
    """Serving: the net's own tensors plus one served batch. No gradients, no optimizer."""
    return MemoryTerm.from_bytes(
        resident_bytes([*_net_tensors(net), sample]), BASIS_CPU_RESIDENT
    )


def _eval_term(build_one, sample: torch.Tensor) -> MemoryTerm:
    """Eval child: TWO nets, because that is what the child's gate block actually holds.

    `src/mantis/eval/child_memory.py` states it: "The GATE BLOCK is the only phase that puts a
    SECOND model and a SECOND `LocalInferenceEngine` on the card", and a round taken before the
    first promotion measures a strictly smaller term. The envelope reports the larger, posture,
    because a floor read as a term is how the 0.881 GiB figure got minted against.
    """
    candidate, opponent = build_one(), build_one()
    return MemoryTerm.from_bytes(
        resident_bytes([*_net_tensors(candidate), *_net_tensors(opponent), sample]),
        BASIS_CPU_RESIDENT,
    )


def _trainer_term(net, loss_of) -> MemoryTerm:
    """Trainer: parameters, gradients and the optimizer's own state after ONE real step.

    `optim.AdamW` is the trainer's optimizer (`train/trainer/core.py`), and its exp_avg /
    exp_avg_sq only exist AFTER a step — so the step is taken rather than the state modelled.
    A modelled multiplier here would be a second authority over what the optimizer allocates.
    """
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss_of(net).backward()
    optimizer.step()
    state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    return MemoryTerm.from_bytes(
        resident_bytes([*_net_tensors(net), *grads, *state_tensors]), BASIS_CPU_RESIDENT
    )


def _gnn_arch(spec, hidden: int, arch_cls=GnnArch):
    """The graph arch under measurement. `arch_cls` is the ONLY difference between the V1 and
    V2 envelopes — the three probes below are the arch's, not the version's."""
    return arch_cls(
        in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
        hidden=hidden, num_layers=1, policy_hidden=8, value_hidden=8,
    )


def _gnn_batch(spec):
    """One collated graph batch through the production surfaces, as T7's floor arm builds it."""
    from mantis._engine import HexgBuffer
    from mantis.selfplay.graph_collate import collate_graph_batch

    from _corpus import build_board

    board = build_board(spec.name, [(i, 0) for i in range(6)])
    legal = board.legal_moves()
    buffer = HexgBuffer(2, spec.name, 8)
    buffer.push_graph_position(
        board.get_stones(), [(legal[0][0], legal[0][1], 1.0)],
        board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8,
    )
    wire, _targets = buffer.sample_graph_batch(1, augment=False)
    batch = collate_graph_batch(
        wire, trunk_size=spec.trunk_size, win_length=spec.win_length,
        node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
    )
    stone_mask = torch.zeros(batch.x.shape[0], dtype=torch.bool)
    stone_mask[: int(batch.n_stones.sum())] = True
    return batch, stone_mask


def _gnn_envelope(hidden: int, arch_cls=GnnArch) -> dict[str, Callable[[Any], MemoryTerm]]:
    def build(spec):
        return build_net(_gnn_arch(spec, hidden, arch_cls))

    def run(net, batch, stone_mask):
        return net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.legal_node_gather, stone_mask, batch.node_offsets,
        )

    def serving(spec) -> MemoryTerm:
        batch, _mask = _gnn_batch(spec)
        return _serving_term(build(spec).eval(), batch.x)

    def evaluation(spec) -> MemoryTerm:
        batch, _mask = _gnn_batch(spec)
        return _eval_term(lambda: build(spec).eval(), batch.x)

    def trainer(spec) -> MemoryTerm:
        batch, stone_mask = _gnn_batch(spec)
        net = build(spec)
        return _trainer_term(
            net, lambda n: run(n, batch, stone_mask)[1].float().sum()
        )

    return {"trainer": trainer, "eval": evaluation, "serving": serving}


def _cnn_arch(spec, filters: int) -> CnnArch:
    return CnnArch(
        board_size=spec.board_size, in_channels=spec.n_planes,
        filters=filters, res_blocks=1, se_reduction_ratio=2,
    )


def _cnn_batch(spec) -> torch.Tensor:
    """The production dense route, as `selfplay/inference_local.py::_forward_boards` builds it."""
    from mantis._engine import Board
    from mantis.env.game_state import GameState

    board = Board.with_encoding_name(spec.name)
    for i in range(4):
        board.apply_move(i, 0)
    tensor, _centers = GameState.from_board(board).to_tensor()
    if tensor.shape[1] != spec.n_planes:
        tensor = tensor[:, list(spec.kept_plane_indices)]
    return torch.from_numpy(tensor).float()


def _cnn_envelope(filters: int) -> dict[str, Callable[[Any], MemoryTerm]]:
    def build(spec):
        return build_net(_cnn_arch(spec, filters))

    def serving(spec) -> MemoryTerm:
        return _serving_term(build(spec).eval(), _cnn_batch(spec))

    def evaluation(spec) -> MemoryTerm:
        return _eval_term(lambda: build(spec).eval(), _cnn_batch(spec))

    def trainer(spec) -> MemoryTerm:
        sample = _cnn_batch(spec)
        return _trainer_term(build(spec), lambda n: n.forward(sample)[1].float().sum())

    return {"trainer": trainer, "eval": evaluation, "serving": serving}


#: The declared widths the derivation control compares. Instrument parameters, not thresholds:
#: nothing is asserted about either level, only that every term moves BETWEEN them.
_NARROW, _WIDE = 8, 24


def registered_envelopes(narrow: bool = False) -> dict[str, MemoryEnvelope]:
    width = _NARROW if narrow else _WIDE
    return {
        "GnnArch": MemoryEnvelope("GnnArch", _gnn_envelope(width)),
        "GnnArchV2": MemoryEnvelope("GnnArchV2", _gnn_envelope(width, GnnArchV2)),
        "CnnArch": MemoryEnvelope("CnnArch", _cnn_envelope(width)),
    }


def specs_for(arch_kind: str) -> tuple[Any, ...]:
    """EVERY registered encoding whose representation this arch serves, NAME-SORTED.

    AUDIT-1 F-41. This returned the FIRST match in `all_specs()` order, so the table measured
    `gnn_axis_v1` and never `gnn_axis_r8` — run6's own identity — and a roster REORDER would
    have changed which encoding the mint's memory partition was derived from without changing
    a line of this file. Name-sorted so the order is a property of the registry's contents
    rather than of its declaration order.

    Raises:
        ArchDeclaresNoMemoryEnvelope: no registered encoding serves this arch's representation.
    """
    graph = arch_kind.startswith("GnnArch")
    matches = tuple(sorted((s for s in roster() if bool(s.is_graph) is graph),
                           key=lambda s: s.name))
    if not matches:
        raise ArchDeclaresNoMemoryEnvelope(
            f"{arch_kind}: no registered encoding carries the representation this arch serves."
        )
    return matches


#: Every (arch kind, encoding) pair the envelope is defined over — the parametrisation roster,
#: derived from the registry so a new row joins every arm below with no test edit.
ARCH_SPEC_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (kind, spec.name)
    for kind in sorted(registered_envelopes())
    for spec in specs_for(kind)
)


def spec_by_name(arch_kind: str, encoding: str):
    return next(s for s in specs_for(arch_kind) if s.name == encoding)


def envelope_terms(arch_kind: str, encoding: str, narrow: bool = False) -> dict[str, MemoryTerm]:
    envelope = registered_envelopes(narrow=narrow)[arch_kind]
    spec = spec_by_name(arch_kind, encoding)
    return check_terms(arch_kind, {name: fn(spec) for name, fn in envelope.terms.items()})


# --------------------------------------------------------------------------------------- #
# The manifest, the unit rule and the derivation control — ALL DEFAULT TIER
# --------------------------------------------------------------------------------------- #
def test_EVERY_arch_build_net_dispatches_HAS_a_registered_memory_envelope(derived):
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    registered = frozenset(registered_envelopes())
    derived("t8.envelopes.registered", sorted(registered))
    assert check_envelope_manifest(dispatched, registered) == dispatched


def test_a_MISSING_envelope_is_refused_by_name():
    """PB-T8a. The state a third arch lands in until it states an envelope."""
    with pytest.raises(ArchDeclaresNoMemoryEnvelope, match="GnnArchNext"):
        check_envelope_manifest(
            frozenset({*registered_envelopes(), "GnnArchNext"}),
            frozenset(registered_envelopes()),
        )


def test_an_envelope_for_an_arch_NOTHING_dispatches_is_refused():
    """PB-T8b. The reverse direction, so a stray envelope cannot pad the count."""
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    with pytest.raises(EnvelopeForUnknownArch, match="GhostArch"):
        check_envelope_manifest(dispatched, frozenset(registered_envelopes()) | {"GhostArch"})


def test_an_EMPTY_dispatch_census_is_refused_rather_than_reported_clean():
    with pytest.raises(ArchDeclaresNoMemoryEnvelope, match="passes for free"):
        check_envelope_manifest(frozenset(), frozenset(registered_envelopes()))


@pytest.mark.parametrize(("arch_kind", "encoding"), ARCH_SPEC_PAIRS)
def test_the_envelope_emits_EXACTLY_the_partitions_three_terms(arch_kind, encoding, derived):
    terms = envelope_terms(arch_kind, encoding)
    derived(f"t8.{arch_kind}.{encoding}.terms", {n: t.nbytes for n, t in terms.items()})
    assert frozenset(terms) == frozenset(REQUIRED_TERMS)


def test_a_MISSING_term_is_refused():
    """PB-T8c. Two of three is the F-816-12 shape: a partition whose members do not all move."""
    with pytest.raises(EnvelopeTermMissing, match="Missing: \\['serving'\\]"):
        check_terms(
            "GnnArch",
            {
                "trainer": MemoryTerm.from_bytes(1, BASIS_CPU_RESIDENT),
                "eval": MemoryTerm.from_bytes(1, BASIS_CPU_RESIDENT),
            },
        )


def test_a_FOURTH_term_the_mint_does_not_know_is_refused():
    """The other direction of the same set equality — a term nothing sums."""
    with pytest.raises(EnvelopeTermMissing, match="unknown: \\['scratch'\\]"):
        check_terms(
            "GnnArch",
            {name: MemoryTerm.from_bytes(1, BASIS_CPU_RESIDENT) for name in REQUIRED_TERMS}
            | {"scratch": MemoryTerm.from_bytes(1, BASIS_CPU_RESIDENT)},
        )


def test_a_BARE_NUMBER_term_is_refused_rather_than_coerced():
    """PB-T8d. The unit rule, and the reason it is a type: `9.431` is a plausible GiB figure and
    a catastrophic byte figure, and nothing about the float says which it is."""
    with pytest.raises(EnvelopeTermNotInPartitionUnits, match="2\\^30 wrong"):
        check_terms(
            "GnnArch",
            {
                "trainer": 9.431,
                "eval": MemoryTerm.from_bytes(1, BASIS_CPU_RESIDENT),
                "serving": MemoryTerm.from_bytes(1, BASIS_CPU_RESIDENT),
            },
        )


def test_an_UNDECLARED_basis_is_refused():
    """PB-T8e. A CPU sum and a device peak are different quantities; the label is what keeps the
    smaller from being read as the larger, exactly as T6 labels its synthetic ladder block."""
    assert MemoryTerm.from_bytes(1, BASIS_CUDA_PEAK).basis == BASIS_CUDA_PEAK
    with pytest.raises(MemoryBasisUnstated, match="undeclared basis"):
        MemoryTerm.from_bytes(1, "whatever")


def test_the_GIB_constructor_performs_the_partitions_own_conversion():
    """The unit rule's positive half. `from_gib` must be `int(gib * 1024 ** 3)` and nothing else,
    because the partition's budget line is written that way and a second conversion here would
    be a second authority over how many bytes a GiB is."""
    assert MemoryTerm.from_gib(1.0, BASIS_CPU_RESIDENT).nbytes == 1024 ** 3
    assert MemoryTerm.from_gib(9.431, BASIS_CPU_RESIDENT).nbytes == int(9.431 * 1024 ** 3)
    assert MemoryTerm.from_bytes(1024 ** 3, BASIS_CPU_RESIDENT).gib == 1.0


@pytest.mark.parametrize(("arch_kind", "encoding"), ARCH_SPEC_PAIRS)
def test_EVERY_term_MOVES_when_the_archs_declared_width_moves(arch_kind, encoding, derived):
    """The derivation control. This is the half that would have caught F-816-12: a term that
    stays put while the model changes is not a term, whatever it is typed as."""
    narrow = envelope_terms(arch_kind, encoding, narrow=True)
    wide = envelope_terms(arch_kind, encoding, narrow=False)
    deltas = check_term_moves_with_arch(arch_kind, narrow, wide)
    derived(f"t8.{arch_kind}.{encoding}.width_deltas", deltas)
    assert all(delta > 0 for delta in deltas.values())


def test_a_FLAT_term_is_refused_by_name():
    """PB-T8f. Drive the control with a stand-in whose serving term does not move."""
    narrow = {name: MemoryTerm.from_bytes(10, BASIS_CPU_RESIDENT) for name in REQUIRED_TERMS}
    wide = dict(narrow)
    wide["trainer"] = MemoryTerm.from_bytes(20, BASIS_CPU_RESIDENT)
    wide["eval"] = MemoryTerm.from_bytes(20, BASIS_CPU_RESIDENT)
    with pytest.raises(EnvelopeTermNotDerived, match="'serving'"):
        check_term_moves_with_arch("GnnArch", narrow, wide)


def test_the_derivation_control_does_NOT_fire_when_every_term_moves():
    """Negative control. A control that fires on a correct envelope measures nothing."""
    narrow = {name: MemoryTerm.from_bytes(10, BASIS_CPU_RESIDENT) for name in REQUIRED_TERMS}
    wide = {name: MemoryTerm.from_bytes(11, BASIS_CPU_RESIDENT) for name in REQUIRED_TERMS}
    assert check_term_moves_with_arch("GnnArch", narrow, wide) == dict.fromkeys(REQUIRED_TERMS, 1)


def test_resident_bytes_COUNTS_A_SHARED_STORAGE_ONCE():
    """The accountant's own guard. A view and its base are one allocation; summing both inflates
    the term, and inflation is the direction that quietly widens a budget."""
    base = torch.zeros(1024, dtype=torch.float32)
    view = base[:512]
    assert resident_bytes([base]) == base.untyped_storage().nbytes()
    assert resident_bytes([base, view]) == resident_bytes([base])
    other = torch.zeros(1024, dtype=torch.float32)
    assert resident_bytes([base, other]) == 2 * resident_bytes([base])


# --------------------------------------------------------------------------------------- #
# The MEASUREMENT (`slow`) — a table, no budget comparison
# --------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_report_the_per_arch_memory_envelope(derived):
    """The table the mint would generalize from. Bytes and GiB per term per arch, with the basis
    on every row. NOTHING is compared to a budget, and no row is a mint input: these are CPU
    resident sums at the smallest net each arch admits, and the module docstring says so."""
    rows = [
        {
            "arch_kind": kind,
            "encoding": encoding,
            "term": name,
            "bytes": term.nbytes,
            "gib": term.gib,
            "basis": term.basis,
        }
        for kind, encoding in ARCH_SPEC_PAIRS
        for name, term in envelope_terms(kind, encoding).items()
    ]
    derived("t8.measurement.rows", rows)
    assert {(row["arch_kind"], row["encoding"], row["term"]) for row in rows} == {
        (kind, encoding, term) for kind, encoding in ARCH_SPEC_PAIRS for term in REQUIRED_TERMS
    }, "the measured table does not carry every arch's every term on every encoding it serves"
    assert {row["basis"] for row in rows} == {BASIS_CPU_RESIDENT}, (
        "a row carries a basis this run did not measure on"
    )
