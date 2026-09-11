# >300 justify (R8): the term type that makes the unit structural, the three probes that emit
# terms, and the breaks that show a missing/undeclared/underived term is caught are one unit —
# a unit rule enforced in one file against terms produced in another is enforced on nothing.
"""Every arch STATES ITS MEMORY ENVELOPE: trainer, eval and serving, in the mint's units.

The memory partition had been re-derived BY HAND at three sittings and was wrong at least twice
invisibly — the eval-child term went 0.881 → 1.1855 → 3.5293 GiB, and one member of a partition
moved by a large factor while its partner kept the value fitted before the move. The terms
belonged to nothing that changes with the model; making them the arch's is what generalizes.

THE UNIT IS A TYPE, NOT A COMMENT: a bare float is one GiB/bytes slip from being 2^30 wrong in a
comparison that still looks reasonable. THE BASIS IS DECLARED, because an undeclared one is how a
CPU sum gets read as a device peak. NOTHING HERE COMPARES A TERM TO A BUDGET — the CI magnitudes
are CPU resident sums at the smallest net each arch admits, and what generalizes is the SHAPE.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from mantis.model.arch import GnnArch, GnnArchV2
from mantis.model.build import build_net

from _corpus import ConformanceRefusal, roster
from test_arch_states_its_perf_floor import BUILD_SOURCE, arch_kinds_dispatched

#: The partition's own conversion, taken from the one place that performs it and written once
#: here, so `MemoryTerm` has exactly one way to cross between the two spellings.
_BYTES_PER_GIB = 1024 ** 3

BASIS_CUDA_PEAK = "cuda_peak"
BASIS_CPU_RESIDENT = "cpu_resident"
_BASES = frozenset({BASIS_CUDA_PEAK, BASIS_CPU_RESIDENT})

#: The three terms the partition is made of, as the mint names them. Trainer and serving are the
#: run's own two device tenants; `eval` is the eval CHILD, a separate process that puts a SECOND
#: model on the card during the gate block.
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
    """One partition term: bytes, always, plus the basis they were measured on. No term can be
    built without stating which quantity it measured."""

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
    """Distinct-storage byte sum: a view and its base share one allocation, and optimizer state,
    parameters and gradients alias in ways that make the naive sum wrong UPWARD."""
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
    """Set equality against the dispatch, both directions. The census is imported rather than
    re-walked: two walkers over one dispatch is two authorities, and the one nobody looks at
    goes stale."""
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
    """Every term strictly grows when the arch's declared width grows — the half a manifest cannot
    give: only this proves the envelope is a function of the arch rather than three constants."""
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


# ── the registered envelopes — one per arch kind `build_net` dispatches ────────────────
def _net_tensors(net) -> list[torch.Tensor]:
    return [*net.parameters(), *net.buffers()]


def _serving_term(net, sample: torch.Tensor) -> MemoryTerm:
    """Serving: the net's own tensors plus one served batch. No gradients, no optimizer."""
    return MemoryTerm.from_bytes(
        resident_bytes([*_net_tensors(net), sample]), BASIS_CPU_RESIDENT
    )


def _eval_term(build_one, sample: torch.Tensor) -> MemoryTerm:
    """Eval child: TWO nets, because the gate block is the only phase that puts a SECOND model on
    the card. A floor read as a term is how the 0.881 GiB figure got minted against."""
    candidate, opponent = build_one(), build_one()
    return MemoryTerm.from_bytes(
        resident_bytes([*_net_tensors(candidate), *_net_tensors(opponent), sample]),
        BASIS_CPU_RESIDENT,
    )


def _trainer_term(net, loss_of) -> MemoryTerm:
    """Trainer: parameters, gradients and optimizer state after ONE real step, because AdamW's
    exp_avg / exp_avg_sq exist only after one and a modelled multiplier would be a second
    authority over what the optimizer allocates."""
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
    """The graph arch under measurement. `arch_cls` is the ONLY difference between the two
    envelopes — the three probes are the arch's, not the version's."""
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
    # The probe nets are built on CPU; the collate's device is stated so a CUDA torch cannot
    # land the batch on another device than the net (the device is a declared fact, never sniffed).
    batch = collate_graph_batch(
        wire, trunk_size=spec.trunk_size, win_length=spec.win_length,
        node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim, device="cpu",
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


#: The declared widths the derivation control compares. Instrument parameters, not thresholds:
#: nothing is asserted about either level, only that every term moves BETWEEN them.
_NARROW, _WIDE = 8, 24


def registered_envelopes(narrow: bool = False) -> dict[str, MemoryEnvelope]:
    width = _NARROW if narrow else _WIDE
    return {
        "GnnArch": MemoryEnvelope("GnnArch", _gnn_envelope(width)),
        "GnnArchV2": MemoryEnvelope("GnnArchV2", _gnn_envelope(width, GnnArchV2)),
    }


def specs_for(arch_kind: str) -> tuple[Any, ...]:
    """EVERY registered encoding whose representation this arch serves, NAME-SORTED — it returned
    the FIRST match in roster order, so a REORDER would have changed which encoding the partition
    was derived from without changing a line here.

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


#: Every (arch kind, encoding) pair the envelope is defined over, derived from the registry so a
#: new row joins every arm below with no test edit.
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


# ── the manifest, the unit rule and the derivation control — ALL DEFAULT TIER ─────────
def test_EVERY_arch_build_net_dispatches_HAS_a_registered_memory_envelope(derived):
    dispatched = arch_kinds_dispatched(BUILD_SOURCE)
    registered = frozenset(registered_envelopes())
    derived("t8.envelopes.registered", sorted(registered))
    assert check_envelope_manifest(dispatched, registered) == dispatched


def test_a_MISSING_envelope_is_refused_by_name():
    """The state a third arch lands in until it states an envelope."""
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
    """Two of three is the shape of a partition whose members do not all move."""
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
    """The unit rule, and why it is a type: `9.431` is a plausible GiB figure and a catastrophic
    byte figure, and nothing about the float says which it is."""
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
    """A CPU sum and a device peak are different quantities; the label is what keeps the smaller
    from being read as the larger."""
    assert MemoryTerm.from_bytes(1, BASIS_CUDA_PEAK).basis == BASIS_CUDA_PEAK
    with pytest.raises(MemoryBasisUnstated, match="undeclared basis"):
        MemoryTerm.from_bytes(1, "whatever")


def test_the_GIB_constructor_performs_the_partitions_own_conversion():
    """`from_gib` must be `int(gib * 1024 ** 3)` and nothing else: a second conversion would be a
    second authority over how many bytes a GiB is."""
    assert MemoryTerm.from_gib(1.0, BASIS_CPU_RESIDENT).nbytes == 1024 ** 3
    assert MemoryTerm.from_gib(9.431, BASIS_CPU_RESIDENT).nbytes == int(9.431 * 1024 ** 3)
    assert MemoryTerm.from_bytes(1024 ** 3, BASIS_CPU_RESIDENT).gib == 1.0


@pytest.mark.parametrize(("arch_kind", "encoding"), ARCH_SPEC_PAIRS)
def test_EVERY_term_MOVES_when_the_archs_declared_width_moves(arch_kind, encoding, derived):
    """The derivation control: a term that stays put while the model changes is not a term,
    whatever it is typed as."""
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
    """A view and its base are one allocation; summing both inflates the term, and inflation is
    the direction that quietly widens a budget."""
    base = torch.zeros(1024, dtype=torch.float32)
    view = base[:512]
    assert resident_bytes([base]) == base.untyped_storage().nbytes()
    assert resident_bytes([base, view]) == resident_bytes([base])
    other = torch.zeros(1024, dtype=torch.float32)
    assert resident_bytes([base, other]) == 2 * resident_bytes([base])


# ── the MEASUREMENT (`slow`) — a table, no budget comparison ──────────────────────────
@pytest.mark.slow
def test_report_the_per_arch_memory_envelope(derived):
    """The table the mint would generalize from — bytes and GiB per term per arch, with the basis
    on every row. No row is a mint input."""
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
