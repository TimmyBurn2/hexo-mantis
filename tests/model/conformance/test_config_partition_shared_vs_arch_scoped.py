# >300 justify (R8): the partition's coverage checks, the two reachability probes that execute
# it, and the planted breaks that prove each probe can fire are one unit — a probe whose
# falsifier lives in another file is a probe nobody re-runs against its own break.
"""T9 — the config partition: SHARED keys vs ARCH-SCOPED keys, and who can reach the latter.

SEAM_V1_DESIGN §3: "The schema splits shared vs arch-scoped. **An arch-scoped key reachable
outside its arch is a red row.** This kills the config-blind class structurally — the defect we
have paid for four times."

WHAT B1 FOUND AND B2 REPAIRED (R322(d)). B1 declared eight red rows — four arch-scoped keys
(`train.microbatch_caps.{max_edges,max_nodes}`, `inference.fused_graph_caps.{max_fused_edges,
max_fused_nodes}`) times two reachability classes — because the two grid configs the repo ships
carried both graph-only cap blocks, counted in EDGES and NODES, at byte-identical values. That
was not laxity: `RunConfig` is `extra="forbid"` with every key required, so a grid run was
REQUIRED to carry them and `tools/mint_config.py` had to write a number for a quantity that run
has none of. B2 repaired it in the two places the two classes name, and **the eight rows are
DELETED here by that repair rather than widened** — a ratchet that only ever grows is a list.

THE REPAIR, and this file executes both halves rather than reading them off the source:

  * SCHEMA — `mantis.config.schema.core.ARCH_SCOPED_KEYS` is the ONE authority on which block
    belongs to which representation, and `RunConfig._arch_scoped_keys_are_present_iff_their_arch`
    makes each block REQUIRED on its own arch and REFUSED on any other. Presence is read off
    `model_fields_set`, so an explicit `null` is carrying the key, not omitting it.
  * READ PATH — each block's ONE resolver calls `refuse_outside_its_arch` BEFORE it looks for
    the block, so it refuses a foreign config BY NAME instead of by accident. A resolver that
    refuses only on ABSENCE is green because the key happens not to be there, and turns red the
    moment anyone re-adds it.

WHAT WAS ALREADY CLEAN, said precisely so the finding above reads as a finding and not a smear.
The CALL SITES were arch-gated before B2 and were gated deliberately: `run.py` resolves
`fused_graph_caps` only `if config.identity.representation == "graph"`, with the reason written
beside it, and `coordinator/step.py` hands `microbatch_caps` to the GRAPH arm as a lazy thunk
for the same reason. The gate was one `if` at one call site; the class this tier exists for is
the one where the NEXT call site forgets it, and that is the half B2 moved into the schema.

TWO RED CLASSES, both EXECUTED against real minted files, never inferred from source text:

  * `schema_requires_outside_arch` — a config that selects the OTHER representation validates,
    through the one loader, carrying this key.
  * `read_path_serves_outside_arch` — the key's ONE read path, handed that same config, RETURNS
    a value instead of refusing.

The declared set is now EMPTY, and the ratchet is still asserted in BOTH directions: a new red
row fails, and a declared row that has gone green ALSO fails. An empty expectation is the one
place a set-equality check can go vacuous, so this file adds the guard that shape needs — the
probes must be shown to still EXECUTE, and each one is driven against a planted break.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from mantis.config.loader import load_config
from mantis.config.resolve.arch_scope import (
    ArchScopedKeyOutsideItsArchError,
    declared_representation,
    refuse_outside_its_arch,
)
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.config.schema.core import ARCH_SCOPED_KEYS, RunConfig

from _corpus import ConformanceRefusal

CONFIGS = Path(__file__).resolve().parents[3] / "configs"

REPRESENTATIONS: tuple[str, ...] = ("grid", "graph")

SCHEMA_REQUIRES = "schema_requires_outside_arch"
READ_PATH_SERVES = "read_path_serves_outside_arch"
RED_CLASSES = frozenset({SCHEMA_REQUIRES, READ_PATH_SERVES})


class PartitionKeyRetired(ConformanceRefusal):
    """A key the partition places no longer exists on `RunConfig` — the declaration is stale."""


class ArchVocabularyKeyUnplaced(ConformanceRefusal):
    """A live key whose name carries arch vocabulary is in neither half of the partition."""


class RedRowAppeared(ConformanceRefusal):
    """An arch-scoped key is reachable outside its arch and was not declared as such."""


class RedRowRepairedButStillDeclared(ConformanceRefusal):
    """A declared red row is green now; the declaration outlived the defect it recorded."""


class ProbeWentVacuous(ConformanceRefusal):
    """A reachability probe ran against nothing, so its empty result means nothing."""


@dataclass(frozen=True)
class ArchScopedLeaf:
    """One LEAF key that belongs to exactly one representation, with its one read path."""

    path: str
    arch: str
    read_path: Any


#: THE READ PATH PER ARCH-SCOPED BLOCK. The only thing this file still declares about the
#: partition, because a resolver is not discoverable from a pydantic model: the block-to-arch
#: judgment now has a producer (`ARCH_SCOPED_KEYS`) and is READ from it, while "which function
#: is this block's one read path" has none. Both directions are checked below — every registry
#: entry must appear here, and every entry here must be in the registry.
READ_PATHS: dict[tuple[str, str], Any] = {
    ("train", "microbatch_caps"): resolve_microbatch_caps,
    ("inference", "fused_graph_caps"): resolve_fused_graph_caps,
}


def arch_scoped_leaves() -> tuple[ArchScopedLeaf, ...]:
    """Every arch-scoped LEAF, DERIVED from the schema registry and the live block models.

    B1 declared these four leaves by hand and argued the declaration, because there was no
    producer in the tree to read the block-to-arch judgment off. B2's repair creates that
    producer — `ARCH_SCOPED_KEYS` is what the schema itself enforces against — so the
    declaration becomes a derivation and the suite can no longer disagree with the schema
    about which keys are scoped.

    Raises:
        PartitionKeyRetired: a registry entry names a section or field `RunConfig` does not
            carry, or a block model with no leaves at all.
    """
    out: list[ArchScopedLeaf] = []
    for key in ARCH_SCOPED_KEYS:
        section = RunConfig.model_fields.get(key.section)
        if section is None:
            raise PartitionKeyRetired(f"ARCH_SCOPED_KEYS names section {key.section!r}, "
                                      "which RunConfig does not carry")
        field = section.annotation.model_fields.get(key.field)  # type: ignore[union-attr]
        if field is None:
            raise PartitionKeyRetired(f"ARCH_SCOPED_KEYS names {key.section}.{key.field}, "
                                      "which that section does not carry")
        block = next(a for a in getattr(field.annotation, "__args__", (field.annotation,))
                     if isinstance(a, type) and issubclass(a, BaseModel))
        if not block.model_fields:
            raise PartitionKeyRetired(f"{key.section}.{key.field} has no leaf members, so "
                                      "every reachability check over it passes for free")
        for member in block.model_fields:
            out.append(ArchScopedLeaf(f"{key.section}.{key.field}.{member}", key.arch,
                                      READ_PATHS[(key.section, key.field)]))
    return tuple(out)


#: The arch vocabulary a key name can carry. Matched on the leaf path so a NEW key called
#: `train.gnn_hidden` or `inference.max_graph_batch` cannot slip into the shared half silently.
#: This is a PROMPT, not a verdict: a match means the key must be placed on one side or the
#: other, and the placement is what says which. It stays an INDEPENDENT instrument after the
#: repair — `ARCH_SCOPED_KEYS` is a declaration too, and a key missing from it would be
#: invisible to the schema and to this suite alike if the probe did not fire on the name.
_ARCH_VOCABULARY = re.compile(
    r"(gnn|graph|edge|node|cluster|plane|filters|res_block|se_reduction|window|augment|"
    r"representation|encoding)",
    re.IGNORECASE,
)

#: SHARED DESPITE THE NAME, each with its grounds. Every row here is a homonym: the word that
#: fires the vocabulary probe means something other than an architecture in this key.
SHARED_DESPITE_THE_NAME: dict[str, str] = {
    "identity.encoding": "the SELECTOR itself — it names which arch a run is, so it is the one "
                         "key that must be readable on every arch",
    "identity.representation": "the selector's other half, for the same reason",
    "train.augment": "symmetry augmentation is a data-pipeline posture; both representations "
                     "have an augmentation path and run5 mints it false on the graph one",
    "selfplay.solver_node_budget": "SEARCH nodes, not graph nodes — the tactical solver's "
                                   "budget, which has no representation in it at all",
    "monitor.alert_loss_increase_window": "a TIME window over training steps, not a board "
                                          "window; the K-cluster window is a different word",
}

#: THE RED ROWS. **EMPTY, and emptied BY THE REPAIR** (R322(d)) — B1's eight rows were
#: `{(leaf, class) for leaf in the four leaves for class in the two classes}` and every one of
#: them is now green, so the declaration goes with the defect. Ratcheted in BOTH directions
#: below; the vacuity guard is what keeps an empty expectation from being a free pass.
DECLARED_RED_ROWS: frozenset[tuple[str, str]] = frozenset()


def live_leaf_paths(model: type[BaseModel] = RunConfig, prefix: str = "") -> tuple[str, ...]:
    """Every leaf key of the shipped schema, walked off `RunConfig` itself.

    The same walk `tools/ci_gates/contract_doc_gate.py` performs, and for the same stated
    reason: a transcribed key list is written in the commit that adds a key and therefore can
    never be the thing that notices one. An arch-scoped block is `Block | None`, so the walk
    descends through the union arm rather than stopping at the `None` — otherwise the repair
    would hide the very leaves this section is about.
    """
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = field.annotation
        arms = getattr(annotation, "__args__", (annotation,))
        nested = next((a for a in arms if isinstance(a, type) and issubclass(a, BaseModel)), None)
        if nested is not None:
            out.extend(live_leaf_paths(nested, path + "."))
        else:
            out.append(path)
    return tuple(out)


def config_for(representation: str) -> Path:
    """A shipped, minted config that selects `representation`. Read off the files, not named."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        if load_config(path).identity.representation == representation:
            return path
    raise ArchVocabularyKeyUnplaced(
        f"no shipped config selects representation={representation!r}, so the cross-arch "
        "reachability of an arch-scoped key cannot be executed against a real minted file"
    )


def other_arch(arch: str) -> str:
    return next(rep for rep in REPRESENTATIONS if rep != arch)


def leaf_present(config_dump: dict, path: str) -> bool:
    node: Any = config_dump
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def observed_red_rows() -> frozenset[tuple[str, str]]:
    """EXECUTE both red classes for every arch-scoped leaf. Nothing here reads source text.

    `schema_requires_outside_arch` loads a real minted config of the OTHER representation
    through the one loader and asks whether the leaf came through validation.
    `read_path_serves_outside_arch` hands that same config to the leaf's ONE read path and asks
    whether it answered.

    Raises:
        ProbeWentVacuous: there is no arch-scoped leaf to probe, or a foreign config that
            declares no representation — either makes an empty result meaningless.
    """
    leaves = arch_scoped_leaves()
    if not leaves:
        raise ProbeWentVacuous(
            "no arch-scoped leaf exists, so both red classes are empty for free"
        )
    rows: set[tuple[str, str]] = set()
    for leaf in leaves:
        foreign = load_config(config_for(other_arch(leaf.arch)))
        dump = foreign.model_dump()
        if declared_representation(dump) is None:
            raise ProbeWentVacuous(
                f"the foreign config for {leaf.path} declares no readable "
                "identity.representation, so the read path's arch refusal cannot fire and a "
                "green result would mean nothing"
            )
        if leaf_present(dump, leaf.path):
            rows.add((leaf.path, SCHEMA_REQUIRES))
        try:
            leaf.read_path(dump)
        except Exception:  # noqa: BLE001 — any refusal at all is the green outcome here
            pass
        else:
            rows.add((leaf.path, READ_PATH_SERVES))
    return frozenset(rows)


def check_red_row_ratchet(
    observed: frozenset[tuple[str, str]], declared: frozenset[tuple[str, str]]
) -> frozenset[tuple[str, str]]:
    """Set equality, both directions. Neither half is optional and they catch opposite things."""
    appeared = sorted(observed - declared)
    if appeared:
        raise RedRowAppeared(
            f"arch-scoped keys reachable outside their arch that B2 did not declare: {appeared}. "
            "An arch-scoped key reachable outside its arch is a red row (SEAM_V1_DESIGN §3) — "
            "either scope the key in `ARCH_SCOPED_KEYS` so the schema and both read paths "
            "refuse it, or place it in the shared half with grounds."
        )
    repaired = sorted(declared - observed)
    if repaired:
        raise RedRowRepairedButStillDeclared(
            f"declared red rows that are GREEN now: {repaired}. The declaration outlived the "
            "defect. Delete the row in the same commit as the repair — a ratchet that only ever "
            "grows is a list, and a list nobody deletes from stops describing the tree."
        )
    return observed


def check_partition_covers_the_live_schema(
    live: tuple[str, ...], scoped: tuple[str, ...], shared: frozenset[str]
) -> frozenset[str]:
    """Every live key carrying arch vocabulary is placed, and every placed key is still live."""
    live_set = frozenset(live)
    if not live_set:
        raise PartitionKeyRetired(
            "the walk of RunConfig returned no leaf key, so every coverage check below passes "
            "for free"
        )
    retired = sorted((frozenset(scoped) | shared) - live_set)
    if retired:
        raise PartitionKeyRetired(
            f"the partition places {retired}, which `RunConfig` no longer carries. A stale "
            "placement is a claim about a key that cannot be wrong because it cannot be read."
        )
    flagged = frozenset(path for path in live_set if _ARCH_VOCABULARY.search(path))
    unplaced = sorted(flagged - frozenset(scoped) - shared)
    if unplaced:
        raise ArchVocabularyKeyUnplaced(
            f"live keys carrying arch vocabulary and placed in neither half: {unplaced}. Place "
            "each one: arch-scoped in `ARCH_SCOPED_KEYS` with its grounds, or shared WITH "
            "GROUNDS saying which homonym fired the probe. Defaulting a new graph-only key into "
            "the shared half is exactly how the four config-blind defects arrived."
        )
    return flagged


# --------------------------------------------------------------------------------------- #
# Coverage — the partition against the live schema
# --------------------------------------------------------------------------------------- #
def test_the_partition_covers_every_live_key_that_carries_arch_vocabulary(derived):
    live = live_leaf_paths()
    scoped = tuple(leaf.path for leaf in arch_scoped_leaves())
    shared = frozenset(SHARED_DESPITE_THE_NAME)
    flagged = check_partition_covers_the_live_schema(live, scoped, shared)
    derived("t9.live_leaf_count", len(live))
    derived("t9.arch_vocabulary_hits", sorted(flagged))
    derived("t9.arch_scoped", sorted(scoped))
    assert flagged, "the vocabulary probe matched no live key at all — it is prompting nobody"


def test_the_read_path_table_and_the_schema_registry_agree_in_BOTH_directions(derived):
    """The one declaration this file still makes, pinned against the one the schema makes.

    A read path table that outlives its registry entry would keep probing a key nobody scopes;
    a registry entry with no read path would silently drop a block out of the second red class.
    """
    registry = {(key.section, key.field) for key in ARCH_SCOPED_KEYS}
    derived("t9.registry_blocks", sorted(registry))
    assert registry == set(READ_PATHS), (
        f"ARCH_SCOPED_KEYS declares {sorted(registry)} and this file names read paths for "
        f"{sorted(READ_PATHS)}; every scoped block needs its one read path and every read path "
        "needs its scope."
    )


def test_a_NEW_arch_vocabulary_key_lands_UNPLACED_rather_than_shared():
    """PB-T9a. The generalisation this section exists for: GnnNetV2's own config keys.

    A key named `train.gnn_v2_hidden` must not become a shared key by arriving. It is refused
    until someone says which half it is in, which is the only moment anyone will think about it.
    """
    live = (*live_leaf_paths(), "train.gnn_v2_hidden")
    with pytest.raises(ArchVocabularyKeyUnplaced, match="gnn_v2_hidden"):
        check_partition_covers_the_live_schema(
            live,
            tuple(leaf.path for leaf in arch_scoped_leaves()),
            frozenset(SHARED_DESPITE_THE_NAME),
        )


def test_a_RETIRED_placement_is_refused_rather_than_carried():
    """PB-T9b. The other direction of coverage — a placement whose key is gone."""
    with pytest.raises(PartitionKeyRetired, match="train.gone_key"):
        check_partition_covers_the_live_schema(
            live_leaf_paths(),
            (*[leaf.path for leaf in arch_scoped_leaves()], "train.gone_key"),
            frozenset(SHARED_DESPITE_THE_NAME),
        )


def test_an_EMPTY_live_walk_is_refused_rather_than_reported_clean():
    with pytest.raises(PartitionKeyRetired, match="passes for free"):
        check_partition_covers_the_live_schema((), (), frozenset())


def test_the_vocabulary_probe_does_NOT_fire_on_an_ORDINARY_key():
    """Negative control. A probe widened until it flags `train.batch_size` would prompt on every
    key, and a prompt that fires everywhere is one nobody reads."""
    for ordinary in ("train.batch_size", "selfplay.n_workers", "eval.n_games", "seed"):
        assert not _ARCH_VOCABULARY.search(ordinary), ordinary


def test_every_SHARED_DESPITE_THE_NAME_row_states_its_grounds():
    """A row here suppresses a prompt, so an ungrounded one is indistinguishable from an
    oversight waved through — the same standard gate 17's exemptions are held to."""
    for path, grounds in SHARED_DESPITE_THE_NAME.items():
        assert path in live_leaf_paths(), f"{path} is not a live key"
        assert len(grounds.split()) >= 5, f"{path}: the grounds do not say which homonym fired"


def test_every_ARCH_SCOPED_KEYS_row_states_its_grounds():
    """The same standard applied to the schema's own half of the partition. A scoping with no
    grounds is a claim about what a key MEANS with nothing behind it."""
    assert ARCH_SCOPED_KEYS, "the schema scopes no key at all, so the partition has one half"
    for key in ARCH_SCOPED_KEYS:
        assert len(key.grounds.split()) >= 5, (
            f"{key.section}.{key.field}: the grounds do not say why the key means something "
            "only on this arch"
        )


# --------------------------------------------------------------------------------------- #
# Reachability — the two red classes, executed
# --------------------------------------------------------------------------------------- #
def test_the_red_row_set_matches_what_B2_declared(derived):
    """The ratchet. GREEN at HEAD: B1 landed the enforcement, B2 landed the repair (R322(d))."""
    observed = observed_red_rows()
    derived("t9.red_rows.observed", sorted(observed))
    derived("t9.red_rows.declared", sorted(DECLARED_RED_ROWS))
    assert check_red_row_ratchet(observed, DECLARED_RED_ROWS) == observed


def test_the_probes_are_still_EXECUTING_and_not_merely_empty(derived):
    """The guard an EMPTY expectation needs, and the one B1 did not need.

    While red rows were declared, the ratchet could not pass vacuously: an empty observation
    would have failed the shrink direction. With the declaration empty, "no red rows" and "the
    probe never ran" produce the identical result, so the probe's own subject is asserted
    here — leaves exist, both red classes are named, and each leaf's foreign config is a real
    minted file that declares a representation for the arch refusal to bite on.
    """
    leaves = arch_scoped_leaves()
    derived("t9.arch_scoped_leaf_count", len(leaves))
    assert leaves, "no arch-scoped leaf: the ratchet above is empty for free"
    assert len(RED_CLASSES) == 2, "a red class went missing; the ratchet covers one direction"
    for leaf in leaves:
        foreign = load_config(config_for(other_arch(leaf.arch))).model_dump()
        assert declared_representation(foreign) == other_arch(leaf.arch)


def test_a_NEW_red_row_is_refused():
    """PB-T9c. The growth direction — a fifth arch-scoped leaf reaching outside its arch."""
    with pytest.raises(RedRowAppeared, match="train.gnn_v2_edges"):
        check_red_row_ratchet(
            frozenset({("train.gnn_v2_edges", SCHEMA_REQUIRES)}), DECLARED_RED_ROWS
        )


def test_a_REPAIRED_row_that_is_still_declared_is_refused():
    """PB-T9d. The shrink direction, and the half a one-sided ratchet cannot give: a fix must
    delete its row, or the declaration keeps asserting a defect that no longer exists. This is
    the half that emptied `DECLARED_RED_ROWS` at B2 — it is driven against a synthetic row now
    that the live set is empty, because a check with nothing to remove proves nothing."""
    stale = ("train.microbatch_caps.max_edges", SCHEMA_REQUIRES)
    with pytest.raises(RedRowRepairedButStillDeclared, match=re.escape(stale[0])):
        check_red_row_ratchet(frozenset(), frozenset({stale}))


def test_the_ratchet_does_NOT_fire_on_the_declared_set():
    """Negative control for the ratchet."""
    assert check_red_row_ratchet(DECLARED_RED_ROWS, DECLARED_RED_ROWS) == DECLARED_RED_ROWS


def test_an_EMPTY_probe_subject_is_refused_rather_than_reported_GREEN(monkeypatch):
    """PB-T9e. The vacuity break the empty declaration makes possible.

    With `DECLARED_RED_ROWS` empty, deleting every arch-scoped key would make this section
    report GREEN while checking nothing. `observed_red_rows` refuses that by name.
    """
    monkeypatch.setattr(
        "test_config_partition_shared_vs_arch_scoped.arch_scoped_leaves", lambda: ()
    )
    with pytest.raises(ProbeWentVacuous, match="empty for free"):
        observed_red_rows()


# --------------------------------------------------------------------------------------- #
# The repair itself — each half executed against a real minted file
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ARCH_SCOPED_KEYS, ids=[f"{k.section}.{k.field}"
                                                       for k in ARCH_SCOPED_KEYS])
def test_the_SCHEMA_refuses_the_block_on_a_foreign_arch(key, derived):
    """Red class 1's repair, executed POSITIVELY: it is not enough that the shipped grid config
    omits the block — the schema must REFUSE one that carries it, or the row is green only for
    as long as nobody re-adds it."""
    foreign = load_config(config_for(other_arch(key.arch))).model_dump()
    foreign[key.section][key.field] = {member: 1 for member in ("max_edges", "max_nodes",
                                                                "max_fused_edges",
                                                                "max_fused_nodes")
                                       if member in _members_of(key)}
    with pytest.raises(ValidationError, match="ARCH-SCOPED"):
        RunConfig.model_validate(foreign)
    derived(f"t9.schema_refuses.{key.section}.{key.field}", other_arch(key.arch))


@pytest.mark.parametrize("key", ARCH_SCOPED_KEYS, ids=[f"{k.section}.{k.field}"
                                                       for k in ARCH_SCOPED_KEYS])
def test_the_SCHEMA_still_REQUIRES_the_block_on_its_own_arch(key):
    """The other side of the same rule, and the one that keeps the repair from being a deletion:
    dropping the block on the arch that HAS it must be an error, not a silent absence."""
    native = load_config(config_for(key.arch)).model_dump()
    native[key.section].pop(key.field)
    with pytest.raises(ValidationError, match="REQUIRED"):
        RunConfig.model_validate(native)


@pytest.mark.parametrize("key", ARCH_SCOPED_KEYS, ids=[f"{k.section}.{k.field}"
                                                       for k in ARCH_SCOPED_KEYS])
def test_an_EXPLICIT_null_is_CARRYING_the_key_and_is_refused_too(key):
    """The distinction the repair reads off `model_fields_set` rather than off the value.

    `inference.fused_graph_caps`' MEMBERS use `null` as the R119 placeholder ("minted but
    uncalibrated"), so a value test would let a grid config satisfy the arch rule by minting a
    placeholder. Absence of the BLOCK and a null BLOCK are different facts and both are refused
    on a foreign arch."""
    foreign = load_config(config_for(other_arch(key.arch))).model_dump()
    foreign[key.section][key.field] = None
    with pytest.raises(ValidationError, match="ARCH-SCOPED"):
        RunConfig.model_validate(foreign)


@pytest.mark.parametrize("key", ARCH_SCOPED_KEYS, ids=[f"{k.section}.{k.field}"
                                                       for k in ARCH_SCOPED_KEYS])
def test_the_READ_PATH_refuses_by_ARCH_and_not_merely_by_ABSENCE(key, derived):
    """Red class 2's repair, executed POSITIVELY. The resolver is handed a foreign config that
    DOES carry the block, so an absence-only refusal would answer here."""
    foreign = load_config(config_for(other_arch(key.arch))).model_dump()
    foreign[key.section][key.field] = {"max_edges": 1, "max_nodes": 1,
                                       "max_fused_edges": 1, "max_fused_nodes": 1}
    with pytest.raises(ArchScopedKeyOutsideItsArchError, match="ARCH-SCOPED"):
        READ_PATHS[(key.section, key.field)](foreign)
    derived(f"t9.read_path_refuses.{key.section}.{key.field}", other_arch(key.arch))


def test_the_arch_guard_REFUSES_a_key_the_partition_does_not_place():
    """PB-T9f. The guard must not answer 'fine' about a rule that does not exist — a policeman
    with no statute reporting compliance is the phantom-gate shape (LAW-07)."""
    with pytest.raises(KeyError, match="not in ARCH_SCOPED_KEYS"):
        refuse_outside_its_arch({"identity": {"representation": "grid"}}, "train", "batch_size")


def test_the_arch_guard_does_NOT_fire_on_its_OWN_arch():
    """Negative control. A guard that refused everywhere would make the block unreadable on the
    arch that needs it, which is a different defect with the same green ratchet."""
    for key in ARCH_SCOPED_KEYS:
        refuse_outside_its_arch({"identity": {"representation": key.arch}},
                                key.section, key.field)


@pytest.mark.parametrize("key", ARCH_SCOPED_KEYS, ids=[f"{k.section}.{k.field}"
                                                       for k in ARCH_SCOPED_KEYS])
def test_the_read_path_ANSWERS_for_its_OWN_arch(key, derived):
    """The control that keeps the finding above precise: the read path is LIVE for its own arch.

    A repair that made both resolvers refuse everything would turn every red row green and
    break every graph run, and the ratchet alone cannot tell the two apart.
    """
    native = load_config(config_for(key.arch)).model_dump()
    resolved = READ_PATHS[(key.section, key.field)](native)
    derived(f"t9.native_resolve.{key.section}.{key.field}", repr(resolved))
    assert resolved is not None, (
        f"{key.section}.{key.field}: the read path does not answer for its OWN arch, so this "
        "key is not arch-scoped — it is broken, which is a different row"
    )


def _members_of(key) -> tuple[str, ...]:
    """The block's leaf member names, off the live model."""
    section = RunConfig.model_fields[key.section].annotation
    field = section.model_fields[key.field]  # type: ignore[union-attr]
    block = next(a for a in getattr(field.annotation, "__args__", (field.annotation,))
                 if isinstance(a, type) and issubclass(a, BaseModel))
    return tuple(block.model_fields)
