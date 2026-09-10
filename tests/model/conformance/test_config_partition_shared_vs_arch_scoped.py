# >300 justify (R8): the partition's coverage checks, the two reachability probes that execute
# it, and the planted breaks that prove each probe can fire are one unit — a probe whose
# falsifier lives in another file is a probe nobody re-runs against its own break.
"""T9 — the config partition: SHARED keys vs ARCH-SCOPED keys, and who can reach the latter.

An arch-scoped key reachable outside its arch is a red row, which kills the config-blind class
structurally. B1 declared eight red rows because the shipped grid configs carried both
graph-only cap blocks — `RunConfig` is `extra="forbid"` with every key required, so a grid run
was REQUIRED to carry them — and B2's repair DELETES those rows rather than widening them.

The repair is executed here rather than read off the source. SCHEMA: `ARCH_SCOPED_KEYS` is the
ONE authority, and each block is REQUIRED on its own arch and REFUSED on any other, with
presence read off `model_fields_set` so an explicit `null` counts. READ PATH: each block's ONE
resolver calls `refuse_outside_its_arch` BEFORE it looks for the block, since a resolver that
refuses only on ABSENCE is green because the key happens not to be there.

Both red classes are EXECUTED against real minted files. The declared set is now EMPTY and the
ratchet is asserted in BOTH directions, so the probes must be shown to still EXECUTE — an empty
expectation is the one place a set-equality check can go vacuous.
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
from mantis.config.schema.leaves import leaf_paths

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


#: THE READ PATH PER ARCH-SCOPED BLOCK — the only thing this file still declares, because a
#: resolver is not discoverable from a pydantic model while the block-to-arch judgment now has a
#: producer. Both directions are checked below.
READ_PATHS: dict[tuple[str, str], Any] = {
    ("train", "microbatch_caps"): resolve_microbatch_caps,
    ("inference", "fused_graph_caps"): resolve_fused_graph_caps,
}


def arch_scoped_leaves() -> tuple[ArchScopedLeaf, ...]:
    """Every arch-scoped LEAF, DERIVED from the schema registry and the live block models — a
    derivation now that `ARCH_SCOPED_KEYS` exists, so this suite can no longer disagree with the
    schema about which keys are scoped.

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


#: The arch vocabulary a key name can carry, matched on the leaf path so a new
#: `train.gnn_hidden` cannot slip into the shared half silently. A PROMPT, not a verdict, and an
#: INDEPENDENT one: a key missing from `ARCH_SCOPED_KEYS` would otherwise be invisible to the
#: schema and this suite alike.
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
    "identity.arch_kind": "the selector ROW itself (R330(e)): it names which arch KIND a run "
                          "builds, so like `identity.encoding` it must be readable on every "
                          "arch; absent until the run6 mint writes it (R323(b))",
    "train.augment": "symmetry augmentation is a data-pipeline posture; both representations "
                     "have an augmentation path and run5 mints it false on the graph one",
    "monitor.alert_loss_increase_window": "a TIME window over training steps, not a board "
                                          "window; the K-cluster window is a different word",
}

#: THE RED ROWS. EMPTY, and emptied BY THE REPAIR — B1's eight rows are all green now, so the
#: declaration goes with the defect. Ratcheted in BOTH directions below, with a vacuity guard
#: keeping an empty expectation from being a free pass.
DECLARED_RED_ROWS: frozenset[tuple[str, str]] = frozenset()


def live_leaf_paths(model: type[BaseModel] = RunConfig) -> tuple[str, ...]:
    """Every field NAME the live schema reaches, as dotted paths, for the vocabulary probe.

    THE ONE WALKER, in `descend_containers` mode. This was the FIFTH hand copy of the schema
    walk, invisible to the audit's census because that was scoped to the name `_leaf_paths`.
    The difference is deliberate: other consumers want key-paths a config can WRITE, this probe
    wants every field name a future key could hide an architecture in.
    """
    return leaf_paths(model, descend_containers=True)


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


def foreign_dump(arch: str) -> dict:
    """A config mapping that DECLARES the arch `arch` does not have, for the READ PATHS. Only one
    representation is registered, so no shipped file selects a foreign arch, but the read paths
    take a plain mapping and dispatch on `identity.representation`. The block is CARRIED, which
    makes this a test of the arch refusal rather than of absence."""
    dump = load_config(config_for(arch)).model_dump()
    dump["identity"]["representation"] = other_arch(arch)
    return dump


def leaf_present(config_dump: dict, path: str) -> bool:
    node: Any = config_dump
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def observed_red_rows() -> frozenset[tuple[str, str]]:
    """EXECUTE both red classes for every arch-scoped leaf. Nothing here reads source text.

    Raises:
        ProbeWentVacuous: no arch-scoped leaf to probe, or a foreign config declaring no
            representation.
    """
    leaves = arch_scoped_leaves()
    if not leaves:
        raise ProbeWentVacuous(
            "no arch-scoped leaf exists, so both red classes are empty for free"
        )
    rows: set[tuple[str, str]] = set()
    for leaf in leaves:
    # SCHEMA_REQUIRES IS NOT PROBED HERE: only one representation is registered, so there is
    # no OTHER-representation file and no synthetic stands in for one. The class's absence is
    # asserted with its reason by the UNREACHABLE row below.
        dump = foreign_dump(leaf.arch)
        if declared_representation(dump) is None:
            raise ProbeWentVacuous(
                f"the foreign config for {leaf.path} declares no readable "
                "identity.representation, so the read path's arch refusal cannot fire and a "
                "green result would mean nothing"
            )
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
    """The one declaration this file still makes, pinned against the one the schema makes: a
    read-path table outliving its registry entry would keep probing a key nobody scopes, and a
    registry entry with no read path would silently drop a block out of the second red class."""
    registry = {(key.section, key.field) for key in ARCH_SCOPED_KEYS}
    derived("t9.registry_blocks", sorted(registry))
    assert registry == set(READ_PATHS), (
        f"ARCH_SCOPED_KEYS declares {sorted(registry)} and this file names read paths for "
        f"{sorted(READ_PATHS)}; every scoped block needs its one read path and every read path "
        "needs its scope."
    )


def test_a_NEW_arch_vocabulary_key_lands_UNPLACED_rather_than_shared():
    """PB-T9a. A key named `train.gnn_v2_hidden` must not become a shared key by arriving: it is
    refused until someone says which half it is in, which is the only moment anyone will think
    about it."""
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


def test_the_red_row_set_matches_what_B2_declared(derived):
    """The ratchet. GREEN at HEAD: B1 landed the enforcement, B2 landed the repair (R322(d))."""
    observed = observed_red_rows()
    derived("t9.red_rows.observed", sorted(observed))
    derived("t9.red_rows.declared", sorted(DECLARED_RED_ROWS))
    assert check_red_row_ratchet(observed, DECLARED_RED_ROWS) == observed


def test_the_probes_are_still_EXECUTING_and_not_merely_empty(derived):
    """The guard an EMPTY expectation needs. While red rows were declared the ratchet could not
    pass vacuously; with the declaration empty, "no red rows" and "the probe never ran" are the
    same result, so leaves, both classes and each foreign declaration are asserted here."""
    leaves = arch_scoped_leaves()
    derived("t9.arch_scoped_leaf_count", len(leaves))
    assert leaves, "no arch-scoped leaf: the ratchet above is empty for free"
    assert len(RED_CLASSES) == 2, "a red class went missing; the ratchet covers one direction"
    for leaf in leaves:
        assert declared_representation(foreign_dump(leaf.arch)) == other_arch(leaf.arch)


def test_a_NEW_red_row_is_refused():
    """PB-T9c. The growth direction — a fifth arch-scoped leaf reaching outside its arch."""
    with pytest.raises(RedRowAppeared, match="train.gnn_v2_edges"):
        check_red_row_ratchet(
            frozenset({("train.gnn_v2_edges", SCHEMA_REQUIRES)}), DECLARED_RED_ROWS
        )


def test_a_REPAIRED_row_that_is_still_declared_is_refused():
    """PB-T9d. The shrink direction a one-sided ratchet cannot give: a fix must delete its row,
    or the declaration keeps asserting a defect that no longer exists. Driven against a
    synthetic row now that the live set is empty."""
    stale = ("train.microbatch_caps.max_edges", SCHEMA_REQUIRES)
    with pytest.raises(RedRowRepairedButStillDeclared, match=re.escape(stale[0])):
        check_red_row_ratchet(frozenset(), frozenset({stale}))


def test_the_ratchet_does_NOT_fire_on_the_declared_set():
    """Negative control for the ratchet."""
    assert check_red_row_ratchet(DECLARED_RED_ROWS, DECLARED_RED_ROWS) == DECLARED_RED_ROWS


def test_an_EMPTY_probe_subject_is_refused_rather_than_reported_GREEN(monkeypatch):
    """PB-T9e. With `DECLARED_RED_ROWS` empty, deleting every arch-scoped key would make this
    section report GREEN while checking nothing. `observed_red_rows` refuses that by name."""
    monkeypatch.setattr(
        "test_config_partition_shared_vs_arch_scoped.arch_scoped_leaves", lambda: ()
    )
    with pytest.raises(ProbeWentVacuous, match="empty for free"):
        observed_red_rows()


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
def test_the_READ_PATH_refuses_by_ARCH_and_not_merely_by_ABSENCE(key, derived):
    """Red class 2's repair, executed POSITIVELY. The resolver is handed a foreign-declaring
    config that DOES carry the block, so an absence-only refusal would answer here."""
    foreign = foreign_dump(key.arch)
    foreign[key.section][key.field] = {"max_edges": 1, "max_nodes": 1,
                                       "max_fused_edges": 1, "max_fused_nodes": 1}
    with pytest.raises(ArchScopedKeyOutsideItsArchError, match="ARCH-SCOPED"):
        READ_PATHS[(key.section, key.field)](foreign)
    derived(f"t9.read_path_refuses.{key.section}.{key.field}", other_arch(key.arch))


def test_the_SCHEMAS_foreign_arch_refusal_is_UNREACHABLE_and_says_why(derived):
    """Red class 1 has NO CONSTRUCTIBLE INPUT since only one representation is registered, and
    that is asserted rather than quietly dropped: `identity.representation` is a one-member
    `Literal` and the identity-consistency validator refuses a mismatch BEFORE the arch rule
    runs, so the refusal cannot fire and the rows that drove it are deleted rather than faked.
    This reds the day a second representation is registered."""
    dump = load_config(config_for("graph")).model_dump()
    dump["identity"]["representation"] = other_arch("graph")
    with pytest.raises(ValidationError) as excinfo:
        RunConfig.model_validate(dump)
    message = str(excinfo.value)
    derived("t9.schema_foreign_arch_unreachable", other_arch("graph"))
    assert "ARCH-SCOPED" not in message, (
        "the schema's arch-scoped refusal fired on a foreign representation — a second "
        "representation is registered and red class 1 has a subject again; restore the "
        "SCHEMA_REQUIRES probe and its positive rows"
    )
    assert "disagrees with the registry representation" in message, message


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
    """The control that keeps the finding precise: the read path is LIVE for its own arch. A
    repair that made both resolvers refuse everything would turn every red row green and break
    every graph run, and the ratchet alone cannot tell the two apart."""
    native = load_config(config_for(key.arch)).model_dump()
    resolved = READ_PATHS[(key.section, key.field)](native)
    derived(f"t9.native_resolve.{key.section}.{key.field}", repr(resolved))
    assert resolved is not None, (
        f"{key.section}.{key.field}: the read path does not answer for its OWN arch, so this "
        "key is not arch-scoped — it is broken, which is a different row"
    )
