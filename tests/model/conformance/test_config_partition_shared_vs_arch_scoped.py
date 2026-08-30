# >300 justify (R8): the partition declaration, the two reachability probes that execute it, and
# the ratchet that keeps the red rows from growing or going stale are one unit — a ratchet whose
# rows are produced elsewhere is a list nobody re-derives, which is the defect it exists to catch.
"""T9 — the config partition: SHARED keys vs ARCH-SCOPED keys, and who can reach the latter.

SEAM_V1_DESIGN §3: "The schema splits shared vs arch-scoped. **An arch-scoped key reachable
outside its arch is a red row.** This kills the config-blind class structurally — the defect we
have paid for four times."

WHAT IS RED AT HEAD, and it is not a guess — it is executed below. Both grid configs the repo
ships (`smoke_radius_curriculum.yaml`, `sustained_kcluster.yaml`) carry `train.microbatch_caps`
and `inference.fused_graph_caps` — graph-only blocks, counted in EDGES and NODES — and they
carry byte-identical values to each other, which is what a copied block looks like. `RunConfig`
is `extra="forbid"` with every key required, so a grid run is not merely allowed to carry the
graph caps: it is REQUIRED to, and the mint tooling has to invent a number for a quantity that
run has none of.

THE HALF THAT IS ALREADY CLEAN, said precisely so this reads as a finding and not a smear. The
CALL SITES are arch-gated today and were gated deliberately: `run.py` resolves
`fused_graph_caps` only `if config.identity.representation == "graph"`, with the reason written
beside it ("reading it on a grid run would make a graph-only key a grid dependency"), and
`coordinator/step.py` hands `microbatch_caps` to the GRAPH arm as a lazy thunk for the same
reason. So no grid run reads these values. What is unguarded is everything BELOW the call site:
the schema demands the key, and the resolver — "THE one read path" in its own docstring — will
serve it to a config of either arch without complaint. The gate is one `if` at one call site,
and the class this tier exists for is the one where the next call site forgets it.

TWO RED-ROW CLASSES, both EXECUTED, never inferred from source text:

  * `schema_requires_outside_arch` — a config that selects the OTHER representation validates,
    through the one loader, carrying this key.
  * `read_path_serves_outside_arch` — the key's ONE read path, handed that same config, RETURNS
    a value instead of refusing.

NO MINTED CONFIG IS REWRITTEN HERE. B1's scope is the enforcement, not the file reshaping (lane
C). So the rows are declared and RATCHETED IN BOTH DIRECTIONS: a new red row fails, and a
declared row that has gone green ALSO fails, so a lane-C fix cannot leave a stale entry behind
claiming a defect that is repaired. That two-sidedness is the whole reason this is a set equality
and not a budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from mantis.config.loader import load_config
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.config.schema.core import RunConfig

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


@dataclass(frozen=True)
class ArchScopedKey:
    """One leaf key that belongs to exactly one representation, with its one read path."""

    path: str
    arch: str
    read_path: Any


#: THE ARCH-SCOPED HALF OF THE PARTITION. Declared, not derived, and that is argued rather than
#: assumed: whether a key belongs to one representation is a judgment about what the key MEANS,
#: and there is no producer in the tree to read it off — the same reason T6 declares tier
#: placement instead of deriving it. What IS derived is everything the declaration is checked
#: against: the live leaf-key set, and both reachability answers.
ARCH_SCOPED: tuple[ArchScopedKey, ...] = (
    ArchScopedKey("train.microbatch_caps.max_edges", "graph", resolve_microbatch_caps),
    ArchScopedKey("train.microbatch_caps.max_nodes", "graph", resolve_microbatch_caps),
    ArchScopedKey(
        "inference.fused_graph_caps.max_fused_edges", "graph", resolve_fused_graph_caps
    ),
    ArchScopedKey(
        "inference.fused_graph_caps.max_fused_nodes", "graph", resolve_fused_graph_caps
    ),
)

#: The arch vocabulary a key name can carry. Matched on the leaf path so a NEW key called
#: `train.gnn_hidden` or `inference.max_graph_batch` cannot slip into the shared half silently.
#: This is a PROMPT, not a verdict: a match means the key must be placed on one side or the
#: other, and the placement below is what says which.
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

#: THE RED ROWS AS B1 FOUND THEM, `(leaf path, red class)`. Ratcheted in BOTH directions below.
#: B1 lands the enforcement; the repairs are lane C's, and each repair deletes its row here.
DECLARED_RED_ROWS: frozenset[tuple[str, str]] = frozenset(
    {(key.path, red) for key in ARCH_SCOPED for red in RED_CLASSES}
)


def live_leaf_paths(model: type[BaseModel] = RunConfig, prefix: str = "") -> tuple[str, ...]:
    """Every leaf key of the shipped schema, walked off `RunConfig` itself.

    The same walk `tools/ci_gates/contract_doc_gate.py` performs, and for the same stated
    reason: a transcribed key list is written in the commit that adds a key and therefore can
    never be the thing that notices one.
    """
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out.extend(live_leaf_paths(annotation, path + "."))
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
    """EXECUTE both red classes for every arch-scoped key. Nothing here reads source text.

    `schema_requires_outside_arch` loads a real minted config of the OTHER representation through
    the one loader and asks whether the key came through validation. `read_path_serves_outside_arch`
    hands that same config to the key's ONE read path and asks whether it answered.
    """
    rows: set[tuple[str, str]] = set()
    for key in ARCH_SCOPED:
        foreign = load_config(config_for(other_arch(key.arch)))
        dump = foreign.model_dump()
        if leaf_present(dump, key.path):
            rows.add((key.path, SCHEMA_REQUIRES))
        try:
            key.read_path(dump)
        except Exception:  # noqa: BLE001 — any refusal at all is the green outcome here
            pass
        else:
            rows.add((key.path, READ_PATH_SERVES))
    return frozenset(rows)


def check_red_row_ratchet(
    observed: frozenset[tuple[str, str]], declared: frozenset[tuple[str, str]]
) -> frozenset[tuple[str, str]]:
    """Set equality, both directions. Neither half is optional and they catch opposite things."""
    appeared = sorted(observed - declared)
    if appeared:
        raise RedRowAppeared(
            f"arch-scoped keys reachable outside their arch that B1 did not declare: {appeared}. "
            "An arch-scoped key reachable outside its arch is a red row (SEAM_V1_DESIGN §3) — "
            "either scope the key at the schema/read-path level, or place it in the shared half "
            "with grounds."
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
            "each one: arch-scoped with its read path, or shared WITH GROUNDS saying which "
            "homonym fired the probe. Defaulting a new graph-only key into the shared half is "
            "exactly how the four config-blind defects arrived."
        )
    return flagged


# --------------------------------------------------------------------------------------- #
# Coverage — the partition against the live schema
# --------------------------------------------------------------------------------------- #
def test_the_partition_covers_every_live_key_that_carries_arch_vocabulary(derived):
    live = live_leaf_paths()
    scoped = tuple(key.path for key in ARCH_SCOPED)
    shared = frozenset(SHARED_DESPITE_THE_NAME)
    flagged = check_partition_covers_the_live_schema(live, scoped, shared)
    derived("t9.live_leaf_count", len(live))
    derived("t9.arch_vocabulary_hits", sorted(flagged))
    derived("t9.arch_scoped", sorted(scoped))
    assert flagged, "the vocabulary probe matched no live key at all — it is prompting nobody"


def test_a_NEW_arch_vocabulary_key_lands_UNPLACED_rather_than_shared():
    """PB-T9a. The generalisation this section exists for: GnnNetV2's own config keys.

    A key named `train.gnn_v2_hidden` must not become a shared key by arriving. It is refused
    until someone says which half it is in, which is the only moment anyone will think about it.
    """
    live = (*live_leaf_paths(), "train.gnn_v2_hidden")
    with pytest.raises(ArchVocabularyKeyUnplaced, match="gnn_v2_hidden"):
        check_partition_covers_the_live_schema(
            live,
            tuple(key.path for key in ARCH_SCOPED),
            frozenset(SHARED_DESPITE_THE_NAME),
        )


def test_a_RETIRED_placement_is_refused_rather_than_carried():
    """PB-T9b. The other direction of coverage — a placement whose key is gone."""
    with pytest.raises(PartitionKeyRetired, match="train.gone_key"):
        check_partition_covers_the_live_schema(
            live_leaf_paths(),
            (*[key.path for key in ARCH_SCOPED], "train.gone_key"),
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


# --------------------------------------------------------------------------------------- #
# Reachability — the two red classes, executed
# --------------------------------------------------------------------------------------- #
def test_the_red_row_set_matches_what_B1_declared(derived):
    """The ratchet. Red at HEAD by design: B1 lands the enforcement, lane C lands the repairs."""
    observed = observed_red_rows()
    derived("t9.red_rows.observed", sorted(observed))
    derived("t9.red_rows.declared", sorted(DECLARED_RED_ROWS))
    assert check_red_row_ratchet(observed, DECLARED_RED_ROWS) == observed


def test_a_NEW_red_row_is_refused():
    """PB-T9c. The growth direction — a fifth arch-scoped key reaching outside its arch."""
    with pytest.raises(RedRowAppeared, match="train.gnn_v2_edges"):
        check_red_row_ratchet(
            DECLARED_RED_ROWS | {("train.gnn_v2_edges", SCHEMA_REQUIRES)}, DECLARED_RED_ROWS
        )


def test_a_REPAIRED_row_that_is_still_declared_is_refused():
    """PB-T9d. The shrink direction, and the half a one-sided ratchet cannot give: lane C's fix
    must delete its row, or the declaration keeps asserting a defect that no longer exists."""
    one = next(iter(sorted(DECLARED_RED_ROWS)))
    with pytest.raises(RedRowRepairedButStillDeclared, match=re.escape(one[0])):
        check_red_row_ratchet(DECLARED_RED_ROWS - {one}, DECLARED_RED_ROWS)


def test_the_ratchet_does_NOT_fire_on_the_declared_set():
    """Negative control for the ratchet."""
    assert check_red_row_ratchet(DECLARED_RED_ROWS, DECLARED_RED_ROWS) == DECLARED_RED_ROWS


@pytest.mark.parametrize("key", ARCH_SCOPED, ids=[k.path for k in ARCH_SCOPED])
def test_the_read_path_ANSWERS_for_its_OWN_arch(key, derived):
    """The control that keeps the finding above precise: the read path is LIVE for its own arch.

    Every arch-scoped key's read path serves a foreign config — that is the red row. What keeps
    it from being a live defect is that no foreign run calls it, and this records the shape of
    that guard as a fact rather than a claim: the key's own arch validates and resolves, so the
    read path is live for the arch it belongs to and the redness is entirely about who ELSE can
    reach it.
    """
    native = load_config(config_for(key.arch)).model_dump()
    resolved = key.read_path(native)
    derived(f"t9.native_resolve.{key.path}", repr(resolved))
    assert resolved is not None, (
        f"{key.path}: the read path does not answer for its OWN arch, so this key is not "
        "arch-scoped — it is broken, which is a different row"
    )
