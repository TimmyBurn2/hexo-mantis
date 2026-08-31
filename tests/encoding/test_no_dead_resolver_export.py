"""ADJ-WP12R-2 producer: `mantis.encoding` exports no resolver without a call site.

`resolve_encoding_for_eval` was exported from `mantis.encoding.__all__` and called from
NOWHERE repo-wide — dead weight, deleted under R116's dead-weight law. It was named for
exactly this card's job, which is why it is dangerous rather than merely unused: a future
reader looking for "the eval encoding resolver" would find a symbol that resolves from a
CHECKPOINT and carries a shape-inference fallback, i.e. precisely the LAW-11 violation the
declared-encoding authority exists to prevent.

The general assertion (no dead `resolve_*` export) is the one that survives this card.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import mantis.encoding as enc

_REPO = Path(__file__).resolve().parents[2]

#: NOT an exemption — a NAMED, QUEUED exclusion (R7's no-silent-caps discipline applied to
#: an oracle's own coverage). ONE entry, and its grounds are R154's condition (b), not
#: deadness: `resolve_arch` has zero call sites anywhere (re-verified by call-site search,
#: below) but IS a dense plane-geometry surface — it derives `kept_indices`, history and
#: turn-phase plane slots, i.e. the 18-plane / lean-4 machinery whose deletion list is
#: WP-LEAN-RENAME's and is operator-sign-off-locked (R117/R140). R154 conditioned the
#: deletion on it NOT being an R20-protected dense surface; it is one, so the row goes back
#: as queue-with-recommendation (R108) and this exclusion stands meanwhile.
#:
#: `resolve_anchor_path` was on this list in Phase Q and has been REMOVED — not deleted,
#: CORRECTED: it has two live call sites (`expand_auto_paths`, resolvers.py:388,403). The
#: Phase-Q evidence was defective because the check below excluded the whole defining
#: MODULE, which hid a sibling consumer. That is fixed here: reachability is now measured by
#: CALL SITE, never by file exclusion.
#:
#: `expand_auto_paths` JOINS THE QUEUE at R327(e), and NOT because anything about it changed —
#: because the census below stopped being scoped to `resolve_*` and can now see it. It is the
#: ROOT of the transitively-dead cluster `test_transitively_dead_cluster_is_recorded_not_
#: silently_deleted` already pins: `resolve_anchor_path` is live only through it, and it has no
#: caller of its own. That test remains the row's grounds and its anti-rot; this entry exists so
#: the widened census reports the cluster once rather than twice.
#:
#: **`resolve_corpus_sha_pin` LEFT THIS SET at R327(e)** — it is wired, at
#: `mantis.train.pretrain.graph_route._assert_launch_pin`, and the anti-rot test below is what
#: forced the removal rather than a memory of having done it.
#:
#: **THE `resolve_` SCOPING WAS A REAL COVERAGE GAP AND IT IS CLOSED HERE.** R326(d)'s one
#: deletion orphaned THREE symbols and the census saw ONE, because two of them
#: (`assert_not_heldout_sha`, `heldout_size_bytes`) do not begin `resolve_`. A census that
#: measures a NAME PREFIX measures its own naming convention; this one measures the export
#: surface. All three are now accounted for: the contamination gate is wired into
#: `mantis.data.bootstrap_encode.encode_corpus`, the launch pin into `graph_route`, and
#: `heldout_size_bytes` is BURIED with a grave line — its stat-only pre-filter could only ever
#: skip a sha stream, and the surviving path streams unconditionally.
#:
#: **WHY THE FAMILY WAS NOT DELETED WITH THE LOADER.** `assert_not_heldout_sha` is the held-out
#: CONTAMINATION gate, and the armed posture still cares about it: BC-pretrain reads a corpus
#: too. Its integrity path has a manifest handshake with a streaming sha256 that hard-fails on
#: disagreement — but that proves the file is the file it claims to be, NOT that it is outside
#: the evaluation hold-out set. Those are different properties, and only one of them had a
#: guard.
_QUEUED_DEAD_EXPORTS = frozenset({"resolve_arch", "expand_auto_paths"})


def _call_sites(name: str) -> list[str]:
    """Every line that CALLS `name`, anywhere in the shipped tree.

    Reachability is measured by call site, never by excluding the defining file. Phase Q's
    version excluded `resolvers.py` wholesale to skip the `def` line, and that hid a sibling
    consumer in the SAME module (`expand_auto_paths` calls `resolve_anchor_path` twice) —
    reporting a live function as dead. Matching `name(` and dropping only the `def` line
    keeps the def out without blinding the search to real callers.
    """
    proc = subprocess.run(
        ["git", "grep", "-n", f"{name}(", "--", "src", "tools", "crates"],
        cwd=_REPO, capture_output=True, text=True, check=False,
    )
    return [ln for ln in proc.stdout.splitlines() if f"def {name}(" not in ln]


def test_the_deleted_resolver_is_gone_from_the_module_and_its_exports() -> None:
    """The direct assertion. Reintroducing the symbol reds this immediately."""
    assert not hasattr(enc, "resolve_encoding_for_eval")
    assert "resolve_encoding_for_eval" not in enc.__all__


def test_no_reference_to_the_deleted_resolver_survives_anywhere() -> None:
    """The name must not linger in an import, an `__all__`, or a doc-string that would
    send a reader looking for it. Searched over the shipped tree, not just this package."""
    proc = subprocess.run(
        ["git", "grep", "-n", "resolve_encoding_for_eval", "--", "src", "tests", "tools", "crates"],
        cwd=_REPO, capture_output=True, text=True, check=False,
    )
    hits = [
        line for line in proc.stdout.splitlines()
        if not line.startswith(f"tests/encoding/{Path(__file__).name}")
    ]
    assert hits == [], "the deleted resolver is still referenced:\n" + "\n".join(hits)


def _exported_callables() -> list[str]:
    """Every FUNCTION `mantis.encoding` exports, derived from the module, never listed here.

    The classes and exception types in `__all__` are excluded because `_call_sites` matches
    `name(` and a class is legitimately named without being constructed (annotations, `except`
    clauses, `isinstance`). Deriving the split with `inspect` rather than transcribing a name
    list is what lets the census grow with the package instead of going stale against it.
    """
    return sorted(
        name for name in enc.__all__
        if inspect.isfunction(getattr(enc, name))
    )


def test_every_exported_function_has_a_call_site() -> None:
    """THE GENERAL PRODUCER — the law, not the instance (LAW-08 live-consumer).

    Every function in `__all__` must be called somewhere outside its own definition module and
    outside `__init__`'s re-export. This is what would have caught ADJ-WP12R-2 before it
    shipped, and what catches the next one.

    WIDENED AT R327(e), from `resolve_*` to the export surface. The prefix scoping was measured
    blind: one ruled deletion (R326(d)) orphaned `resolve_corpus_sha_pin`, `assert_not_heldout_
    sha` and `heldout_size_bytes` in a single act, and this census reported ONE of the three. A
    census that keys on a naming convention measures the convention, not the surface.
    """
    dead = [
        name for name in _exported_callables()
        if name not in _QUEUED_DEAD_EXPORTS and not _call_sites(name)
    ]
    assert dead == [], (
        f"exported functions with zero call sites outside their own module: {dead} "
        "— dead weight (R116); delete or wire before exporting"
    )


def test_the_census_covers_the_whole_export_surface_not_a_name_prefix() -> None:
    """ANTI-REGRESSION on the widening itself (R327(e)).

    The gap this closes was invisible for exactly as long as every dead export happened to be
    called `resolve_*`. If the census ever narrows back to a prefix, the family that proved the
    gap goes unwatched again — so the property is asserted rather than trusted to the diff.
    """
    covered = set(_exported_callables())
    assert "assert_not_heldout_sha" in covered and "held_out_shas" in covered, (
        "the corpus-integrity family is outside the census again — the R326(d) orphans are "
        f"exactly the members that do not begin `resolve_`; covered: {sorted(covered)}"
    )
    non_prefixed = {n for n in covered if not n.startswith("resolve_")}
    assert len(non_prefixed) >= 2, (
        "the census is watching a name prefix, not the export surface; that is the scoping "
        f"R327(e) removed. Non-`resolve_` members seen: {sorted(non_prefixed)}"
    )


def test_the_queued_dead_exports_are_still_dead() -> None:
    """ANTI-ROT on the exclusion above — it SURVIVES the R154 dispositions by design (R154:
    the anti-rot test is the law's enforcement, not the exclusions' registry). If the
    excluded row is ever wired, this reds and forces the list to shrink, so the exclusion
    cannot outlive its grounds."""
    for name in sorted(_QUEUED_DEAD_EXPORTS):
        sites = _call_sites(name)
        assert not sites, (
            f"{name} now HAS call sites — it is no longer dead, so remove it from "
            f"_QUEUED_DEAD_EXPORTS and re-rule its queue row:\n  " + "\n  ".join(sites)
        )


def test_transitively_dead_cluster_is_recorded_not_silently_deleted() -> None:
    """R154 condition (a) as an ASSERTION, not a claim in a document.

    `resolve_anchor_path`'s zero-ref evidence FAILED re-verification: it has live callers.
    It is dead only TRANSITIVELY — its callers sit inside `expand_auto_paths`, which is
    itself unreferenced. Deleting the leaf alone would break the caller; discharging it
    means taking the whole cluster, which R154 did not authorize.

    This pins the shape so the next reader cannot mistake "has callers" for "is reachable",
    and reds the moment the cluster's root gains a consumer (making the leaf genuinely live)
    or loses its body (making the leaf genuinely deletable).
    """
    leaf = _call_sites("resolve_anchor_path")
    assert leaf, "resolve_anchor_path lost its callers — re-rule ADJ-WP12R-19's leaf half"
    assert all("resolvers.py" in ln for ln in leaf), (
        "resolve_anchor_path gained a caller OUTSIDE resolvers.py — it is now genuinely "
        "live, not transitively dead:\n  " + "\n  ".join(leaf)
    )
    assert not _call_sites("expand_auto_paths"), (
        "expand_auto_paths — the dead cluster's ROOT — gained a caller, which makes "
        "resolve_anchor_path genuinely reachable. Re-rule ADJ-WP12R-19 accordingly."
    )
