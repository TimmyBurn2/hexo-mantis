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
_QUEUED_DEAD_EXPORTS = frozenset({"resolve_arch"})


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


def test_every_exported_resolver_has_a_call_site() -> None:
    """THE GENERAL PRODUCER — the law, not the instance (LAW-08 live-consumer).

    Every `resolve_*` in `__all__` must be called somewhere outside its own definition
    module and outside `__init__`'s re-export. This is what would have caught ADJ-WP12R-2
    before it shipped, and what catches the next one.
    """
    dead = [
        name for name in sorted(n for n in enc.__all__ if n.startswith("resolve_"))
        if name not in _QUEUED_DEAD_EXPORTS and not _call_sites(name)
    ]
    assert dead == [], (
        f"exported resolvers with zero call sites outside their own module: {dead} "
        "— dead weight (R116); delete or wire before exporting"
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
