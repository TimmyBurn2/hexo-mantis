"""Producer: `mantis.encoding` exports no callable without a call site.

`resolve_encoding_for_eval` was exported and called from nowhere, and it was dangerous rather
than merely unused: it resolved from a CHECKPOINT with a shape-inference fallback, the exact
violation the declared-encoding authority exists to prevent.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import mantis.encoding as enc

_REPO = Path(__file__).resolve().parents[2]

#: NOT an exemption — a NAMED, QUEUED exclusion, kept honest by the anti-rot test below.
#: `resolve_arch` has zero call sites but IS a dense plane-geometry surface whose deletion is
#: operator-sign-off-locked; `expand_auto_paths` is the ROOT of the transitively-dead cluster
#: pinned further down, so the census reports that cluster once rather than twice.
_QUEUED_DEAD_EXPORTS = frozenset({"resolve_arch", "expand_auto_paths"})


def _call_sites(name: str) -> list[str]:
    """Every line that CALLS `name`, anywhere in the shipped tree.

    Reachability is measured by call site, never by excluding the defining file: excluding the
    module wholesale once hid a sibling consumer and reported a live function as dead.
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
    """The name must not linger in an import, an `__all__` or a docstring that would send a
    reader looking for it, anywhere in the shipped tree."""
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

    Classes and exception types are excluded because `_call_sites` matches `name(` and a class
    is legitimately named without being constructed.
    """
    return sorted(
        name for name in enc.__all__
        if inspect.isfunction(getattr(enc, name))
    )


def test_every_exported_function_has_a_call_site() -> None:
    """Every function in `__all__` is called somewhere outside its own definition module.

    The census keys on the EXPORT SURFACE, not a `resolve_*` name prefix: a prefix measures the
    naming convention, and one deletion that orphaned three symbols was reported as one.
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
    """Anti-regression on the widening: if the census ever narrows back to a name prefix, the
    family that proved the gap goes unwatched again."""
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
    """Anti-rot on the exclusion above: if an excluded row is ever wired, this reds and forces
    the list to shrink, so an exclusion cannot outlive its grounds."""
    for name in sorted(_QUEUED_DEAD_EXPORTS):
        sites = _call_sites(name)
        assert not sites, (
            f"{name} now HAS call sites — it is no longer dead, so remove it from "
            f"_QUEUED_DEAD_EXPORTS and re-rule its queue row:\n  " + "\n  ".join(sites)
        )


def test_transitively_dead_cluster_is_recorded_not_silently_deleted() -> None:
    """A transitively dead cluster is recorded, not silently deleted.

    `resolve_anchor_path` has live callers, but they sit inside `expand_auto_paths`, which is
    itself unreferenced. This reds the moment the cluster's root gains a consumer (making the
    leaf genuinely live) or loses its body (making the leaf genuinely deletable).
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
