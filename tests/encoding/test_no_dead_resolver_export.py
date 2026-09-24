"""Producer: `mantis.encoding` exports no callable without a call site.

`resolve_encoding_for_eval` was exported and called from nowhere, and it was dangerous rather
than merely unused: it resolved from a CHECKPOINT with a shape-inference fallback, the exact
violation the declared-encoding authority exists to prevent.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

import mantis.encoding as enc
from mantis.encoding import resolvers

_REPO = Path(__file__).resolve().parents[2]

#: Deleted resolver surface: the eval resolver with a shape fallback, then the dense arch
#: resolver and the `<auto>`-path cluster (anchor paths had no registered row).
_DELETED = ("resolve_encoding_for_eval", "resolve_arch", "ArchSpec", "expand_auto_paths",
            "resolve_anchor_path")


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


@pytest.mark.parametrize("name", _DELETED)
def test_the_deleted_resolver_is_gone_from_the_module_and_its_exports(name: str) -> None:
    """The direct assertion. Reintroducing the symbol reds this immediately."""
    assert not hasattr(enc, name)
    assert name not in enc.__all__
    assert not hasattr(resolvers, name)


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
        if not _call_sites(name)
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
