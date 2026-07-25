"""WPSC Phase 3 SC-B1 — grep-gate: the radius_override chain is fully deleted (A9, R25
commit B; DESIGN_P3.md §2.1/§2.3, REV1). `Board::override_legal_move_radius` is ALSO
deleted (§2.1's corrected verdict, not just the curriculum plumbing around it), so there
are no positive-control survivors left for that name — only for the byte-identical
sibling `set_legal_move_radius`, which stays.

RED at HEAD (`507c23b`): the whole chain (Rust core + bridge + selfplay crate + Python
pool/pool_hooks + both `.pyi` twins) is still present. All three greps below currently
return nonzero hits; the `hasattr` absence check is currently `True` (the method exists).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from mantis import _engine
from mantis.selfplay import pool, pool_hooks

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAN_DIRS = ("src/", "tests/", "crates/")
_SCAN_INCLUDES = ("--include=*.py", "--include=*.rs", "--include=*.pyi")
# Exclude THIS file — it necessarily names every banned string as a grep pattern/literal.
_SELF_EXCLUDE = "--exclude=test_radius_chain_removed.py"


def _grep(pattern: str) -> list[str]:
    proc = subprocess.run(
        ["grep", "-rn", pattern, *_SCAN_DIRS, *_SCAN_INCLUDES, _SELF_EXCLUDE],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    # grep rc=1 means "no matches" (not an error); rc>1 is a real grep failure.
    assert proc.returncode in (0, 1), proc.stderr
    return [line for line in proc.stdout.splitlines() if line]


def test_radius_override_zero_hits() -> None:
    hits = _grep("radius_override")
    assert hits == [], f"radius_override survivors: {hits}"


def test_set_radius_override_zero_hits() -> None:
    hits = _grep("set_radius_override")
    assert hits == [], f"set_radius_override survivors: {hits}"


def test_override_legal_move_radius_zero_hits() -> None:
    hits = _grep("override_legal_move_radius")
    assert hits == [], f"override_legal_move_radius survivors: {hits}"


def test_set_legal_move_radius_positive_control_survives() -> None:
    """The byte-identical, non-deleted sibling method must still resolve to real call
    sites under `crates/` — proves the deletion did not over-reach."""
    proc = subprocess.run(
        ["grep", "-rln", "set_legal_move_radius", "crates/"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    hits = [line for line in proc.stdout.splitlines() if line]
    assert hits, "set_legal_move_radius must still resolve to real crates/ call sites"


def test_pool_surface_exposes_no_radius_override() -> None:
    """`mantis.selfplay.pool_hooks`/`mantis.selfplay.pool` import cleanly and neither
    module (nor `Board`/`PyBoard`) exposes any radius-override surface any more."""
    assert not hasattr(pool_hooks, "set_radius_override")
    assert not hasattr(pool, "set_radius_override")
    assert not hasattr(pool, "_set_radius_override")
    assert not hasattr(pool.WorkerPool, "set_radius_override")
    assert not hasattr(_engine.Board, "override_legal_move_radius")
