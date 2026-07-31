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
#: an oracle's own coverage). Closing ADJ-WP12R-2 armed the general law below, and the law
#: immediately found two MORE dead exports of the identical class: zero references across
#: `src tests tools crates` outside their own definition module and `__init__`'s re-export
#: (measured, Phase Q). They are NOT deleted here — they were not carded, and R119's hard
#: stop forbids widening scope beyond carded scope, so they are queued as ADJ-WP12R-19
#: with a delete recommendation. This list must SHRINK to empty when that row is ruled; it
#: must never grow silently.
_QUEUED_DEAD_EXPORTS = frozenset({"resolve_anchor_path", "resolve_arch"})


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
    dead: list[str] = []
    for name in sorted(n for n in enc.__all__ if n.startswith("resolve_")):
        if name in _QUEUED_DEAD_EXPORTS:
            continue
        proc = subprocess.run(
            ["git", "grep", "-l", name, "--", "src", "tools", "crates"],
            cwd=_REPO, capture_output=True, text=True, check=False,
        )
        files = {
            f for f in proc.stdout.split()
            if f not in ("src/mantis/encoding/__init__.py", "src/mantis/encoding/resolvers.py")
        }
        if not files:
            dead.append(name)
    assert dead == [], (
        f"exported resolvers with zero call sites outside their own module: {dead} "
        "— dead weight (R116); delete or wire before exporting"
    )


def test_the_queued_dead_exports_are_still_dead() -> None:
    """ANTI-ROT on the exclusion above. If ADJ-WP12R-19 is ruled 'wire it' and a call site
    appears, this reds and forces the allowlist to shrink — so the exclusion cannot quietly
    outlive its grounds, which is how allowlists rot into permanent exemptions."""
    for name in sorted(_QUEUED_DEAD_EXPORTS):
        proc = subprocess.run(
            ["git", "grep", "-l", name, "--", "src", "tools", "crates"],
            cwd=_REPO, capture_output=True, text=True, check=False,
        )
        files = {
            f for f in proc.stdout.split()
            if f not in ("src/mantis/encoding/__init__.py", "src/mantis/encoding/resolvers.py")
        }
        assert not files, (
            f"{name} now HAS a call site ({sorted(files)}) — it is no longer dead, so "
            f"remove it from _QUEUED_DEAD_EXPORTS and close ADJ-WP12R-19 accordingly"
        )
