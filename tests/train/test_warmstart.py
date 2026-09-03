"""The warm-start module carries NO host-coupled path — the one guard that outlives the arm.

WHAT THIS FILE USED TO BE, and what stayed. O-WARM tested the value-head warm-start arm:
`resolve_warmstart_head_file`, `default_head_for_arm`, `maybe_warmstart_value_head`. That arm is
DELETED (AUDIT-1 F-19's dead-code half) — it had no production entry, and it read two keys the
schema does not have with code-side defaults on identity quantities. Its tests go with it, which
is the discipline: a test whose subject is gone is not evidence of anything.

WHAT SURVIVES IS THE CENSUS, and it survives because its subject is the MODULE, not the arm. The
killed `_HEADSWAP_AB_DIR = "/home/…"` default is exactly the host-coupling class R1 bans and CI
gate 17 (rule 7) exists to keep out of a public repo, and a personal path could be reintroduced
by any future edit to this file — including one that re-adds a head-seeding arm. So the ban is
asserted over the whole source, and the assertion now covers the BC entry that replaced the arm.

The BC warm-start entry R332(d) built has its own suite: `tests/train/test_bc_warm_start_entry.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import mantis.train.warmstart as warmstart

_WARMSTART_SRC = Path(warmstart.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("needle", ["/home/", "/Users/", "expanduser"])
def test_no_personal_path_in_warmstart_source(needle: str) -> None:
    """Census: warmstart.py carries NO host-coupled personal path / expanduser (the deleted
    `_HEADSWAP_AB_DIR = "/home/…"` default). Bites: any personal-path default resurrected."""
    assert needle not in _WARMSTART_SRC, (
        f"warmstart.py must not contain {needle!r} (R1 host-coupling ban; CI gate 17 rule 7)"
    )


def test_the_killed_headswap_default_stays_killed() -> None:
    """By NAME as well as by shape — the old default's identifier must not come back either."""
    assert "_HEADSWAP_AB_DIR" not in _WARMSTART_SRC


def test_no_absolute_path_literal_anywhere_in_the_module() -> None:
    """The general form of the same ban, so a NEW host-coupled default under a different name
    is caught. Structure, not the one name: any string literal starting with `/` is refused.

    This is strictly wider than the two tests above, and it is the arm that would have caught
    the original `_HEADSWAP_AB_DIR` without anyone having to know its name.
    """
    import ast

    absolutes = [
        node.value for node in ast.walk(ast.parse(_WARMSTART_SRC))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.startswith("/") and len(node.value) > 1
    ]
    assert absolutes == [], (
        f"absolute path literal(s) in warmstart.py: {absolutes}. A path this module does not "
        "receive from its caller is a host-coupled default, whatever it is named"
    )


def test_the_deleted_value_head_arm_stays_deleted() -> None:
    """The inverse pin. Re-adding the arm must be a decision, not a merge.

    Every member below read `combined_config["warm_start"]` or `["value_head_type"]` — keys the
    schema does not have — with code-side defaults under them. R332(d) decided what the entry
    IS (`identity.warm_start`: a checkpoint named by path AND by the net hash it must turn out
    to be), and a second, key-less entry sitting beside it is the ambiguity that decision closed.
    """
    for gone in ("resolve_warmstart_head_file", "default_head_for_arm", "load_value_head",
                 "maybe_warmstart_value_head", "assert_dist65_bins_seeded",
                 "HEAD_FILE_BY_TYPE"):
        assert not hasattr(warmstart, gone), (
            f"{gone} is back. If a value-head warm-start arm is wanted, it needs a SCHEMA key "
            "with a live consumer (R1) — which is a mint act (R323(b)), not a module edit"
        )
    # Over the AST's string CONSTANTS, not the raw source: the comment that records this
    # deletion necessarily names the key it removed, and a text census over source cannot tell
    # a defect from a note about a defect. REPAIR-2 recorded that trap three times in one leg;
    # this is the same file paying it forward.
    import ast

    sniffed = [n.value for n in ast.walk(ast.parse(_WARMSTART_SRC))
               if isinstance(n, ast.Constant) and n.value == "value_head_type"]
    assert sniffed == [], (
        "the arch sniff `combined_config.get('value_head_type', 'scalar')` is back; it is one "
        "of the sites AUDIT-1 F-24 counts, and it defaults an IDENTITY quantity"
    )
