"""AUDIT-1 F-36 — artifact pins have ONE authority, and it is the live one.

THE DEFECT. `crates/mantis-encoding/src/manifests.{rs,toml}` carried corpus / anchor /
held-out pins whose own header said they had been "moved out of the old Python resolver
dicts". The dicts were still there and were the LIVE authority; the Rust parser and its
accessors were reached by nothing but their own test, with no bridge export wrapping them.
And the two had DRIFTED: the TOML declared `anchor_path` for `gnn_axis_v1` and `gnn_axis_r8`,
which `_ANCHOR_PATHS` does not have, and the hold-out gate that can actually fire is
`assert_not_heldout_sha` — the Rust `assert_not_heldout` could never fire at all.

THE REPAIR was to delete the unreachable side. This row is what keeps the class closed: a
second artifact-pin authority may not reappear in `crates/`, and the live one must answer for
every registered encoding it claims to cover.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.encoding import lookup
from mantis.encoding.registry import EncodingRegistryError
from mantis.encoding.resolvers import (
    _ANCHOR_PATHS,
    _CORPUS_PATHS,
    held_out_shas,
    resolve_anchor_path,
    resolve_corpus_path,
    resolve_corpus_sha_pin,
)

_REPO = Path(__file__).resolve().parents[2]
_CRATES = _REPO / "crates"


def test_no_second_artifact_pin_manifest_lives_in_the_rust_tree() -> None:
    """Structure, not text: any TOML under `crates/` naming a corpus, anchor or held-out pin
    is a second authority over the same invariant. `registry.toml` is encoding SHAPE and is
    the one file allowed to sit there."""
    offenders = []
    for toml in sorted(_CRATES.rglob("*.toml")):
        if toml.name in {"Cargo.toml", "registry.toml"}:
            continue
        body = toml.read_text(encoding="utf-8")
        if any(key in body for key in ("corpus_path", "anchor_path", "held_out", "corpus_sha256")):
            offenders.append(str(toml.relative_to(_REPO)))
    assert not offenders, (
        f"artifact pins re-typed in the Rust tree: {offenders}. "
        "`mantis.encoding.resolvers` is the ONE authority (AUDIT-1 F-36)."
    )


def test_the_live_authority_answers_for_every_encoding_it_claims() -> None:
    """The pins are keyed by encoding NAME, so a registered name with a corpus entry and no
    anchor entry — or the reverse — is the drift that produced F-36 in the first place."""
    for name in _CORPUS_PATHS:
        assert lookup(name) is not None, f"{name!r} is pinned but not registered"
        assert resolve_corpus_path(lookup(name)).name, f"{name}: corpus path resolves"
    for name in _ANCHOR_PATHS:
        assert lookup(name) is not None, f"{name!r} has an anchor pin but is not registered"
        assert resolve_anchor_path(lookup(name)).name, f"{name}: anchor path resolves"


def test_an_encoding_with_no_anchor_pin_RAISES_rather_than_guessing() -> None:
    """The graph rows have no anchor pin, and the deleted TOML declared a PLACEHOLDER path
    for both. A named refusal is the correct answer; a placeholder path is a file that does
    not exist, handed back as if it did."""
    for name in ("gnn_axis_v1", "gnn_axis_r8"):
        assert name not in _ANCHOR_PATHS
        with pytest.raises(EncodingRegistryError, match="anchor path"):
            resolve_anchor_path(lookup(name))


def test_the_held_out_and_pin_sets_are_disjoint() -> None:
    """A corpus that is both launch-pinned and held out would be required to load and
    forbidden to load. The deleted Rust side implemented this invariant a second time."""
    pinned = {
        sha for sha in (resolve_corpus_sha_pin(lookup(n)) for n in _CORPUS_PATHS) if sha
    }
    assert pinned.isdisjoint(held_out_shas()), (
        f"a launch-pinned corpus sha is also registered as held out: {pinned & held_out_shas()}"
    )
