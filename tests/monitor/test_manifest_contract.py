"""⊕ O-01 / O-02 — the producer-manifest contract + the LAW-07 mutation self-tests.

RED-at-import until IMPL writes `mantis.monitor.manifest`. ORACLE-FIRST (⊕): the top-level
`import mantis.monitor.manifest` raises ModuleNotFoundError before any port code exists.

O-01 (P-01): the SHIPPED `src/mantis/monitor/producer_manifest.yaml` loads and EVERY row
resolves — its producer (importable dotted symbol, OR a quoted event-literal present in the
named module's source, OR a `seam` attr whose row carries a `pending:` WP name) AND its
`producer_test` node (file exists + ast-contains the named test function).

O-02 (P-02): the checker BITES — a manifest with (i) a nonexistent producer symbol and one
with (ii) a nonexistent producer_test each raise `ManifestError` naming the offending row.
A checker that cannot bite is the phantom-gate class (F-10, LAW-07 provenance).

Contract (§c.3): `verify_manifest(path, repo_root)` raising `ManifestError(row_id, reason)`;
producer_test paths resolve relative to `repo_root`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.monitor.manifest import ManifestError, load_manifest, verify_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_MANIFEST = _REPO_ROOT / "src" / "mantis" / "monitor" / "producer_manifest.yaml"


def test_shipped_manifest_every_row_resolves() -> None:
    """O-01 / P-01 — the shipped manifest loads and every gate row resolves (producer AND
    producer_test). Any unresolved row is a dead/renamed producer feeding a gate."""
    assert _SHIPPED_MANIFEST.exists(), (
        f"the seam-7 manifest must ship at {_SHIPPED_MANIFEST.relative_to(_REPO_ROOT)}"
    )
    verify_manifest(_SHIPPED_MANIFEST, _REPO_ROOT)  # must NOT raise


def _write_manifest(path: Path, rows: list[str]) -> Path:
    body = "version: 1\nchannel: jsonl_event_sink\ngates:\n" + "".join(rows)
    path.write_text(body)
    return path


# A producer_test node that genuinely exists (this test function), so a mutation manifest
# fails ONLY on the mutated field, never incidentally on the other.
_LIVE_TEST_NODE = "tests/monitor/test_manifest_contract.py::test_shipped_manifest_every_row_resolves"


def test_dead_producer_symbol_bites(tmp_path: Path) -> None:
    """O-02 / P-02 (arm 1) — a row whose producer symbol does not exist ⇒ ManifestError
    naming that row. Proves the checker resolves symbols, not just parses yaml."""
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: dead_symbol_row\n"
        "    producer: {kind: symbol, module: mantis.train.checkpoints, symbol: no_such_symbol}\n"
        f"    producer_test: {_LIVE_TEST_NODE}\n",
    ])
    with pytest.raises(ManifestError) as ei:
        verify_manifest(manifest, _REPO_ROOT)
    assert "dead_symbol_row" in str(ei.value), "the error must name the offending row id"


def test_missing_producer_test_bites(tmp_path: Path) -> None:
    """O-02 / P-02 (arm 2) — a row whose producer_test node does not exist ⇒ ManifestError
    naming that row. Proves the checker ast-verifies the producer test, not just the producer."""
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: missing_test_row\n"
        "    producer: {kind: symbol, module: mantis.train.checkpoints, symbol: persist_errors_total}\n"
        "    producer_test: tests/monitor/test_manifest_contract.py::test_does_not_exist_anywhere\n",
    ])
    with pytest.raises(ManifestError) as ei:
        verify_manifest(manifest, _REPO_ROOT)
    assert "missing_test_row" in str(ei.value)


def test_event_literal_substring_does_not_falsely_resolve(tmp_path: Path) -> None:
    """O-02 — a `kind: event_literal` row is satisfied ONLY by a QUOTED literal in the named
    module, never by an identifier substring. `train_step` as a bare identifier (e.g.
    `self._train_step`) must NOT satisfy a `train_step` literal row → ManifestError."""
    # coordinator/step.py contains the identifier `_train_step` but (pre-wiring) not the
    # quoted literal "train_step"; a substring-matching checker would wrongly pass.
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: literal_needs_quotes\n"
        "    producer: {kind: event_literal, module: mantis.train.coordinator.config, literal: train_step}\n"
        f"    producer_test: {_LIVE_TEST_NODE}\n",
    ])
    with pytest.raises(ManifestError) as ei:
        verify_manifest(manifest, _REPO_ROOT)
    assert "literal_needs_quotes" in str(ei.value)


def test_pending_seam_row_requires_a_wp_name(tmp_path: Path) -> None:
    """O-02 — a `kind: seam` row that is `pending:` but names no WP is an error (a pending gate
    with no owner is the silently-dead-forever class). Bites a pending row missing its WP tag."""
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: seam_no_wp\n"
        "    producer: {kind: seam, module: mantis.train.coordinator.step, symbol: StepCoordinator.on_eval_round_complete}\n"
        "    pending:\n"
        f"    producer_test: {_LIVE_TEST_NODE}\n",
    ])
    with pytest.raises(ManifestError) as ei:
        verify_manifest(manifest, _REPO_ROOT)
    assert "seam_no_wp" in str(ei.value)


def test_empty_manifest_is_a_failure(tmp_path: Path) -> None:
    """O-01 — an empty manifest (no gate rows) is a FAIL, not a vacuous pass (R4: a gate
    surface with zero producers is a phantom-armed abort chain waiting to happen)."""
    path = tmp_path / "empty.yaml"
    path.write_text("version: 1\nchannel: jsonl_event_sink\ngates: []\n")
    with pytest.raises(ManifestError):
        verify_manifest(path, _REPO_ROOT)


# ── ⊕ WP12-R Phase O / O-29 (R164/LAW-07) — the two Phase-O rows EXIST ────────────────
#: The rows Phase O authors, by id. Transcribed rather than derived from the shipped file:
#: an oracle that read its expectation out of the document under test could not witness a
#: deletion, which is the entire subject here (R81).
_PHASE_O_ROW_IDS = ("target_integrity_counters", "terminal_eval_broken")


def test_the_shipped_manifest_contains_the_phase_O_rows() -> None:
    """O-29 — PRESENCE, which is a different mechanism from RESOLUTION and is covered by
    nothing else in this file.

    `verify_manifest` iterates the rows it is HANDED. Only an EMPTY gate list is a failure
    (`manifest.py:78-79`, and `docs/contracts/event_manifest.md:65` says so verbatim); every
    row that is present is then resolved one at a time. So DELETING a row leaves
    `test_shipped_manifest_every_row_resolves` GREEN — the checker has nothing left to
    resolve and reports success over the smaller set. That hole is generic to all rows; this
    oracle closes it for the two Phase O authors, which is the scope Phase O owns.

    Why the two rows matter enough to pin their existence: they are the LAW-07 provenance for
    the two things Phase O ships. `target_integrity_counters` is the anti-rot leg that
    `solver_deltas` never had — a defaulted parameter with no row and no producer test, which
    is exactly why eight solver counters silently never reached the stream and nothing
    noticed. `terminal_eval_broken` cites the producer for exit code 48.

    Read off the raw document, not through `verify_manifest`: presence must be observable
    even when resolution would fail, otherwise the two mechanisms collapse into one and
    M-O29's "O-27 stays green" asymmetry would be unobservable.

    MUTATION THAT REDS IT (M-O29): delete either row from `producer_manifest.yaml`. The
    resolution oracle above stays GREEN under it — that asymmetry is why this row exists."""
    document = load_manifest(_SHIPPED_MANIFEST)
    gates = document["gates"]
    assert isinstance(gates, list) and gates, (
        "premise: the shipped manifest declares gate rows at all"
    )
    present = [row["id"] for row in gates]
    missing = [row_id for row_id in _PHASE_O_ROW_IDS if row_id not in present]
    assert missing == [], (
        f"the shipped producer manifest is missing {missing}. A gate/monitor input with no "
        "row cites no producer and no producer test (R4/LAW-07), and nothing else in this "
        f"file can see its absence. Present rows: {present}"
    )
    assert len(present) == len(set(present)), (
        f"duplicate row ids in the shipped manifest: {present}"
    )
