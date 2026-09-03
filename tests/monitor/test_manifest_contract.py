# R8 justify: the manifest checker's oracle is one suite because each row drives the SAME
# `verify_manifest` entry over a mutated one-row manifest, and the mutations are meaningful
# only against each other — a dead symbol, a missing producer test, a docstring-only literal
# and a deselected producer are four ways the same checker can resolve against nothing, and
# splitting them would let one file assert a shape another file's fixture no longer builds.
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
from mantis.monitor.manifest import DEFAULT_MANIFEST_PATH

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: The SHIPPED manifest, from its own module — one authority, not a path copy (AUDIT-1 F-47).
_SHIPPED_MANIFEST = DEFAULT_MANIFEST_PATH


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


# ── item 9 (R4/LAW-07) — the heartbeat family is registered IN FULL ───────────────────
def test_every_armed_heartbeat_source_has_a_manifest_row() -> None:
    """Each member of `HEARTBEAT_SOURCES` carries a `heartbeat.<source>` row.

    MEASURED GAP THIS CLOSES: three of the four sources had rows and `eval_round` — added
    as the fourth at WP11-A — did not, while being armed identically (a deadline in
    `MonitorConfig`, a `wired_sources` entry from `mantis.run`, and
    `WATCHDOG_STALL_EXIT_CODE` on staleness). That is the F-10 shape: an input that can stop
    a run, citing no producer and no producer test. Nothing could see it —
    `test_shipped_manifest_every_row_resolves` resolves the rows it is HANDED and an absent
    row is not a row (the O-29 asymmetry, argued in full above), and
    `docs/contracts/event_manifest.md` tabulates `heartbeat.eval_round` among its shipped
    rows, so the doc AGREED the row existed.

    DERIVED, not transcribed (R192(e) / R8's derive-or-delete): the expectation comes from
    `HEARTBEAT_SOURCES` — the tuple the registry, the watchdog deadlines and the arm event
    all key on — so a FIFTH source added later without a row reds here on the commit that
    adds it. A transcribed list of four would have to be re-edited to notice, which is the
    same rot as an asserted line count. The naming convention `heartbeat.<source>` is the
    shipped one for all four and is asserted, not guessed at: a row that registered the
    source under some other id would satisfy no reader looking for it.

    MUTATIONS THAT RED IT: (1) delete the `heartbeat.eval_round` row from
    `producer_manifest.yaml` — `test_shipped_manifest_every_row_resolves` stays GREEN under
    it; (2) append a source to `HEARTBEAT_SOURCES` without adding its row.
    """
    from mantis.monitor.heartbeat import HEARTBEAT_SOURCES

    assert HEARTBEAT_SOURCES, "premise: the watchdog declares heartbeat sources at all"
    present = {row["id"] for row in load_manifest(_SHIPPED_MANIFEST)["gates"]}
    missing = [s for s in HEARTBEAT_SOURCES if f"heartbeat.{s}" not in present]
    assert missing == [], (
        f"armed heartbeat sources with no producer-manifest row: {missing}. Each of these "
        "feeds the stall watchdog and can exit the run at WATCHDOG_STALL_EXIT_CODE, so each "
        "owes a live producer and a named producer test (R4/LAW-07). Rows present: "
        f"{sorted(present)}"
    )


# ── item 10(b) (R4/LAW-07) — the K histogram's row cannot be quietly deleted ──────────
def test_the_k_cluster_histogram_instrument_has_a_manifest_row() -> None:
    """The in-run K histogram carries a producer-manifest row, keyed on the emitter's OWN
    name for the field.

    THE ASYMMETRY THIS CLOSES (O-29, argued in full above): `verify_manifest` resolves the
    rows it is HANDED, and an absent row is not a row — so `test_shipped_manifest_every_row_
    resolves` stays GREEN under a deletion. Every registered family needs its own presence
    pin; this is the K histogram's.

    DERIVED, not transcribed: the expected row id comes from
    `mantis.train.events.K_CLUSTER_HISTOGRAM_KEY`, the single authority the emitter, the
    absence rule and the payload key all read, so renaming the field without renaming the row
    reds here rather than leaving a row that resolves and describes nothing. The PRODUCER is
    asserted to be the emitter function itself for the same reason resolution alone is not
    enough: a row re-pointed at some other live symbol resolves perfectly and cites the wrong
    thing.

    MUTATIONS THAT RED IT: (1) delete the `k_cluster_histogram` row from
    `producer_manifest.yaml`; (2) re-point its producer symbol at anything but the emitter.

    RECORDED, NOT FIXED HERE (out of this card's scope): the sibling R250 subtraction — the
    `iteration_complete` cluster block landed at ADJ-D32 — has NO manifest row at all, so
    this pin is deliberately single-row rather than derived over the whole R250 absence
    family. A family-derived expectation would red on that gap today.
    """
    from mantis.train.events import K_CLUSTER_HISTOGRAM_KEY

    rows = {row["id"]: row for row in load_manifest(_SHIPPED_MANIFEST)["gates"]}
    assert K_CLUSTER_HISTOGRAM_KEY in rows, (
        f"the in-run K histogram publishes `iteration_complete.{K_CLUSTER_HISTOGRAM_KEY}` and "
        f"owes a live producer plus a named producer test (R4/LAW-07). Rows present: "
        f"{sorted(rows)}"
    )
    producer = rows[K_CLUSTER_HISTOGRAM_KEY]["producer"]
    assert producer["module"] == "mantis.train.events", producer
    assert producer["symbol"] == "k_cluster_histogram_block", (
        f"the row must cite the function that BUILDS the field, not a symbol that merely "
        f"resolves; got {producer['symbol']!r}"
    )


# ── AUDIT-1 F-10 (+ its sibling GATE-C03): the two ways a row resolved against nothing ──
#
# `_verify_event_literal` was `re.search` over RAW MODULE SOURCE, so ANY occurrence of the
# quoted literal satisfied the row — including one inside a docstring. `eval/pipeline.py`'s
# module docstring contains `heartbeat("eval_round")` as PROSE while the live producer is
# `EvalPipeline._poll_loop`'s `self._beat("eval_round")`: delete the call and the row still
# resolved. That is precisely the class the manifest exists to make un-shippable — a producer
# vanishing with the gate green.
#
# `_verify_producer_test` found a `test_*` FunctionDef and stopped there, so a cited test
# carrying `@pytest.mark.slow` / `skip` / `skipif` / `integration` — or a module-level
# `pytestmark` with one — satisfied the row while running in no default-tier invocation.

def _module_row(tmp_path: Path, module: str, literal: str) -> Path:
    return _write_manifest(tmp_path / "m.yaml", [
        "  - id: literal_row\n"
        f"    producer: {{kind: event_literal, module: {module}, literal: {literal}}}\n"
        f"    producer_test: {_LIVE_TEST_NODE}\n",
    ])


def _fake_module(tmp_path: Path, name: str, body: str) -> None:
    """Write an importable module into a directory on `sys.path` for the duration of a test."""
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")


def test_a_literal_that_appears_ONLY_in_a_docstring_does_not_satisfy_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE PIN (F-10). Documented is not live."""
    import sys

    _fake_module(tmp_path, "_f10_docstring_only", '''"""A module whose only mention of
    heartbeat("ghost_beat") is this sentence."""


def beat() -> None:
    return None
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("_f10_docstring_only", None)
    with pytest.raises(ManifestError) as ei:
        verify_manifest(_module_row(tmp_path, "_f10_docstring_only", "ghost_beat"), _REPO_ROOT)
    assert "literal_row" in str(ei.value)
    assert "docstring" in str(ei.value), (
        "the refusal must say WHY: a reader who hit this needs to know the literal IS in the "
        "file"
    )


def test_a_literal_in_LIVE_code_still_satisfies_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control, and it is the load-bearing one: the tightened check must not reject the
    real producers. Same module, same literal, moved from prose into a call."""
    import sys

    _fake_module(tmp_path, "_f10_live_literal", '''"""A module with a live beat."""


def beat(sink) -> None:
    sink.emit("ghost_beat")
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("_f10_live_literal", None)
    verify_manifest(_module_row(tmp_path, "_f10_live_literal", "ghost_beat"), _REPO_ROOT)


def test_a_function_docstring_does_not_satisfy_a_row_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docstrings are excluded STRUCTURALLY — module, class and function alike — rather than
    by "is it near the top of the file"."""
    import sys

    _fake_module(tmp_path, "_f10_func_docstring", '''"""Module."""


class Thing:
    """Class."""

    def beat(self) -> None:
        """Emits nested_ghost when it fires."""
        return None
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("_f10_func_docstring", None)
    with pytest.raises(ManifestError):
        verify_manifest(_module_row(tmp_path, "_f10_func_docstring", "nested_ghost"), _REPO_ROOT)


@pytest.mark.parametrize("marker", ["slow", "skip", "integration"])
def test_a_producer_test_DESELECTED_from_the_tier_does_not_satisfy_a_row(
    tmp_path: Path, marker: str
) -> None:
    """GATE-C03. A producer test that does not run is the same evidence as no producer test."""
    suite = tmp_path / "tests" / "monitor"
    suite.mkdir(parents=True)
    module = suite / "test_f10_deselected.py"
    # The decorator is assembled rather than written inline: a literal "\n" immediately
    # before "@pytest.mark" matches CI gate 17's `user@host` class, and a fixture that
    # trips a gate teaches the next reader to add a hatch reflexively.
    mark = "@pytest.mark"
    module.write_text(
        f"import pytest\n\n\n{mark}.{marker}\ndef test_deselected_producer() -> None:\n"
        "    assert True\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: deselected_row\n"
        "    producer: {kind: symbol, module: mantis.train.checkpoints, "
        "symbol: persist_errors_total}\n"
        "    producer_test: tests/monitor/test_f10_deselected.py::test_deselected_producer\n",
    ])
    with pytest.raises(ManifestError) as ei:
        verify_manifest(manifest, tmp_path)
    assert "deselected_row" in str(ei.value)
    assert marker in str(ei.value), str(ei.value)


def test_a_module_level_pytestmark_is_caught_too(tmp_path: Path) -> None:
    """The decorator is the obvious form; `pytestmark = [pytest.mark.slow]` deselects the
    WHOLE module and is the one a reader scanning the function would miss."""
    suite = tmp_path / "tests" / "monitor"
    suite.mkdir(parents=True)
    (suite / "test_f10_modmark.py").write_text(
        "import pytest\n\npytestmark = [pytest.mark.slow]\n\n\n"
        "def test_module_marked_producer() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: modmark_row\n"
        "    producer: {kind: symbol, module: mantis.train.checkpoints, "
        "symbol: persist_errors_total}\n"
        "    producer_test: tests/monitor/test_f10_modmark.py::test_module_marked_producer\n",
    ])
    with pytest.raises(ManifestError) as ei:
        verify_manifest(manifest, tmp_path)
    assert "slow" in str(ei.value)


def test_an_UNMARKED_producer_test_still_satisfies_a_row(tmp_path: Path) -> None:
    """The control for the sibling: only DESELECTING markers are refused. A test carrying, say,
    `@pytest.mark.parametrize` runs in the default tier and is fine."""
    suite = tmp_path / "tests" / "monitor"
    suite.mkdir(parents=True)
    # The decorator is assembled rather than written inline: a literal "\n" immediately
    # before "@pytest.mark" matches CI gate 17's `user@host` class, and a fixture that
    # trips a gate teaches the next reader to add a hatch reflexively.
    mark = "@pytest.mark"
    (suite / "test_f10_ok.py").write_text(
        f"import pytest\n\n\n{mark}.parametrize(\"n\", [1, 2])\n"
        "def test_live_producer(n: int) -> None:\n    assert n\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / "m.yaml", [
        "  - id: ok_row\n"
        "    producer: {kind: symbol, module: mantis.train.checkpoints, "
        "symbol: persist_errors_total}\n"
        "    producer_test: tests/monitor/test_f10_ok.py::test_live_producer\n",
    ])
    verify_manifest(manifest, tmp_path)
