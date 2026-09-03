"""Test the tester: CI gate 13 must bite on a stale contract-doc citation (LAW-07).

The gate exists because contract #5's doc drifted through four schema versions with every gate
green — repo_design §4's v2 amendment MEASURED that nothing in the repo named the file. A gate
added to close that class is worthless unless its trigger is itself demonstrated, so every arm
below is a MUTATION with its counterexample beside it, in both directions:

* the DOC side — a phantom key, a retired symbol, a wrong count, a live key smuggled into the
  "deliberately absent" list, and the heading deleted;
* the SCHEMA side — a new leaf on `RunConfig` reds the gate against the UNMODIFIED shipped doc.
  That arm is the one that proves the gate reads the live authority rather than a transcription:
  a gate built on a copied key list would stay green through it.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import create_model

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "contract_doc_gate.py"
REAL_DOC = REPO_ROOT / "docs" / "contracts" / "run_config_schema.md"


def _run(doc: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE_PATH), "--doc", str(doc)],
        capture_output=True, text=True, check=False, cwd=REPO_ROOT,
    )


@pytest.fixture
def gate_module():
    spec = importlib.util.spec_from_file_location("contract_doc_gate", GATE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def doc_text() -> str:
    return REAL_DOC.read_text(encoding="utf-8")


def _mutate(tmp_path: Path, text: str, old: str, new: str, count: int = -1) -> Path:
    """Write a mutated copy of the doc, REFUSING a no-op edit.

    A mutation arm whose `str.replace` silently matched nothing is a test that passes for the
    wrong reason — the exact vacuity class R87 exists to catch — so the substring is asserted
    present before the edit.
    """
    assert old in text, f"mutation anchor not present in the shipped doc: {old!r}"
    mutated = text.replace(old, new, count) if count >= 0 else text.replace(old, new)
    assert mutated != text
    path = tmp_path / "run_config_schema.md"
    path.write_text(mutated, encoding="utf-8")
    return path


# ── the negative pole: the shipped doc is clean ────────────────────────────────────────

def test_the_shipped_contract_doc_passes_the_gate():
    res = _run(REAL_DOC)
    assert res.returncode == 0, res.stdout + res.stderr


def test_an_absent_doc_is_a_named_failure_not_a_silent_pass(tmp_path):
    res = _run(tmp_path / "nope.md")
    assert res.returncode == 2
    assert "does not exist" in res.stdout


# ── doc-side mutations ─────────────────────────────────────────────────────────────────

def test_a_phantom_config_key_reds_the_gate(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "`train.max_train_steps`", "`train.no_such_knob`", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "train.no_such_knob" in res.stdout
    assert "not a key path of RunConfig" in res.stdout


def test_a_retired_resolver_symbol_reds_the_gate(tmp_path, doc_text):
    # The exact citation that was FALSE in the working copy this gate was written from.
    doc = _mutate(tmp_path, doc_text, "mantis.config.resolve.nsims",
                  "mantis.config.resolve.radius.resolve_radius_from_schedule", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "resolve_radius_from_schedule" in res.stdout
    assert "does not resolve" in res.stdout


#: The doc's own stated leaf count, DERIVED from the live schema rather than transcribed.
#: A literal here is exactly the defect gate 13 exists to catch, one layer up: it goes stale
#: the moment any phase adds a leaf, and it goes stale SILENTLY as a mutation anchor that
#: matches nothing — which `_mutate` refuses, but only after the arm has stopped testing what
#: it names. WP12-R F2 is the phase that made this concrete (174 -> 176).
#:
#: The gate module arrives through the `gate_module` FIXTURE, which loads it by
#: `importlib.util.spec_from_file_location` (`:37`). The first version of this helper did
#: `from tools.ci_gates.contract_doc_gate import _leaf_paths` instead, and that import is
#: UNRESOLVABLE under the project's own test command: `tools/` and `tools/ci_gates/` carry no
#: `__init__.py`, so the name only resolves as a PEP 420 namespace package when the repo root
#: happens to be on `sys.path` — true under `python -m pytest` (which prepends CWD), FALSE
#: under `uv run pytest` (a console script, which does not). It was the only `from tools.`
#: import in the suite. Recorded here rather than only in a log: a helper whose import cannot
#: resolve is indistinguishable from one that was never called.
def _live_count_claim(gate_module) -> str:
    """The count read through the SAME symbol the gate uses, so this can never drift from it.

    AUDIT-1 F-44: the gate's private `_leaf_paths` is gone and `mantis.config.schema.leaf_paths`
    is the one walker. It is read off `gate_module` rather than imported here, so a gate that
    swapped in a different walker would still be measured by ITS walker, which is the property
    the fixture-loaded module exists to give.
    """
    from mantis.config.schema import RunConfig
    return f"**{len(gate_module.leaf_paths(RunConfig))} leaf key-paths**"


def test_a_stale_leaf_count_reds_the_gate(tmp_path, doc_text, gate_module):
    doc = _mutate(tmp_path, doc_text, _live_count_claim(gate_module),
                  "**148 leaf key-paths**")
    res = _run(doc)
    assert res.returncode == 1
    assert "states 148 leaf key-paths" in res.stdout


def test_a_doc_that_states_no_count_reds_the_gate(tmp_path, doc_text, gate_module):
    doc = _mutate(tmp_path, doc_text, _live_count_claim(gate_module), "a lot of leaves")
    res = _run(doc)
    assert res.returncode == 1
    assert "does not state its leaf-key-path count" in res.stdout


# ── the reversed region: "deliberately absent" is checked, not exempted ────────────────

def test_a_live_key_listed_as_deliberately_absent_reds_the_gate(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "**`eval.gate.screen_confirm_hi`.**",
                  "**`train.batch_size`.**", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "train.batch_size" in res.stdout
    assert "but RunConfig HAS it" in res.stdout


def test_deleting_the_absent_heading_cannot_silently_retire_the_reversed_check(
    tmp_path, doc_text
):
    doc = _mutate(tmp_path, doc_text, "## Deliberately absent", "## Notes on absence")
    res = _run(doc)
    assert res.returncode == 1
    assert "has no \"## Deliberately absent\" section" in res.stdout


def test_the_absent_section_accepts_a_key_that_really_is_gone(tmp_path, doc_text):
    """The counterexample to the arm above: the reversed check is a real discriminator, not a
    blanket refusal of every key path under the heading."""
    doc = _mutate(tmp_path, doc_text, "**`eval.gate.screen_confirm_hi`.**",
                  "**`eval.gate.no_such_dead_knob`.**", 1)
    assert _run(doc).returncode == 0


# ── the bare-symbol arm (WPCLEAN Phase RES — the DSV2-2 blind-spot closure) ────────────

def test_a_dead_validator_name_in_the_claim_column_reds_the_gate(tmp_path, doc_text):
    """The DSV2-2 reproduction, now with the opposite verdict: at WPMINT close-out a doc
    naming a validator that does not exist left this gate at rc 0 (the recorded blind
    spot). The bare-symbol arm makes exactly that mutation red."""
    doc = _mutate(tmp_path, doc_text, "`_draw_rate_evidence_bar_within_configured_capacity`",
                  "`_draw_rate_floor_validator_that_never_existed`", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "_draw_rate_floor_validator_that_never_existed" in res.stdout
    assert "not defined anywhere in mantis.config.schema" in res.stdout


def test_a_stale_bare_name_in_prose_stays_clean_the_stated_bound(tmp_path, doc_text):
    """The discriminating negative that DOCUMENTS the arm's bound: `min_samples` is a
    retired key cited as history in the version table and in a rule cell's prose — both
    legitimate, both outside the claim columns, both must stay rc 0. The arm is a
    claim-column check, not a doc-wide truth oracle (the gate docstring says so)."""
    assert "`min_samples`" in doc_text  # the historical citations are really there
    assert _run(REAL_DOC).returncode == 0


def test_emptying_the_cross_field_table_cannot_silently_retire_the_arm(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "## Cross-field rules (the invariants no single field can carry)",
                  "## Former rules table")
    res = _run(doc)
    assert res.returncode == 1
    assert "bare-symbol arm" in res.stdout


def test_a_dead_model_name_in_the_second_cell_reds_the_gate(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "| `_entropy_sign` | `TrainConfig` |",
                  "| `_entropy_sign` | `RetiredTrainConfig` |", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "RetiredTrainConfig" in res.stdout


# ── the schema-side mutation: the gate reads the LIVE authority ────────────────────────

def test_the_unmutated_schema_agrees_with_the_shipped_doc(gate_module):
    assert gate_module.check(REAL_DOC) == []


def test_a_new_leaf_on_RunConfig_reds_the_gate_against_the_UNCHANGED_doc(gate_module):
    """The gate-12 pattern's own proof. Nothing about the doc changes here — only the schema —
    and the gate must notice. A gate built over a transcribed key list would stay green."""
    live = gate_module.RunConfig
    mutated = create_model("_MutatedRunConfig", __base__=live, phantom_leaf=(int, ...))
    gate_module.RunConfig = mutated
    try:
        failures = gate_module.check(REAL_DOC)
    finally:
        gate_module.RunConfig = live
    assert failures, "a new schema leaf must red the contract-doc gate"
    assert any("leaf key-paths" in line for line in failures), failures
    # and the restoration is real, not assumed
    assert gate_module.check(REAL_DOC) == []
