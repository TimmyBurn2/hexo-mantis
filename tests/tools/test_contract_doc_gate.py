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


def test_a_stale_leaf_count_reds_the_gate(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "**170 leaf key-paths**", "**148 leaf key-paths**")
    res = _run(doc)
    assert res.returncode == 1
    assert "states 148 leaf key-paths" in res.stdout


def test_a_doc_that_states_no_count_reds_the_gate(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "**170 leaf key-paths**", "a lot of leaves")
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
