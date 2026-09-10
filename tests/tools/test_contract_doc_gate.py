"""Test the tester: CI gate 13 must bite on a stale contract-doc citation.

Every arm is a mutation with its counterexample beside it, in both directions:

* the DOC side — a phantom key, a retired symbol, a wrong count, a live key smuggled into the
  "deliberately absent" list, and the heading deleted;
* the SCHEMA side — a new leaf on `RunConfig` reds the gate against the UNMODIFIED shipped doc,
  which is what proves the gate reads the live authority rather than a transcribed key list.
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
    """Write a mutated copy of the doc, refusing a no-op edit.

    A `str.replace` that matched nothing would leave the arm passing for the wrong reason, so
    the anchor is asserted present first.
    """
    assert old in text, f"mutation anchor not present in the shipped doc: {old!r}"
    mutated = text.replace(old, new, count) if count >= 0 else text.replace(old, new)
    assert mutated != text
    path = tmp_path / "run_config_schema.md"
    path.write_text(mutated, encoding="utf-8")
    return path


def test_the_shipped_contract_doc_passes_the_gate():
    res = _run(REAL_DOC)
    assert res.returncode == 0, res.stdout + res.stderr


def test_an_absent_doc_is_a_named_failure_not_a_silent_pass(tmp_path):
    res = _run(tmp_path / "nope.md")
    assert res.returncode == 2
    assert "does not exist" in res.stdout


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


#: The doc's stated leaf count, derived from the live schema: a literal would go stale as a
#: mutation anchor that matches nothing, which is the defect gate 13 exists to catch.
#: The gate arrives through the `gate_module` fixture because `tools/` carries no `__init__.py`,
#: so `from tools.ci_gates...` resolves only when the repo root happens to be on `sys.path`.
def _live_count_claim(gate_module) -> str:
    """Return the count read through the same walker the gate uses, off `gate_module` itself."""
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
    """Prove the reversed check discriminates, rather than refusing every key under the heading."""
    doc = _mutate(tmp_path, doc_text, "**`eval.gate.screen_confirm_hi`.**",
                  "**`eval.gate.no_such_dead_knob`.**", 1)
    assert _run(doc).returncode == 0


def test_a_dead_validator_name_in_the_claim_column_reds_the_gate(tmp_path, doc_text):
    """Prove a doc naming a validator that does not exist reds the gate, not rc 0 as it once did."""
    doc = _mutate(tmp_path, doc_text, "`_draw_rate_evidence_bar_within_configured_capacity`",
                  "`_draw_rate_floor_validator_that_never_existed`", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "_draw_rate_floor_validator_that_never_existed" in res.stdout
    assert "not defined anywhere in mantis.config.schema" in res.stdout


def test_a_stale_bare_name_in_prose_stays_clean_the_stated_bound(tmp_path, doc_text):
    """Prove a retired name cited in prose stays clean: the arm checks claim columns, not the doc."""
    assert "`min_samples`" in doc_text  # the historical citations are really there
    assert _run(REAL_DOC).returncode == 0


def test_emptying_the_cross_field_table_cannot_silently_retire_the_arm(tmp_path, doc_text):
    doc = _mutate(tmp_path, doc_text, "## Cross-field rules (the invariants no single field can carry)",
                  "## Former rules table")
    res = _run(doc)
    assert res.returncode == 1
    assert "bare-symbol arm" in res.stdout


def test_a_dead_model_name_in_the_second_cell_reds_the_gate(tmp_path, doc_text):
    # The anchor must be a LIVE cross-field row.
    doc = _mutate(tmp_path, doc_text, "| `_stages_are_strictly_increasing` | `TrainConfig` |",
                  "| `_stages_are_strictly_increasing` | `RetiredTrainConfig` |", 1)
    res = _run(doc)
    assert res.returncode == 1
    assert "RetiredTrainConfig" in res.stdout


def test_the_unmutated_schema_agrees_with_the_shipped_doc(gate_module):
    assert gate_module.check(REAL_DOC) == []


def test_a_new_leaf_on_RunConfig_reds_the_gate_against_the_UNCHANGED_doc(gate_module):
    """Prove a new schema leaf reds the gate with the doc unchanged."""
    live = gate_module.RunConfig
    mutated = create_model("_MutatedRunConfig", __base__=live, phantom_leaf=(int, ...))
    gate_module.RunConfig = mutated
    try:
        failures = gate_module.check(REAL_DOC)
    finally:
        gate_module.RunConfig = live
    assert failures, "a new schema leaf must red the contract-doc gate"
    assert any("leaf key-paths" in line for line in failures), failures
    # The restoration is real, not assumed.
    assert gate_module.check(REAL_DOC) == []
