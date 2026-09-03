"""O6 — resolved-config emit (emit.ResolvedConfig.to_event_payload; B1 8-knob payload)
and O7 — death-of-merge census (grep-gate + mutation self-test).

The merge/layer-reconstruct machinery is deleted; emit is thin per-knob (value, source)
tagging. The payload carries EXACTLY the 7 schema leaves (source="file") plus the derived
amp_dtype (source="derived") = 8 knobs (WPSC Phase 2 SC-A2: `selfplay.
legal_move_radius_schedule` dropped out of the schema entirely, DESIGN_P2.md §5/§9 — no
replacement leaf). The 7-key schema portion is identical to O15's CONSUMER_REGISTRY's
original WP8 set (B1 — no phantom emit consumer).
"""
from pathlib import Path

import pytest

from mantis.config.emit import ResolvedConfig, ResolvedKnob, resolve_config
from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

_EIGHT_SCHEMA_LEAVES = {
    "schema_version",
    "run_id",
    "seed",
    "identity.encoding",
    "identity.representation",
    "eval.random_model_sims",
    "eval.sealbot_model_sims",
}


def _run5() -> ResolvedConfig:
    return resolve_config(load_config(REPO_ROOT / "configs" / "run5.yaml"))


# ── O6 emit ────────────────────────────────────────────────────────────────
def test_payload_event_and_eight_knob_key_set():
    payload = _run5().to_event_payload()
    assert payload["event"] == "resolved_config"
    assert set(payload["knobs"]) == _EIGHT_SCHEMA_LEAVES | {"amp_dtype"}


def test_payload_pins_production_values():
    """The EMIT carries the config's own values — a transport assertion, not a mint assertion.

    AUDIT-1 F-49: `== 96` / `== 128` were typed here, a third copy of run5's minted sims with
    no provenance line. What this test is for is that `to_event_payload` does not transform or
    drop a value on the way out, so it compares the payload to the LOADED CONFIG. The one
    provenance pin for 96/128 with its grounds is
    `tests/config/test_eval_config_remint.py::test_run3_parity_values_pinned`.
    """
    cfg = load_config(REPO_ROOT / "configs" / "run5.yaml")
    knobs = _run5().to_event_payload()["knobs"]
    assert knobs["schema_version"]["value"] == cfg.schema_version
    assert knobs["identity.encoding"]["value"] == cfg.identity.encoding
    assert knobs["identity.representation"]["value"] == cfg.identity.representation
    assert knobs["eval.random_model_sims"]["value"] == cfg.eval.random_model_sims
    assert knobs["eval.sealbot_model_sims"]["value"] == cfg.eval.sealbot_model_sims
    assert knobs["amp_dtype"]["value"] == "bf16"


def test_schema_leaves_are_file_source_amp_is_derived():
    rc = _run5()
    for leaf in _EIGHT_SCHEMA_LEAVES:
        assert rc.provenance(leaf).source == "file"
    assert rc.provenance("amp_dtype").source == "derived"


def test_encoding_source_remap_variant_to_file():
    # NIT-2: the encoding resolver returns source="variant"; emit remaps variant->file.
    assert _run5().provenance("identity.encoding").source == "file"


def test_one_knob_mutation_reflected_in_payload():
    cfg = load_config(REPO_ROOT / "configs" / "run5.yaml")
    mutated = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert resolve_config(mutated).to_event_payload()["knobs"]["seed"]["value"] == cfg.seed + 1


def test_payload_has_no_merge_provenance_or_checkpoint_source():
    knobs = _run5().to_event_payload()["knobs"]
    for rec in knobs.values():
        assert set(rec) == {"value", "source"}
        assert rec["source"] != "checkpoint"


def test_provenance_unknown_knob_raises():
    with pytest.raises(KeyError):
        _run5().provenance("no_such_knob")


def test_resolved_knob_shape():
    kb = ResolvedKnob(value=1, source="file")
    assert kb.value == 1 and kb.source == "file"


# ── O7 death-of-merge census (grep-gate, LAW-07 mutation self-test) ─────────
_FORBIDDEN = (
    "capture_config_layers",
    "merged_layers",
    "assert_layers_reconstruct",
    "_variant_layers_only",
    "_lookup_in_layers",
    "_deep_merge",
    "resolve_preload_config",
    "_PRELOAD_SEED_DEFAULT",
)

_CONFIG_PKG = REPO_ROOT / "src" / "mantis" / "config"


def _scan(text: str) -> set[str]:
    return {sym for sym in _FORBIDDEN if sym in text}


def test_no_merge_machinery_in_config_package():
    offenders = {}
    for py in _CONFIG_PKG.rglob("*.py"):
        found = _scan(py.read_text())
        if found:
            offenders[str(py)] = found
    assert not offenders, f"deleted merge machinery resurrected: {offenders}"


def test_census_checker_bites():
    # Mutation self-test: a source that reintroduces any symbol MUST be flagged (LAW-07).
    assert _scan("x = _deep_merge(a, b)") == {"_deep_merge"}
    assert _scan("_PRELOAD_SEED_DEFAULT = 42") == {"_PRELOAD_SEED_DEFAULT"}
    assert _scan("clean source with no forbidden symbol") == set()
