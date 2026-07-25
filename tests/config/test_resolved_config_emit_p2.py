"""SC-A4 oracle — `emit.py` payload shrinks to 7 schema leaves + derived `amp_dtype` = 8
knobs (DESIGN_P2.md §9 / PREREG_P2.md suite #9; edit-target was
tests/config/test_resolved_config_emit.py).

DEVIATION FROM PREREG PATH (logged in ORACLE_NOTES_P2.md): PREREG marks this suite as an
edit to the existing file. ORACLE-WRITE's writable surface is NEW files only — this is a
new file pinning the POST-radius-removal payload shape; IMPL retires the old 9-knob
assertions in the existing file at port time.

RED at HEAD (not RED-at-import — `mantis.config.emit` already exists): today's payload
still carries `selfplay.legal_move_radius_schedule` (9 knobs), so every assertion below
about the 7-leaf/8-knob shape fails until SC-A4 lands. Per DESIGN_P2.md §9, following the
WP11-A precedent: NO new `train.*`/`selfplay.*`/`monitor.*` leaf is added to `emit.py`'s
payload — only the radius leaf's REMOVAL touches it.
"""
from pathlib import Path

from mantis.config.emit import resolve_config
from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

_SEVEN_SCHEMA_LEAVES = {
    "schema_version",
    "run_id",
    "seed",
    "identity.encoding",
    "identity.representation",
    "eval.random_model_sims",
    "eval.sealbot_model_sims",
}


def _run5_payload() -> dict:
    return resolve_config(load_config(REPO_ROOT / "configs" / "run5.yaml")).to_event_payload()


def test_payload_key_set_is_seven_leaves_plus_derived_amp_dtype():
    payload = _run5_payload()
    assert set(payload["knobs"]) == _SEVEN_SCHEMA_LEAVES | {"amp_dtype"}
    assert len(payload["knobs"]) == 8


def test_legal_move_radius_schedule_absent_from_payload():
    payload = _run5_payload()
    assert "selfplay.legal_move_radius_schedule" not in payload["knobs"]


def test_no_train_selfplay_monitor_leaves_threaded_into_payload():
    # Explicit negative assertion (WP11-A precedent, DESIGN_P2.md §9): none of the new
    # train.*/selfplay.*/monitor.* schema surface is added to emit.py's payload.
    payload = _run5_payload()
    for knob in payload["knobs"]:
        assert not knob.startswith("train."), knob
        assert not knob.startswith("selfplay."), knob
        assert not knob.startswith("monitor."), knob
        assert not knob.startswith("inference."), knob


def test_schema_leaves_still_file_source_amp_still_derived():
    payload = _run5_payload()
    for leaf in _SEVEN_SCHEMA_LEAVES:
        assert payload["knobs"][leaf]["source"] == "file"
    assert payload["knobs"]["amp_dtype"]["source"] == "derived"


def test_payload_pins_production_values_unchanged_by_radius_removal():
    knobs = _run5_payload()["knobs"]
    assert knobs["schema_version"]["value"] == 1
    assert knobs["identity.encoding"]["value"] == "gnn_axis_v1"
    assert knobs["identity.representation"]["value"] == "graph"
    assert knobs["eval.random_model_sims"]["value"] == 96
    assert knobs["eval.sealbot_model_sims"]["value"] == 128
    assert knobs["amp_dtype"]["value"] == "bf16"
