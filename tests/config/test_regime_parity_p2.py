"""SC-A4 oracle — O9-O12 regime parity, POST-radius-removal (DESIGN_P2.md §5.1 /
PREREG_P2.md suite #8; rewrite of tests/config/test_regime_parity.py).

DEVIATION FROM PREREG PATH (logged in ORACLE_NOTES_P2.md): PREREG names this suite's home
as the existing `tests/config/test_regime_parity.py`. ORACLE-WRITE's writable surface is
NEW files only — this is therefore a new file, not an edit; IMPL retires the old file's
O12 content at port time. Reuses the `production_config` fixture already defined in
`tests/config/conftest.py` (loads `configs/run5.yaml`) — no import needed, pytest
auto-discovers sibling-directory conftest fixtures.

O9 (sims)/O10 (amp=bf16)/O11 (encoding) are UNCHANGED by Phase 2 — GREEN today, and must
stay green. O12 (radius) is REPLACED, not edited: the OLD "non-null schedule scan"
assertion is retired outright (DESIGN_P2.md §5, shape (ii) — no schema field survives to
scan); the new O12 assertion is `production_config` has no `selfplay.legal_move_radius_*`
field at all — RED at HEAD (the field is still present).
"""
from pathlib import Path

from mantis.config.loader import load_config
from mantis.config.resolve.amp import resolve_amp_dtype
from mantis.config.resolve.nsims import resolve_eval_model_sims

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── O9-O11 — UNCHANGED by Phase 2 ───────────────────────────────────────────────────────
def test_o9_sims_regime_parity_unchanged(production_config):
    assert production_config.eval.random_model_sims == 96
    assert production_config.eval.sealbot_model_sims == 128
    assert resolve_eval_model_sims("random", production_config.eval.random_model_sims) == 96
    assert resolve_eval_model_sims("sealbot", production_config.eval.sealbot_model_sims) == 128


def test_o10_amp_is_bf16_on_graph_unchanged(production_config):
    assert production_config.identity.representation == "graph"
    assert resolve_amp_dtype(production_config.identity.representation) == "bf16"


def test_o11_encoding_regime_parity_unchanged(production_config):
    assert production_config.identity.encoding == "gnn_axis_v1"
    assert production_config.identity.representation == "graph"


# ── O12 — REPLACED (radius field removed entirely, DESIGN_P2.md §5 shape (ii)) ─────────
def test_o12_production_config_has_no_radius_field_at_all(production_config):
    assert not hasattr(production_config.selfplay, "legal_move_radius_schedule")
    assert not hasattr(production_config.selfplay, "legal_move_radius")
    assert "legal_move_radius_schedule" not in type(production_config.selfplay).model_fields
    assert "legal_move_radius" not in type(production_config.selfplay).model_fields


def test_o12_smoke_radius_curriculum_loads_clean_and_keeps_v6w25_encoding():
    # DESIGN_P2.md §5.1 STOP CANDIDATE 2: post-SC-A4 this file's only remaining
    # distinguishing feature is its v6w25 encoding (the "non-null radius scan" it used to
    # exercise no longer exists anywhere in the repo). RESUME forbids retiring the file.
    smoke = load_config(REPO_ROOT / "configs" / "smoke_radius_curriculum.yaml")
    assert smoke.identity.encoding == "v6w25"
