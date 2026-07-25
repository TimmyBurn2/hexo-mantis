"""O9–O11 — regime-parity per LAW knob (repo_design §8).

Each asserts *suite default == production default* over production_config() (configs/run5.yaml):
the suite expectation is DERIVED from the shipped config, never a hardcoded regime knob
(CONTEXT bug-class #5). Three §8 knobs remain here: sims (O9), amp=bf16 (O10), encoding
(O11). O12 (radius schedule) is RETIRED (WPSC Phase 2 SC-A2 forced-fallout: DESIGN_P2.md §5
removes `selfplay.legal_move_radius_schedule`/`RadiusStage` from the schema entirely — the
encoding registry alone is the radius authority, so there is no regime-parity knob left to
compare here). `tests/config/test_regime_parity_p2.py` (a later chunk's oracle) owns the
O12-replacement assertion.
"""
from pathlib import Path

from mantis.config.loader import load_config
from mantis.config.resolve.amp import resolve_amp_dtype
from mantis.config.resolve.nsims import resolve_eval_model_sims
from mantis.encoding import lookup

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_representation_matches_registry_for_every_config():
    # identity-key consistency (LAW-11): a config's representation must equal the registry's for
    # its encoding. This is the representation-consistency consumer named in O15's registry.
    configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert configs
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        spec = lookup(cfg.identity.encoding)
        assert cfg.identity.representation == spec.representation, cfg_path


def test_o9_sims_regime_parity(production_config):
    assert production_config.eval.random_model_sims == 96
    assert production_config.eval.sealbot_model_sims == 128
    # The resolver reads exactly the shipped config value (one authority; no eval-only re-derivation).
    assert resolve_eval_model_sims("random", production_config.eval.random_model_sims) == 96
    assert resolve_eval_model_sims("sealbot", production_config.eval.sealbot_model_sims) == 128


def test_o10_amp_is_bf16_on_graph(production_config):
    assert production_config.identity.representation == "graph"
    assert resolve_amp_dtype(production_config.identity.representation) == "bf16"


def test_o11_encoding_regime_parity(production_config):
    assert production_config.identity.encoding == "gnn_axis_v1"
    assert production_config.identity.representation == "graph"


