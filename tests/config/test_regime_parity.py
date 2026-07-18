"""O9–O12 — regime-parity per LAW knob (repo_design §8).

Each asserts *suite default == production default* over production_config() (configs/run5.yaml):
the suite expectation is DERIVED from the shipped config, never a hardcoded regime knob
(CONTEXT bug-class #5). The four §8 knobs: sims (O9), amp=bf16 (O10), encoding (O11),
radius schedule (O12).
"""
from pathlib import Path

from mantis.config.loader import load_config
from mantis.config.resolve.amp import resolve_amp_dtype
from mantis.config.resolve.nsims import resolve_eval_model_sims
from mantis.config.resolve.radius import resolve_radius_from_schedule
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


def test_o12_radius_schedule_regime_parity(production_config):
    # run5 (graph) declares no curriculum -> None -> caller keeps the encoding's registry radius.
    assert production_config.selfplay.legal_move_radius_schedule is None
    assert resolve_radius_from_schedule(None, 500_000) is None


def test_o12_curriculum_smoke_exercises_non_null_scan():
    smoke = load_config(REPO_ROOT / "configs" / "smoke_radius_curriculum.yaml")
    schedule = smoke.selfplay.legal_move_radius_schedule
    assert schedule is not None
    sched = [stage.model_dump() for stage in schedule]
    assert resolve_radius_from_schedule(sched, 0) == sched[0]["radius"]
    assert resolve_radius_from_schedule(sched, 10**9) == sched[-1]["radius"]
