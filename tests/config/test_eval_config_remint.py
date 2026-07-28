"""⊕ WP11-A — eval schema extension + config remint (CI gate 7 adjunct; O15 registry).

RED-at-import: NONE of the fields below (`eval.gate`, `eval.ladder`, `eval.kraken_model_sims`,
`eval.strix_model_sims`, `eval.random_floor_games`, `eval.worker_device`,
`eval.round_timeout_sec`, `eval.worker_kill_grace_sec`) exist on today's `EvalConfig`
(src/mantis/config/schema.py, already read — today's EvalConfig has exactly
`random_model_sims` + `sealbot_model_sims`). Every config file at HEAD (configs/*.yaml,
tools/config_templates/{dev,grid}.yaml) also lacks them. `mantis.config.loader.load_config`
and `mantis.config.schema.RunConfig` ALREADY EXIST and import fine — every test here is
RED-BY-ASSERTION (a `pydantic.ValidationError` on load, or an equality/membership assertion
that is false today), never a collection-time ModuleNotFoundError. `configs/run5.yaml` is
the run3-PARITY config (its ladder is asserted verbatim against the six STATE §5 rungs
elsewhere in this WP — design §b); `dev_example.yaml` / `smoke_gnn.yaml` /
`smoke_radius_curriculum.yaml` / `sustained_kcluster.yaml` are the dev/smoke templates
(adjudication A-2: parity mints `random_floor_games=0`, dev/smoke mint `4`).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import RunConfig

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO / "configs"
_TEMPLATES_DIR = _REPO / "tools" / "config_templates"

_PARITY_CONFIG = _CONFIGS_DIR / "run5.yaml"
_DEV_SMOKE_CONFIGS = (
    _CONFIGS_DIR / "dev_example.yaml",
    _CONFIGS_DIR / "smoke_gnn.yaml",
    _CONFIGS_DIR / "smoke_radius_curriculum.yaml",
    _CONFIGS_DIR / "sustained_kcluster.yaml",
)

# The run3-parity gate recipe (deploy_strength_eval.py:249-274, round_robin.py:168; adjudication
# A-3 for seed_base). NO screen_confirm_hi key (MUST-FIX 1 — inert in run3, non-ported).
_PARITY_GATE = {
    "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
    "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
    "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625,
}

# The six STATE §5 rungs (design §c.1 "Minted ladder" — verbatim, order binding).
_PARITY_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "kraken_raw", "bot": "kraken", "variant": "raw", "depth": None,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "sealbot_d6", "bot": "sealbot", "variant": "d6", "depth": 6,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "kraken_mcts200", "bot": "kraken", "variant": "mcts200", "depth": None,
     "opponent_sims": 200, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "strix_128", "bot": "strix", "variant": "s128", "depth": None,
     "opponent_sims": 128, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "strix_256", "bot": "strix", "variant": "s256", "depth": None,
     "opponent_sims": 256, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
]

_PARITY_LADDER = {
    "rungs": _PARITY_RUNGS, "round_games": 64, "min_games_per_active_rung": 4,
    "graduation_wr_lower_ci": 0.75, "graduation_consec_rounds": 3,
    "activation_wr_lower_ci": 0.65, "calibration_every_k_rounds": 4,
    "calibration_games": 8, "bootstrap_resamples": 1000, "bootstrap_ci_level": 0.95,
    "bt_prior_games": 1.0, "bootstrap_seed": 1234,
}


def _config_paths() -> list[Path]:
    # ADJ-13 F-1 corrective pass (recheck R-5): the ONE discovery authority, not a
    # sixth flat glob. A flat `*.yaml` census is blind to `configs/prod/run6.yaml`,
    # which gate 7 and gate 12 both now make legal.
    paths = discover_configs(_CONFIGS_DIR)
    for name in ("dev.yaml", "grid.yaml"):
        p = _TEMPLATES_DIR / name
        if p.exists():
            paths.append(p)
    assert paths, "no configs/templates found — glob pattern or file-plan drifted"
    return paths


def test_all_configs_and_templates_carry_the_new_eval_block_and_validate() -> None:
    """CI gate 7 adjunct: every committed config + template must validate with `eval.ladder`
    present. RED today: none carry `eval.gate`/`eval.ladder`, so every load raises
    ValidationError (missing required fields)."""
    failures: list[str] = []
    for path in _config_paths():
        try:
            cfg = load_config(path)
        except ValidationError as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if cfg.eval.ladder is None or not cfg.eval.ladder.rungs:
            failures.append(f"{path.name}: eval.ladder present but empty/None")
    assert failures == [], (
        "every config/template must validate WITH a populated eval.ladder:\n"
        + "\n".join(failures)
    )


def test_run3_parity_values_pinned() -> None:
    """96/128 sims verbatim + the full run3-parity gate recipe, pinned against
    `configs/run5.yaml` — including seed_base=20260625 (run3's ACTUAL gate default,
    deploy_strength_eval.py:272 — adjudication A-3; NOT 42, the legacy evaluator's default,
    eval/defaults.py:31)."""
    cfg = load_config(_PARITY_CONFIG)
    assert cfg.eval.random_model_sims == 96
    assert cfg.eval.sealbot_model_sims == 128
    gate = cfg.eval.gate
    assert gate.screen_games == 80
    assert gate.confirm_games == 128
    assert gate.promotion_winrate == 0.55
    assert gate.screen_confirm_lo == 0.44
    assert gate.deploy_sims == 150
    assert gate.bootstrap_resamples == 1000
    assert gate.min_distinct_per_pair == 10
    assert gate.seed_base == 20260625, (
        "seed_base must be run3's ACTUAL gate default (20260625, adjudication A-3), not 42"
    )
    assert not hasattr(gate, "screen_confirm_hi"), (
        "screen_confirm_hi is a deliberate non-port (MUST-FIX 1) — must not exist as a field"
    )


def test_screen_confirm_hi_key_is_rejected_everywhere() -> None:
    """A minted config carrying `screen_confirm_hi` (the inert, non-ported run3 knob) must
    be REJECTED by `extra="forbid"` — constructed directly against a GateConfig-shaped
    payload (schema-level; no need for a full RunConfig round trip)."""
    from mantis.config.schema import GateConfig  # RED-at-import: not yet defined

    payload = dict(_PARITY_GATE)
    payload["screen_confirm_hi"] = 1.0
    with pytest.raises(ValidationError):
        GateConfig.model_validate(payload)


def test_minted_configs_carry_the_ladder_verbatim() -> None:
    """`configs/run5.yaml`'s ladder must equal the six STATE §5 rungs, in order, verbatim
    (@128/@256 are DISTINCT rungs); 0.75/0.65/3 must appear ONLY as field VALUES, never as
    source-code literals under src/mantis/eval (rule 4 — schema fields, not literals)."""
    cfg = load_config(_PARITY_CONFIG)
    ladder = cfg.eval.ladder
    assert [r.name for r in ladder.rungs] == [r["name"] for r in _PARITY_RUNGS], (
        "rung order must match STATE §5 verbatim"
    )
    for got, want in zip(ladder.rungs, _PARITY_RUNGS):
        assert got.bot == want["bot"]
        assert got.variant == want["variant"]
        assert got.opponent_sims == want["opponent_sims"]
        assert got.deploy_matched is True
    assert ladder.graduation_wr_lower_ci == 0.75
    assert ladder.activation_wr_lower_ci == 0.65
    assert ladder.graduation_consec_rounds == 3

    eval_src_dir = _REPO / "src" / "mantis" / "eval"
    if eval_src_dir.exists():
        for path in sorted(eval_src_dir.rglob("*.py")):
            text = path.read_text()
            assert "0.75" not in text, f"{path}: 0.75 must be a schema field value, not a literal"
            assert "0.65" not in text, f"{path}: 0.65 must be a schema field value, not a literal"


def test_parity_config_mints_random_floor_disabled_and_dev_smoke_enabled() -> None:
    """Adjudication A-2: the run3-parity config mints `random_floor_games=0` (run3 disabled
    random); dev/smoke templates mint a small non-zero n (4) so the headless end-to-end
    round exercises a REAL bot and the resolver key has a live EXERCISED consumer."""
    parity_cfg = load_config(_PARITY_CONFIG)
    assert parity_cfg.eval.random_floor_games == 0, (
        "the run3-parity config must mint random_floor_games=0 (random was disabled in run3)"
    )
    for path in _DEV_SMOKE_CONFIGS:
        if not path.exists():
            continue
        cfg = load_config(path)
        assert cfg.eval.random_floor_games == 4, (
            f"{path.name}: dev/smoke templates must mint random_floor_games=4 (A-2)"
        )


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Leaf key-paths of a StrictModel; recurse into nested models but stop at any field
    whose annotation is not itself a BaseModel subclass (mirrors
    tests/config/test_every_key_has_consumer.py's `_leaf_paths` — reimplemented locally
    rather than imported across test modules, per house convention)."""
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out.extend(_leaf_paths(ann, path))
        else:
            out.append(path)
    return out


# The NEW leaf keys this WP's schema extension must introduce (forward-looking pin: asserts
# the FUTURE schema's leaf-set, not today's — RED today because none of these leaves exist
# yet on RunConfig, so `_leaf_paths(RunConfig)` today is just the 8 WP8 leaves).
_NEW_LEAF_CONSUMERS = {
    "eval.kraken_model_sims": "resolve_eval_model_sims (kraken rungs)",
    "eval.strix_model_sims": "resolve_eval_model_sims (strix rungs)",
    "eval.random_floor_games": "worker.py random-floor block game count",
    "eval.worker_device": "build_eval_pipeline child-process device",
    "eval.round_timeout_sec": "pipeline.py mid-round subprocess join bound",
    "eval.worker_kill_grace_sec": "pipeline.py terminate->kill grace",
    "eval.gate.stride": "pipeline.py promotion-capable round stride",
    "eval.gate.screen_games": "worker.py gate screen block",
    "eval.gate.confirm_games": "worker.py gate confirm block",
    "eval.gate.promotion_winrate": "aggregate.py gate truth table",
    "eval.gate.screen_confirm_lo": "aggregate.py escalation test",
    "eval.gate.deploy_sims": "arena/deploy_head.py sims",
    "eval.gate.opening_book": "arena/books.py gate openings",
    "eval.gate.bootstrap_resamples": "aggregate.py bootstrap CI",
    "eval.gate.min_distinct_per_pair": "aggregate.py low-power guard",
    "eval.gate.seed_base": "aggregate.py bootstrap seed + worker.py opening seeds",
    "eval.ladder.rungs": "ladder.py LadderState rungs",
    "eval.ladder.round_games": "ladder.py allocate_games budget",
    "eval.ladder.min_games_per_active_rung": "ladder.py allocate_games floor",
    "eval.ladder.graduation_wr_lower_ci": "ladder.py graduation transition",
    "eval.ladder.graduation_consec_rounds": "ladder.py graduation streak",
    "eval.ladder.activation_wr_lower_ci": "ladder.py activation transition",
    "eval.ladder.calibration_every_k_rounds": "ladder.py calibration cadence",
    "eval.ladder.calibration_games": "ladder.py calibration allocation",
    "eval.ladder.bootstrap_resamples": "bt.py / aggregate.py bootstrap",
    "eval.ladder.bootstrap_ci_level": "aggregate.py pair_bootstrap_wr_ci",
    "eval.ladder.bt_prior_games": "bt.py fit_bt prior",
    "eval.ladder.bootstrap_seed": "aggregate.py pair_bootstrap_wr_ci seed",
}


def test_new_keys_all_have_consumers_in_o15_registry() -> None:
    """Forward-looking pin: once the schema extension lands, every new EvalConfig/
    GateConfig/LadderConfig leaf must be present — no more, no fewer than the registry this
    test defines. RED today: `_leaf_paths(RunConfig)` today has none of these leaves (the
    schema hasn't been extended), so the intersection assertion below is false."""
    leaves = set(_leaf_paths(RunConfig))
    missing = set(_NEW_LEAF_CONSUMERS) - leaves
    assert missing == set(), (
        f"new schema leaves not yet present (expected once EvalConfig/GateConfig/"
        f"LadderConfig land): {sorted(missing)}"
    )


def test_configs_have_no_unminted_manual_edits_signature() -> None:
    """Every re-minted config keeps the `minted-by: tools/mint_config.py` header stamp
    (repo_design §5 — configs are minted, never hand-varied) even after the eval-block
    remint; a config missing the stamp would indicate a manual edit slipped past minting."""
    for path in _config_paths():
        if path.parent == _TEMPLATES_DIR:
            continue  # templates are hand-authored sources, not minted outputs
        text = path.read_text()
        assert "minted-by: tools/mint_config.py" in text, (
            f"{path.name}: missing the mint-provenance header stamp"
        )
