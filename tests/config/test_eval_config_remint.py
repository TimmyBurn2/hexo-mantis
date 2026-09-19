"""The eval schema extension and the config remint that must carry it (CI gate 7 adjunct).

Every committed config and template must validate with the gate block, and the eval leaf keys
must each have a named consumer. (`eval.ladder` and `eval.sealbot_model_sims` were deleted with
the sealbot rung by R362(c); the parity pins below are what remains.)
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import RunConfig, leaf_paths

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO / "configs"
_TEMPLATES_DIR = _REPO / "tools" / "config_templates"

_PARITY_CONFIG = _CONFIGS_DIR / "run6.yaml"
_DEV_SMOKE_CONFIGS = (_CONFIGS_DIR / "dev_example.yaml",)

# The gate recipe. NO screen_confirm_hi key — it was inert and deliberately not ported.
_PARITY_GATE = {
    "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
    "screen_confirm_lo": 0.44, "deploy_sims": 160, "opening_book": "book_v1_s20260625_p4",
    "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625, "sequential": None,
}


def _config_paths() -> list[Path]:
    # the ONE discovery authority: a flat `*.yaml` census is blind to `configs/prod/`
    paths = discover_configs(_CONFIGS_DIR)
    for name in ("dev.yaml", "grid.yaml"):
        p = _TEMPLATES_DIR / name
        if p.exists():
            paths.append(p)
    assert paths, "no configs/templates found — glob pattern or file-plan drifted"
    return paths


def test_all_configs_and_templates_carry_the_new_eval_block_and_validate() -> None:
    """Every committed config and template validates with the gate block populated."""
    failures: list[str] = []
    for path in _config_paths():
        try:
            cfg = load_config(path)
        except ValidationError as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if cfg.eval.gate is None:
            failures.append(f"{path.name}: eval.gate missing")
    assert failures == [], (
        "every config/template must validate WITH the eval.gate block:\n"
        + "\n".join(failures)
    )


def test_run3_parity_values_pinned() -> None:
    """Pin the random sims and the full gate recipe against the production config."""
    cfg = load_config(_PARITY_CONFIG)
    assert cfg.eval.random_model_sims == 96
    gate = cfg.eval.gate
    assert gate.screen_games == 80
    assert gate.confirm_games == 128
    assert gate.promotion_winrate == 0.55
    assert gate.screen_confirm_lo == 0.44
    assert gate.deploy_sims == 160  # R346(b)/R348(e): eval deploy Gumbel 160 / m 16
    assert gate.bootstrap_resamples == 1000
    assert gate.min_distinct_per_pair == 10
    assert gate.seed_base == 20260625, (
        "seed_base must be run3's ACTUAL gate default (20260625, adjudication A-3), not 42"
    )
    assert not hasattr(gate, "screen_confirm_hi"), (
        "screen_confirm_hi is a deliberate non-port (MUST-FIX 1) — must not exist as a field"
    )


def test_screen_confirm_hi_key_is_rejected_everywhere() -> None:
    """A config carrying the non-ported `screen_confirm_hi` must be REJECTED by the schema."""
    from mantis.config.schema import GateConfig

    payload = dict(_PARITY_GATE)
    payload["screen_confirm_hi"] = 1.0
    with pytest.raises(ValidationError):
        GateConfig.model_validate(payload)


def test_parity_config_mints_random_floor_disabled_and_dev_smoke_enabled() -> None:
    """The production config mints the operator-owed `random_floor_games=20`; dev/smoke mint 4,
    so the headless round exercises a REAL bot and the key has a live EXERCISED consumer."""
    parity_cfg = load_config(_PARITY_CONFIG)
    assert parity_cfg.eval.random_floor_games == 20, (
        "run6 mints the operator-owed random_floor_games=20 (R147/R272(d))"
    )
    for path in _DEV_SMOKE_CONFIGS:
        if not path.exists():
            continue
        cfg = load_config(path)
        assert cfg.eval.random_floor_games == 4, (
            f"{path.name}: dev/smoke templates must mint random_floor_games=4 (A-2)"
        )


# The eval leaf keys the schema must carry, each against its named consumer.
_NEW_LEAF_CONSUMERS = {
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
}


def test_new_keys_all_have_consumers_in_o15_registry() -> None:
    """Every registered eval leaf is present in the live schema.

    The walker is the schema's own `leaf_paths`: a local copy once stopped at `Block | None`
    and walked to a different leaf count than the gate did.
    """
    leaves = set(leaf_paths(RunConfig))
    missing = set(_NEW_LEAF_CONSUMERS) - leaves
    assert missing == set(), (
        f"new schema leaves not yet present (expected once EvalConfig/GateConfig land): "
        f"{sorted(missing)}"
    )


def test_configs_have_no_unminted_manual_edits_signature() -> None:
    """Every config keeps its mint-provenance stamp; a missing one means a hand edit."""
    for path in _config_paths():
        if path.parent == _TEMPLATES_DIR:
            continue  # templates are hand-authored sources, not minted outputs
        text = path.read_text()
        assert "minted-by: tools/mint_config.py" in text, (
            f"{path.name}: missing the mint-provenance header stamp"
        )
