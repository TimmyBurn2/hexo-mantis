"""The wiring config is the armed smoke with its compute shrunk and its draw-rate row disarmed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis.config import census
from mantis.config.loader import load_config

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_WIRING = "smoke_wiring.yaml"
_ARMED_SMOKE = "smoke_preflight_armed.yaml"

#: The compute leaves the wiring config moves off the armed smoke (its draw-rate disarm is nulled out
#: below): trunk, self-play fan-out (4 x leaf batch 8 reaches the collector threshold), batch, eval.
SHRUNK = {
    "run_id": "smoke_wiring",
    "model.gnn.hidden": 8,
    "model.gnn.num_layers": 1,
    "selfplay.n_workers": 4,
    "train.batch_size": 8,
    "eval.max_plies": 24,
    "eval.random_floor_games": 2,
}


def _leaves(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): node}


def test_the_wiring_config_moves_only_the_shrunk_leaves_off_the_armed_smoke() -> None:
    """A wiring row boots the armed smoke's composition; draw-rate alone is disarmed (the tree ships ONE armed non-production config; the 16-step rows cannot reach its step-30 earliest fire and the 50-step row disarms it itself)."""
    smoke_dump = load_config(_CONFIGS / _ARMED_SMOKE).model_dump()
    assert smoke_dump["train"]["draw_rate_abort"] is not None
    smoke_dump["train"]["draw_rate_abort"] = None
    wiring = _leaves(load_config(_CONFIGS / _WIRING).model_dump())
    smoke = _leaves(smoke_dump)
    assert wiring.keys() == smoke.keys()
    moved = {key: wiring[key] for key in wiring if wiring[key] != smoke[key]}
    assert moved == SHRUNK, (
        f"the wiring config must differ from the armed smoke in exactly {sorted(SHRUNK)}; "
        f"got {moved}"
    )


def test_the_wiring_config_is_exempt_from_the_production_census() -> None:
    assert f"configs/{_WIRING}" in census.exempt_config_paths()
