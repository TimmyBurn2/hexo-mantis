"""The wiring config is the armed smoke with its compute shrunk, and nothing else moved."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis.config import census
from mantis.config.loader import load_config

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_WIRING = "smoke_wiring.yaml"
_ARMED_SMOKE = "smoke_preflight_armed.yaml"

#: The ONLY leaves the wiring config moves off the armed smoke: the trunk, the self-play fan-out,
#: the learner batch and the eval games' length and count — compute, never a route or an arming.
SHRUNK = {
    "run_id": "smoke_wiring",
    "model.gnn.hidden": 8,
    "model.gnn.num_layers": 1,
    "selfplay.n_workers": 2,
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
    """A wiring test booting this config drives the armed smoke's composition, arming and route.

    Killer: mint a route, arming or cadence change into the wiring config — a wiring row would
    then prove a composition the armed smoke does not have.
    """
    wiring = _leaves(load_config(_CONFIGS / _WIRING).model_dump())
    smoke = _leaves(load_config(_CONFIGS / _ARMED_SMOKE).model_dump())
    assert wiring.keys() == smoke.keys()
    moved = {key: wiring[key] for key in wiring if wiring[key] != smoke[key]}
    assert moved == SHRUNK, (
        f"the wiring config must differ from the armed smoke in exactly {sorted(SHRUNK)}; "
        f"got {moved}"
    )


def test_the_wiring_config_is_exempt_from_the_production_census() -> None:
    assert f"configs/{_WIRING}" in census.exempt_config_paths()
