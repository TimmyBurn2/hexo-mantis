"""The wiring config is the armed smoke on the production regime, its compute shrunk and its draw-rate row disarmed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mantis.config import census
from mantis.config.loader import load_config

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"
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

#: The regime leaves it moves onto the production configs' (Gumbel, its target, the playout cap, the aux head), at smoke budgets.
REGIME = {
    "selfplay.search.kind": "gumbel",
    "train.policy_target": "completed_improved_policy",
    "selfplay.q_rescale": False,
    "selfplay.mcts.dirichlet_enabled": False,
    "selfplay.gumbel_m": 4,
    "selfplay.gumbel_m_quick": 4,
    "selfplay.playout_cap.full_search_prob": 0.25,
    "selfplay.playout_cap.n_sims_quick": 2,
    "selfplay.playout_cap.n_sims_full": 4,
    "identity.arch_kind": "GnnArchV2SoftPolicy",
    "model.aux_soft_policy.target_temperature": 4.0,
    "model.aux_soft_policy.weight": 4.0,
}


def _leaves(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): node}


def test_the_wiring_config_moves_only_the_shrunk_and_regime_leaves_off_the_armed_smoke() -> None:
    """A wiring row boots the armed smoke's composition; draw-rate alone is disarmed (the tree ships ONE armed non-production config; the 16-step rows cannot reach its step-30 earliest fire, and the 200-step clean-stop row sits inside it, where an untrained net's ply-cap draws would abort a healthy run on a slow host)."""
    smoke_dump = load_config(_CONFIGS / _ARMED_SMOKE).model_dump()
    assert smoke_dump["train"]["draw_rate_abort"] is not None
    smoke_dump["train"]["draw_rate_abort"] = None
    wiring = _leaves(load_config(_CONFIGS / _WIRING).model_dump())
    smoke = _leaves(smoke_dump)
    # The aux head's null leaf becomes its two rows; every other leaf is shared.
    assert wiring.keys() == (smoke.keys() - {"model.aux_soft_policy"}) | {
        key for key in REGIME if key.startswith("model.aux_soft_policy.")}
    moved = {key: wiring[key] for key in wiring if wiring[key] != smoke.get(key)}
    assert moved == SHRUNK | REGIME, (
        f"the wiring config must differ from the armed smoke in exactly {sorted(SHRUNK | REGIME)}; "
        f"got {moved}"
    )


def test_the_wiring_config_runs_every_production_configs_regime() -> None:
    """PLANTED BREAK: re-mint the wiring config on PUCT, or with the aux head null, and this reds against the census."""
    wiring = load_config(_CONFIGS / _WIRING)
    productions = census.production_configs(_REPO)
    assert productions, "the census is empty, so the regime has nothing to match"
    for path in productions:
        prod = load_config(path)
        regime = {
            "search kind": (wiring.selfplay.search.kind, prod.selfplay.search.kind),
            "policy target": (wiring.train.policy_target, prod.train.policy_target),
            "arch kind": (wiring.identity.arch_kind, prod.identity.arch_kind),
            "aux head armed": (wiring.model.aux_soft_policy is not None, prod.model.aux_soft_policy is not None),
            "playout cap armed": (wiring.selfplay.playout_cap.full_search_prob > 0,
                                  prod.selfplay.playout_cap.full_search_prob > 0),
            "q rescale": (wiring.selfplay.q_rescale, prod.selfplay.q_rescale),
            "dirichlet": (wiring.selfplay.mcts.dirichlet_enabled, prod.selfplay.mcts.dirichlet_enabled),
        }
        off = {name: pair for name, pair in regime.items() if pair[0] != pair[1]}
        assert not off, f"the wiring config leaves {path}'s regime as (wiring, production): {off}"


def test_the_wiring_config_is_exempt_from_the_production_census() -> None:
    assert f"configs/{_WIRING}" in census.exempt_config_paths()
