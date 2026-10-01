"""The dotted config paths the schema RETIRED after configs and checkpoint stamps carried them — the one authority."""
from __future__ import annotations

import copy
from typing import Any

#: A retired path is tolerated only where a record PREDATES the retirement: a stamp or a pre-retirement config.
RETIRED_PATHS: frozenset[str] = frozenset({
    "search",
    "eval.ladder", "eval.sealbot_model_sims", "eval.rung_concurrency",
    "monitor.wr_hard_abort_enabled", "monitor.wr_rolling_consecutive_evals",
    "monitor.wr_rolling_threshold", "monitor.wr_rolling_min_step",
    "monitor.wr_collapse_from_peak_ratio", "monitor.wr_collapse_min_step",
    "monitor.wr_collapse_consecutive_evals", "monitor.wr_early_death_threshold",
    "monitor.wr_early_death_min_step",
    "train.value_target", "train.draw_reward", "train.ply_cap_value",
    "selfplay.playout_cap.fast_sims", "selfplay.playout_cap.fast_prob", "selfplay.playout_cap.standard_sims",
    "selfplay.mcts.n_simulations", "train.policy_loss_weight_schedule", "train.total_steps",
    "train.replay_capacity_schedule", "train.fast_policy_weight",
    "train.hard_gn_threshold", "train.hard_gn_min_steps",
})


#: A retired path whose value now lives at a live leaf: a record predating the fold left that leaf unset (null or 0).
FOLDED_PATHS: dict[str, str] = {
    "train.total_steps": "train.scheduler_t_max",
    "selfplay.mcts.n_simulations": "selfplay.playout_cap.n_sims_full",
}


def _parent(config: dict[str, Any], dotted: str) -> tuple[Any, str]:
    """`(the mapping holding the dotted leaf, or None, the leaf's key)`."""
    *parents, leaf = dotted.split(".")
    node: Any = config
    for part in parents:
        node = node.get(part) if isinstance(node, dict) else None
    return node, leaf


def split_retired(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(a copy without the retired paths, a fold's unset target filled, {dotted path: value})`; the input is untouched."""
    kept = copy.deepcopy(config)
    removed: dict[str, Any] = {}
    for dotted in sorted(RETIRED_PATHS):
        node, leaf = _parent(kept, dotted)
        if isinstance(node, dict) and leaf in node:
            removed[dotted] = node.pop(leaf)
    for source, target in FOLDED_PATHS.items():
        node, leaf = _parent(kept, target)
        if source in removed and isinstance(node, dict) and node.get(leaf) in (None, 0):
            node[leaf] = removed[source]
    return kept, removed
