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
})


def split_retired(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(a deep copy without the retired paths, {dotted path: the value it carried})`; the input is untouched."""
    kept = copy.deepcopy(config)
    removed: dict[str, Any] = {}
    for dotted in sorted(RETIRED_PATHS):
        *parents, leaf = dotted.split(".")
        node: Any = kept
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and leaf in node:
            removed[dotted] = node.pop(leaf)
    return kept, removed
