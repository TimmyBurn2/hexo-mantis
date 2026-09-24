"""The composition-wiring fakes both config wiring suites drive compose_run with."""
from __future__ import annotations

from types import SimpleNamespace


class RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


def fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )
