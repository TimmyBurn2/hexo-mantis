"""The pool as `StepCoordinator.iteration_complete` reads it — ONE stub (R367(a)) whose ply-cap window is a SCRIPT over the games completed: `flags[i]` is game i's cap flag, and a pool built without flags reports an empty window."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class CoordinatorPoolStub:
    """One game per step; every reader the coordinator touches returns a fixed, plausible value."""

    def __init__(self, flags: list[int] | None = None, *, search_kind: str = "gumbel") -> None:
        self.games_completed = 0
        self.n_workers = 1
        self.search_kind = search_kind
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self._flags = [] if flags is None else flags
        self.window_calls: list[int] = []

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...

    def buffer_composition(self) -> dict[str, Any]:
        return {}

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def ply_cap_window_counts(self, window_games: int) -> tuple[int, int]:
        self.window_calls.append(window_games)
        tail = self._flags[: self.games_completed][-window_games:]
        return (sum(tail), len(tail))

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return SimpleNamespace(mcts_mean_depth=5.0, mcts_mean_root_concentration=0.1, cluster_value_std_mean=0.0,
                               cluster_policy_disagreement_mean=0.0, cluster_variance_sample_count=0)

    def update_checkpoint_step(self, step: int) -> None: ...
