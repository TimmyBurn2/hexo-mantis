"""The composition seam's drivable doubles and drive helpers, ONE copy each for every root/wiring test; importable from any test directory through the root conftest's own path."""
from __future__ import annotations

import dataclasses
import shutil
import time
from types import SimpleNamespace
from typing import Any, Callable


class DrivableTrainerStub:
    """The ONE trainer double (R367(a)): the declared entry points plus `device`; `actor_sd` and `inference_sd` are DISTINCT so a root that hands the deploy view to the actors reds."""

    def __init__(self, *, step: int = 0, grad_norm: float = 0.1, on_step: Any = None, model: Any = None) -> None:
        self.step = step
        self.device = "cpu"
        self.model = object() if model is None else model
        self.grad_norm = grad_norm
        self.on_step = on_step
        self.saves: list = []
        self.actor_sd: dict = {"w": "ACTOR-SENTINEL"}
        self.inference_sd: dict = {"w": "DEPLOY-SENTINEL"}

    def loss_info(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": self.grad_norm,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        if self.on_step is not None:
            self.on_step(self.step)
        return self.loss_info()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def actor_state_dict(self) -> dict:
        return self.actor_sd

    def deploy_module(self) -> Any:
        return self.model

    def save_checkpoint(self, loss_info: Any) -> Any:
        self.saves.append(loss_info)
        return None


class RunnerStats:
    """The runner-stats snapshot fields the coordinator's emissions read."""

    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1


def fake_run_safety(**_kwargs: Any) -> SimpleNamespace:
    """An inert `build_run_safety` stand-in: every collaborator it returns is a no-op."""
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


class BufferStub:
    """A shapeless replay buffer for drives that never sample."""

    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...

    def save_to_path(self, p: Any) -> None: ...


class DrivablePoolStub:
    """The ONE WorkerPool double; `stop()` on a never-started pool raises as `Thread.join` does.

    `game_per_read` reports one new game on every `games_completed` read, standing in for a live
    feeder: a compose drive whose pool reports no new games waits forever.
    """

    def __init__(self, *, search_kind: str = "gumbel", games: int = 0,
                 draw_counts: tuple[int, int] = (0, 0), flags: list[int] | None = None,
                 game_per_read: bool = False, on_start: Callable[[], None] | None = None,
                 observer: Callable[[str], None] | None = None, rstats: Any = None) -> None:
        self.n_workers = 1
        self.search_kind = search_kind
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self.started = False
        self.stopped = False
        self.sync_payloads: list = []
        self.step_calls: list[int] = []
        self.window_calls: list[int] = []
        self.rstats = rstats
        self._games = int(games)
        self._game_per_read = game_per_read
        self._draw_counts = (int(draw_counts[0]), int(draw_counts[1]))
        self._flags = [] if flags is None else flags
        self._on_start = on_start
        self._observer = observer

    @property
    def games_completed(self) -> int:
        if self._game_per_read:
            self._games += 1
        return self._games

    @games_completed.setter
    def games_completed(self, value: int) -> None:
        self._games = int(value)

    def start(self) -> None:
        self.started = True
        if self._observer is not None:
            self._observer("pool.start")
        if self._on_start is not None:
            self._on_start()

    def stop(self) -> None:
        """Raises: RuntimeError when the pool was never started."""
        if not self.started:
            raise RuntimeError("cannot join thread before it is started")
        self.stopped = True
        if self._observer is not None:
            self._observer("pool.stop")

    def check_producer_health(self) -> None: ...

    def buffer_composition(self) -> dict[str, Any]:
        return {}

    def pooled_draw_counts(self) -> tuple[int, int]:
        return self._draw_counts

    def ply_cap_window_counts(self, window_games: int) -> tuple[int, int]:
        """Game i's cap flag is `flags[i]`; the window is the last `window_games` completed."""
        self.window_calls.append(window_games)
        tail = self._flags[: self._games][-window_games:]
        return (sum(tail), len(tail))

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return self.rstats if self.rstats is not None else RunnerStats()

    def sync_inference_weights(self, state_dict: Any) -> None:
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


def with_deltas(builder: Callable[..., Any], **deltas: Any) -> Callable[..., Any]:
    """`builder` with `deltas` replaced on its output; every keyword it is called with forwards untouched."""
    def _patched(**kwargs: Any) -> Any:
        return dataclasses.replace(builder(**kwargs), **deltas)
    return _patched


class ExitSpy:
    """An `exit_fn` that records each code instead of exiting."""

    def __init__(self) -> None:
        self.codes: list[int] = []

    def __call__(self, code: int) -> None:
        self.codes.append(int(code))


def inert_heartbeat_registry() -> SimpleNamespace:
    """A heartbeat registry whose one source was beaten just now."""
    return SimpleNamespace(
        sources=("train_step",), ages=lambda: {"train_step": 0.0},
        beaten_sources=lambda: frozenset({"train_step"}), arm=lambda: None,
    )


def fake_disk_usage(free_gb: float) -> Callable[[Any], Any]:
    """A `shutil.disk_usage` stand-in reporting `free_gb` free of four times that total."""
    def _usage(_path: Any) -> Any:
        total = int(free_gb * 1_000_000_000) * 4
        return shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            total=total, used=total - int(free_gb * 1_000_000_000),
            free=int(free_gb * 1_000_000_000),
        )
    return _usage


def await_signal(state: Any) -> None:
    """Wait up to 5 s for a `ShutdownState` to record a stop: CPython delivers a signal at a bytecode boundary."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and state.stop_count < 1:
        time.sleep(0.005)


class FakeClock:
    """A controllable monotonic clock: calling it returns the fake time `t`."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t
