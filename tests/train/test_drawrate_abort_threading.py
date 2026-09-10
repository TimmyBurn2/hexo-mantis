"""The resolved draw-rate value REACHES the runtime, and every key of the block reaches its
own call site.

Each oracle is the only witness to one defect: a construction site that does not thread the
RESOLVED value, so the audit reads the config and goes green over a disarmed run (a source pin
and defaultless fields can both hold while the wrong value flows); the gate left as a numeric
comparison, where `None > 0` raises `TypeError` once per `step()` on every disarmed run; and
each key observed at its own call site, since the registry bijection proves only that someone
wrote down a consumer.

>300 justify (R8): one seam, one set of drivable fakes shared by the composed drive and the
directly-built coordinator. `_Pool` is the one fake this delta's Protocol widening changes, so
splitting the file would fork it into copies free to drift in exactly that direction.
"""
from __future__ import annotations

from mantis._engine import HexgBuffer

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run
import mantis.train.coordinator.step as step_module

# `ruff --fix` re-sorts the `resolve.draw_rate` import into the third-party block while the
# module it names does not exist; it belongs here with its `mantis.*` siblings.
from mantis.config.armed_aborts import audit_arming
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import (  # RED anchor (R80) — the ONE read path
    DrawRateAbortSpec,
    resolve_draw_rate_abort,
)
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config  # RED anchor — the renamed builder (R73)
from mantis.train.coordinator.config import StepCoordinatorConfig, pooled_draw_rate
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through; cross-test imports are barred,
    so each file that needs one builds it."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



#: The declaration a `StepCoordinator` reads on the graph route: the identity it dispatches on
#: plus the sections the route's own resolvers read. The caps are the template's NON-BINDING
#: pair — nothing here exercises a split.
_GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}


_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_DRIVE_STEPS = 4
#: Deliberately NOT run5's own numbers: a harness driving the production values cannot
#: distinguish "the config reached the coordinator" from "the builder hardcodes the same
#: numbers the config happens to carry".
_OFF_PREREG = {"threshold": 0.37, "min_step": 2, "N_pool_min": 7, "consec": 2}


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    """The pool surface the composer and `StepCoordinator` touch, plus a recorder on the ONE
    method whose shape this delta changes: `pooled_draw_counts()` takes NO bar, the evidence bar
    having moved to the abort decision, so the recorder counts CALLS."""

    def __init__(self, *, counts: tuple[int, int] = (0, 0),
                 fresh_game_per_read: bool = True) -> None:
        self._games = 0
        self._fresh = fresh_game_per_read
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate, self.o_winrate, self.draws = 0.5, 0.45, 1
        self.draw_rate = 0.05  # the third outcome share.
        self.sims_per_sec, self.batch_fill_pct = 100.0, 0.9
        self.recent_move_histories: list = []
        self.started = self.stopped = False
        self.n_workers = 1
        self._counts = (int(counts[0]), int(counts[1]))
        #: how many times the producer was CALLED — the disarmed arm's observation.
        self.counts_calls = 0

    @property
    def games_completed(self) -> int:
        if self._fresh:
            self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        self.counts_calls += 1
        return self._counts

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        return None

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self, step: int = 0) -> None:
        self.step = step
        self.model = object()
        self.device = "cpu"
        self.inference_sd = {"w": "SENTINEL"}

    # The double conforms to the DECLARED seam (typed entry points + `device`).
    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def save_checkpoint(self, loss_info) -> None:
        return None


class _Buffer:

    def __init__(self) -> None:
        self._hexg = _filled_hexg()

    size, capacity = 1000, 100_000

    def resize(self, n: int) -> None:
        return None

    def save_to_path(self, p) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # The graph route's sampler, DELEGATED to a real `HexgBuffer`: the dispatcher collates
        # the wire for real, so a hand-built payload would be a second wire format.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _SpySink:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def named(self, name: str) -> list:
        return [e for e in self.events if isinstance(e, dict) and e.get("event") == name]


def _fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


def _bounded(factory, *, block, name: str = "smoke_preflight_armed.yaml", steps: int = _DRIVE_STEPS):
    """A REAL minted config, bounded so a composed drive terminates, carrying `block`. The three
    step-clock knobs move together because the reachability validator spans them."""
    return factory(name,
                   train={"actor_sync_cadence_steps": 1, "max_train_steps": steps,
                          "draw_rate_abort": block,
                          # the compose drive runs the real graph route.
                          "batch_size": 8},
                   monitor={"actor_lag_threshold_steps": steps - 1,
                            # `smoke_preflight_armed.yaml` ships actor-lag DISARMED, so with
                            # draw-rate required the audit would report BOTH rows disarmed and
                            # the exact-list assertion below could never hold.
                            "actor_lag_abort_enabled": True},
                   # `eval_enabled` is a CONFIG fact, declared here; posture byte-preserved.
                   eval_enabled=False)


def _coordinator(*, config, pool, trainer=None):
    """A `StepCoordinator` on fakes, built directly: these oracles are about what `step()` does
    with a config, not about how one is composed."""
    shutdown, sink = ShutdownState(), _SpySink()
    coord = StepCoordinator(
        trainer=trainer or _Trainer(), buffer=_Buffer(), pretrained_buffer=None,
        recent_buffer=None, pool=pool, eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        # Unit drives declare the grid identity their _Buffer fake serves.
        full_config=_GRAPH_FULL_CONFIG, train_cfg={}, mixing_cfg={}, sink=sink,
        heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, sink=sink)


def _coordinator_config(spec, **overrides) -> StepCoordinatorConfig:
    """The production builder's own output with `draw_rate_abort` set — never a hand-written
    census of the knobs it still owns. `dataclasses.replace` supplies the harness-only cadence
    knobs so the builder stays the single source of the rest, and `drain_caps` arrives from a
    MINTED block for `stop_step`'s reason: a literal would be a second authority."""
    base = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=spec,
        drain_caps=resolve_drain_caps(load_config(_CONFIGS / "dev_example.yaml").monitor),
        gate_interval=load_config(_CONFIGS / "dev_example.yaml").monitor.gate_interval,
        knobs=resolve_coordinator_knobs(load_config(_CONFIGS / "dev_example.yaml").train))
    # The gate cadence mirrors the narration cadence, the shipped posture.
    settings = {"log_interval": 1, "eval_interval": 1, "min_buf_size": 1,
                "terminal_eval_enabled": False, **overrides}
    settings.setdefault("gate_interval", settings["log_interval"])
    return dataclasses.replace(base, **settings)


# The audited value IS the value the coordinator runs on.
def test_the_audited_value_IS_the_value_the_coordinator_runs_on(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """The production builder's own output with `draw_rate_abort` set, never a hand-written census
    of the knobs it owns; `dataclasses.replace` supplies the harness-only cadence knobs, and
    `drain_caps` arrives from a MINTED block because a literal would be a second authority."""
    cfg = _bounded(smoke_run_config, block=dict(_OFF_PREREG))
    block = cfg.train.draw_rate_abort
    resolved = resolve_draw_rate_abort(cfg.train)
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    runtime = handles.coordinator.config.draw_rate_abort
    assert runtime is not None, (
        "an ARMED config reached the coordinator as `None` — the composition root is not "
        "threading the resolved value, which is exactly the state where the audit reads "
        "0.37 from the config and the run aborts on nothing"
    )
    for key in ("threshold", "min_step", "N_pool_min", "consec"):
        assert getattr(runtime, key) == getattr(resolved, key) == getattr(block, key), (
            f"the three readers disagree on {key!r}: schema says {getattr(block, key)!r}, "
            f"the resolver says {getattr(resolved, key)!r}, the coordinator runs on "
            f"{getattr(runtime, key)!r}. One fact, one authority (R79)"
        )
    assert runtime.threshold != 0.25, (
        "harness precondition: the driven threshold must differ from run5's pre-registered "
        "0.25, or a hardcoded builder passes this test"
    )

    disarmed_cfg = _bounded(smoke_run_config, block=None)
    handles = mantis.run.compose_run(
        config=disarmed_cfg, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "off"), checkpoint_dir=str(tmp_path / "off_ckpt"),
    )
    assert handles.coordinator.config.draw_rate_abort is None, (
        "`train.draw_rate_abort: null` is EXPLICITLY OFF and must arrive as `None`. A "
        "code-side default resurrected anywhere on this path turns a deliberate disarm into "
        "an inherited posture, which is R79(1)'s whole subject"
    )
    assert [row.name for row in audit_arming(disarmed_cfg).disarmed] == ["draw_rate_collapse"], (
        "…and the same config must read DISARMED to the manifest. If the runtime says None "
        "while the audit says armed, the pin and the manifest are bound to different facts"
    )


# The disarmed posture skip-counts and never raises.
def test_a_disarmed_threshold_skip_counts_and_never_raises_TypeError() -> None:
    """Three readers agree on one fact: the schema block, the resolver, and
    `StepCoordinator.config` after a real composed drive. The driven values are deliberately NOT
    run5's, since a builder hardcoding those would satisfy an oracle driven on them while reading
    nothing; the second arm requires the disarmed posture to survive as `None` and audit
    disarmed."""
    pool = _Pool(counts=(99, 100))
    h = _coordinator(config=_coordinator_config(None), pool=pool)
    h.pool._games = 5
    h.coord.step()

    stats = h.coord._gate_stats["draw_rate_collapse"]
    assert stats["checks"] >= 1, (
        "harness precondition: the gate must have been REACHED. A step() that never reaches "
        "`_run_hard_abort_gates` cannot witness the TypeError this test exists for"
    )
    assert stats["skips"] >= 1, (
        "a disarmed gate must SKIP-COUNT (LAW-18): a lever under test logs its own fire-rate "
        f"in-run, and 'explicitly off' is a posture the operator must be able to read; {stats}"
    )
    assert stats["fires"] == 0 and h.shutdown.running is True, (
        "`train.draw_rate_abort: null` means the gate cannot fire, however bad the draw rate"
    )

    h2 = _coordinator(
        config=_coordinator_config(
            DrawRateAbortSpec(threshold=0.4, min_step=0, N_pool_min=1, consec=3)),
        pool=_Pool(counts=(90, 100)))
    for _ in range(8):
        if not h2.shutdown.running:
            break
        h2.pool._games += 5
        h2.coord.step()
    assert h2.shutdown.running is False, (
        "an ARMED spec must still fire on a sustained collapse — a gate that only ever skips "
        "is as useless as one that only ever fires"
    )

    h3 = _coordinator(
        config=_coordinator_config(
            DrawRateAbortSpec(threshold=0.4, min_step=10**9, N_pool_min=1, consec=3)),
        pool=_Pool(counts=(90, 100)))
    for _ in range(8):
        h3.pool._games += 5
        h3.coord.step()
    assert h3.shutdown.running is True, (
        "`min_step` is a REAL guard: past-the-floor is the only regime the abort may fire in "
        "(`rules.py:261`). Its True arm becomes reachable in production for the first time "
        "with this delta, because `min_step` is non-zero for the first time"
    )


# All the block's keys reach their own call site.
def test_all_THREE_block_keys_reach_their_runtime_destination(monkeypatch) -> None:
    """`None > 0` inside the gate raises `TypeError` once per `step()` on every disarmed run, so
    `is not None` is required by the type change and this is its only witness. The skip
    accounting is asserted with a NON-EMPTY producer, because a coordinator that never reached
    the gate would satisfy "no TypeError" while witnessing nothing."""
    cfg = load_config(_CONFIGS / "run6.yaml")
    spec = resolve_draw_rate_abort(cfg.train)
    seen: list[dict] = []

    def _spy(history, current_step, **kwargs):
        seen.append(dict(kwargs))
        return None

    monkeypatch.setattr(step_module, "check_draw_rate_collapse", _spy)
    pool = _Pool(counts=(90, 100))
    h = _coordinator(config=_coordinator_config(spec), pool=pool)
    h.pool._games = 5
    h.coord.step()

    assert seen, (
        "`check_draw_rate_collapse` was never called: with an armed spec and a live producer "
        "the gate must RUN, or none of the three destinations below is observable"
    )
    call = seen[-1]
    assert call["threshold"] == spec.threshold == cfg.train.draw_rate_abort.threshold, (
        f"train.draw_rate_abort.threshold must reach `check_draw_rate_collapse(threshold=)`; "
        f"config says {cfg.train.draw_rate_abort.threshold!r}, the rule saw "
        f"{call.get('threshold')!r}"
    )
    assert call["min_step"] == spec.min_step == cfg.train.draw_rate_abort.min_step, (
        f"train.draw_rate_abort.min_step must reach `check_draw_rate_collapse(min_step=)`; "
        f"config says {cfg.train.draw_rate_abort.min_step!r}, the rule saw "
        f"{call.get('min_step')!r}"
    )
    # A DELIBERATE BOUNDARY MARKER, read from the other side: `consec` is the block's own
    # authored term now, so the marker's job — do not leave the reader guessing which of these
    # the config authors — is served by asserting it at the same call site as the other two.
    assert call["consec"] == spec.consec == cfg.train.draw_rate_abort.consec, (
        f"train.draw_rate_abort.consec must reach `check_draw_rate_collapse(consec=)`; "
        f"config says {cfg.train.draw_rate_abort.consec!r}, the rule saw "
        f"{call.get('consec')!r}. It is a CONFIG term since WPMINT Phase K-B, not the "
        "coordinator's own default — that boundary is what this arm has always marked"
    )
    # `N_pool_min` at ITS destination, the observation boundary. Both sides are driven, because
    # "no observation ever" and "observation always" each satisfy one side alone.
    bar = spec.N_pool_min
    below = _coordinator(config=_coordinator_config(spec), pool=_Pool(counts=(bar - 1, bar - 1)))
    below.pool._games = 5
    below.coord.step()
    assert below.coord._draw_rate_history == [], (
        f"with {bar - 1} completed games — ONE under train.draw_rate_abort.N_pool_min "
        f"({cfg.train.draw_rate_abort.N_pool_min!r}) — the gate must make NO OBSERVATION. A "
        "1.0 appended here is a total-collapse reading taken on evidence the operator "
        "declared insufficient; a 0.0 is DR-4's fabricated healthy reading"
    )
    assert below.coord._gate_stats["draw_rate_collapse"]["skips"] >= 1, (
        "…and it must SKIP-COUNT (LAW-18): insufficient evidence is a posture an operator "
        f"must be able to read in-run; {below.coord._gate_stats['draw_rate_collapse']}"
    )
    assert below.pool.counts_calls >= 1, (
        "harness precondition: the producer must have been CALLED — a gate that never reached "
        "it would satisfy the two arms above while witnessing nothing"
    )

    at_bar = _coordinator(config=_coordinator_config(spec), pool=_Pool(counts=(bar, bar)))
    at_bar.pool._games = 5
    at_bar.coord.step()
    assert at_bar.coord._draw_rate_history == [1.0], (
        f"at exactly {bar} completed games the bar is MET and the observation is the true "
        f"pooled rate (1.0 here). If the boundary sat anywhere but at the config's own "
        f"N_pool_min, one of these two drives would disagree"
    )

    disarmed = _coordinator(config=_coordinator_config(None), pool=_Pool(counts=(90, 100)))
    disarmed.pool._games = 5
    disarmed.coord.step()
    assert disarmed.pool.counts_calls == 0, (
        "on the disarmed posture the producer must never be CALLED: there is no `N_pool_min` "
        "to judge its answer against, and sampling it anyway would put a reading in the abort "
        "history on the axis the config just declined to arm (the disarmed path — `_sample` "
        "takes its producer-absent arm and SKIP-counts)"
    )


def test_the_pooled_rate_is_the_thing_the_threshold_is_compared_against() -> None:
    """Each key is observed AT ITS OWN DESTINATION, with expected values read off the config
    through the resolver rather than written as literals. The third key's arm drives the
    BOUNDARY: at `N_pool_min - 1` completed games there is NO observation and at exactly
    `N_pool_min` there is one, so a delta threading a different number moves it and one
    threading none cannot produce it."""
    assert pooled_draw_rate((0, 0), N_pool_min=1) is None, (
        "no completed games is NO OBSERVATION, not a healthy 0.0 (DR-4). The old function "
        "returned 0.0 here and the gate recorded it as a reading"
    )
    assert pooled_draw_rate((5, 100), N_pool_min=50) == 0.05
    assert pooled_draw_rate((100, 100), N_pool_min=50) == 1.0, (
        "1.0 is the metric's maximum at every worker count — no legal input can exceed it, "
        "so any threshold > 1.0 is 'armed in the config, absent in effect'"
    )
    assert pooled_draw_rate((0, 100), N_pool_min=50) == 0.0, (
        "…and a MEASURED zero over sufficient evidence is a real healthy reading that must "
        "still be a float. R92 removed the fabricated zero, not the measured one"
    )


def test_the_harness_pool_refuses_the_call_shape_this_delta_retires() -> None:
    """The harness's own vacuity guard: if the fake still carried the RETIRED
    `per_worker_draw_rates`, the coordinator would find it by `getattr`, take the live path,
    and the failure would surface somewhere other than in the disarmed arm."""
    pool = _Pool(counts=(1, 4))
    assert pool.pooled_draw_counts() == (1, 4)
    assert pool.counts_calls == 1
    assert not hasattr(pool, "per_worker_draw_rates"), (
        "the retired producer must be GONE from the fake, not merely unused: `step.py` reaches "
        "it by `getattr(self.pool, ...)`, so a surviving attribute is a live call site"
    )
    with pytest.raises(TypeError):
        pool.pooled_draw_counts(min_samples=9)  # type: ignore[call-arg]
