# >300 justify (R8). The seven rows are ONE claim — a BROKEN TERMINAL eval round is
# supervisor-distinguishable from a clean run and a MID-RUN one deliberately is not — over one
# seam running from `drain.run_terminal_eval` through a set-once latch to `mantis.run.main`'s
# rc. Measuring it needs the whole ladder in one place: drivable collaborators, the `main()`
# driver, a rigged volume and a real `EvalPipeline`. R5 bars cross-test imports, so a split
# forks that harness into copies that drift while both stay green.
"""The terminal round's reason reaches the process exit code, and a mid-run one still does not.

At HEAD `drain.close_out` computed the terminal round's result, routed it, then threw the
return value away one frame below `ShutdownState` — the only object that can carry an outcome
to `main` — so a run whose terminal battery was killed exited 0 and the supervisor recorded a
clean finish.

Per-row mutations: M-O5 call the recorder from `flush_pending_eval`; M-O6 latch a constant;
M-O7 a second recorder call, M-O7b `self.run_terminal_eval()` inside `step()`; M-O8 delete the
root's read; M-O9 move the terminal read before the disk-guard read; M-O10 replace the
re-parse with `if raw:`; M-O32 drop the `_terminal` round-id suffix.

REAL in the `main()` drives: everything from `mantis.run.main` down to `exit_code_for_abort`.
FAKE: the three injected collaborators, `shutil.disk_usage`, and the eval pipeline, whose
`run_evaluation(ignore_stride=True)` returns a rigged result — the subject boundary, since
which reason a real round PRODUCES is `tests/eval/test_eval_broken_reason_routes.py`'s subject.
O-32 uses the REAL `EvalPipeline` because its subject is the round id that pipeline mints.
"""
from __future__ import annotations

import ast
import dataclasses
import multiprocessing
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import mantis.run as mantis_run
from mantis._engine import HexgBuffer
from mantis.config.armed_aborts import (
    DISK_SPACE_ABORT_RULE,
    MANIFEST,
    exit_code_for_abort,
)
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung, RunConfig
from mantis.encoding import lookup
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.model import GnnArch, build_net
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import DRAW_RATE_COLLAPSE_EXIT_CODE
from mantis.run import RunCollaborators, _step_coordinator_config
from mantis.train.coordinator import drain
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.signals import ShutdownState

#: The declaration a `StepCoordinator` reads on the graph route: the identity it dispatches on
#: plus the sections the route's resolvers read. The caps are the template's NON-BINDING pair.
_GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}


def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through (R5 bars cross-test imports,
    so each file that needs one builds it)."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"
_CONFIGS = _REPO / "configs"

#: The seven censused reason spellings, transcribed rather than derived from the enum under
#: test: an oracle reading its expectation off the subject accepts any consistent renaming.
_SEVEN_REASONS = (
    "join_timeout", "killed", "exit_nonzero", "result_missing", "result_invalid",
    "ladder_persist_failed", "round_completion_error",
)

#: The rule name the composition root records for a broken terminal round, and the code the
#: manifest authors for it — the INDEPENDENT statement of what the row must be called.
_TERMINAL_RULE = "terminal_eval_broken"

#: The bounded burst. 3 is the smallest legal run at cadence 1 (the reachability validator
#: spans cadence < actor_lag_threshold < max_train_steps).
_DRIVE_STEPS = 3
#: Rigged free space, in decimal GB (`disk_guard.py`'s `/1e9` divisor).
_HEALTHY_GB = 500.0
_CRITICAL_GB = 1.0
#: A guard cadence short enough to fire inside a sub-second burst; which side of the
#: thresholds a drive lands on is decided by the rigged `shutil.disk_usage`.
_DRIVE_GUARD = {"interval_sec": 0.02, "warn_gb": 4.0, "fail_gb": 2.0}

_DEV_CONFIG = load_config(_CONFIGS / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)
#: The builder's fifth config-authored parameter, `monitor.gate_interval` — the ARMING
#: cadence. Harnesses that set `log_interval` mirror it onto `gate_interval`, the shipped
#: posture: every committed config mints the two equal.
_GATE_INTERVAL = _DEV_CONFIG.monitor.gate_interval


def _mirrored(settings: dict) -> dict:
    """The GATE cadence mirrors the NARRATION cadence unless a drive names it — the shipped
    posture, so a drive that moves only `log_interval` keeps its pre-split cadence."""
    settings.setdefault("gate_interval", settings["log_interval"])
    return settings


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    search_kind = "gumbel"
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # F-816-2: the third outcome share.
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    def __init__(self, *, draw_counts: tuple[int, int] = (0, 0)) -> None:
        self.started = False
        self._games = 0
        self.recent_move_histories: list = []
        self.sync_calls: list = []
        self.draw_counts = draw_counts

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        if not self.started:
            raise RuntimeError("cannot join thread before it is started")

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return self.draw_counts

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict: Any) -> None:
        self.sync_calls.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    """A trainer stand-in carrying a REAL declared arch + net: `resolve_anchor` builds the
    anchor from `trainer.arch`, so a bare object cannot stand in once a pipeline is composed."""

    def __init__(self, on_step: Any = None) -> None:
        self.step = 0
        self.device = "cpu"
        self.saves: list = []
        self._on_step = on_step
        self.arch = GnnArch(in_dim=8, edge_dim=4, hidden=8, num_layers=1,
                            policy_hidden=8, value_hidden=8)
        self.model = build_net(self.arch)

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        if self._on_step is not None:
            self._on_step(self.step)
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return self.model.state_dict()

    def save_checkpoint(self, loss_info: Any) -> None:
        self.saves.append(loss_info)


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, path: Any) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # The graph route's sampler, DELEGATED to a real `HexgBuffer`: the dispatcher collates
        # the wire for real, so a hand-built payload would be a second wire format.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        # `event` is subscripted, not `.get`-ed: a payload without it is a producer defect and
        # must be loud rather than silently filtered out of every assertion below.
        return [e for e in self.events if e["event"] == name]

    def order(self) -> list[str]:
        return [e["event"] for e in self.events]


def _broken_round(reason: str, *, round_id: str = "r000001_3_terminal", step: int = 3) -> dict:
    """A BROKEN round-result mapping in the post-R152 shape, hand-built: which reason a real
    round produces is `tests/eval/test_eval_broken_reason_routes.py`'s subject."""
    return {"step": step, "round_id": round_id, "promoted": False, "promoted_step": None,
            "wr_sealbot": None, "wr_random": None, "eval_round_wall_sec": 0.5,
            "eval_broken_reason": reason, "eval_broken_detail": None,
            "gate": None, "rungs": {}, "skipped_rungs": [], "bt": {"ratings": {}, "p_hat": {}},
            "schedule_next": {}}


def _clean_round(*, round_id: str = "r000001_3_terminal", step: int = 3) -> dict:
    result = _broken_round("unused", round_id=round_id, step=step)
    result["eval_broken_reason"] = None
    return result


class _FakeEvalPipeline:
    """An eval pipeline whose terminal and mid-run answers are supplied by the drive."""

    def __init__(self, *, terminal_result: Any = None, poll_result: Any = None,
                 drain_result: Any = None) -> None:
        self._terminal_result = terminal_result
        self._poll_result = poll_result
        self._drain_result = drain_result
        self.terminal_calls: list[bool] = []
        self.polled = 0

    def run_evaluation(self, model: Any, step: int, best: Any, *, full_config: Any,
                       best_model_step: Any, ignore_stride: bool = False) -> Any:
        self.terminal_calls.append(ignore_stride)
        if ignore_stride:
            return self._terminal_result
        return {"kicked": False, "round_id": "r-inflight", "step": step, "reason": "busy"}

    def poll_completed(self) -> Any:
        self.polled += 1
        result, self._poll_result = self._poll_result, None
        return result

    def drain_pending(self) -> Any:
        result, self._drain_result = self._drain_result, None
        return result

    def apply_gate_decision(self, result: Any) -> int | None:
        return None

    def stop(self) -> None:
        return None


def _make_coordinator(*, eval_pipeline: Any, sink: _SpySink,
                      config_overrides: dict | None = None) -> SimpleNamespace:
    """A REAL `StepCoordinator`, so the latch's ABSENCE is an AttributeError here rather than a
    `SimpleNamespace` silently answering `None`."""
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        **_mirrored({"eval_interval": 10**9, "log_interval": 1, "min_buf_size": 10,
                     **(config_overrides or {})}),
    )
    pool = _Pool()
    shutdown = ShutdownState()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=eval_pipeline, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=_tiny_model(), bufs=None, config=config,
        full_config=_GRAPH_FULL_CONFIG,
        train_cfg={}, mixing_cfg={}, sink=sink, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, sink=sink)


def _fake_disk_usage(free_gb: float):
    def _usage(_path):
        total = int(free_gb * 1_000_000_000) * 4
        return shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            total=total, used=total - int(free_gb * 1_000_000_000),
            free=int(free_gb * 1_000_000_000),
        )
    return _usage


class _Drive:
    """What one `main()` drive observed: its rc, and the live objects it composed."""

    def __init__(self) -> None:
        self.rc: int | None = None
        self.handles: Any = None
        self.guards: list[DiskGuard] = []
        self.pipeline: _FakeEvalPipeline | None = None


def _write_config(tmp_path: Path, **train_overrides: Any) -> Path:
    """A REAL minted config written to disk so `main --config` reads it back through the ONE
    loader. `smoke_preflight_armed.yaml` already mints the two keys the rc needs."""
    base = load_config(_CONFIGS / "smoke_preflight_armed.yaml").model_dump()
    train = dict(base["train"])
    train.update({"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
                  "batch_size": 8, "log_interval": 1})
    # The armed draw-rate floor is rescaled with the run: this drive is 3 steps long and the
    # base config's floor of 10 is an abort that can never fire.
    if train["draw_rate_abort"] is not None:
        train["draw_rate_abort"] = {**train["draw_rate_abort"], "min_step": 1}
    train.update(train_overrides)
    base["train"] = train
    monitor = dict(base["monitor"])
    # The ARMING cadence is `monitor.gate_interval`, not `train.log_interval`, and this drive
    # needs an observation on every step of a 3-step burst; at the config's own interval the
    # gate would never run and O-09 arm (b) would measure a different stop reason.
    monitor.update({"actor_lag_threshold_steps": _DRIVE_STEPS - 1,
                    "gate_interval": 1,
                    "disk_guard": dict(_DRIVE_GUARD)})
    base["monitor"] = monitor
    config = RunConfig.model_validate(base)
    path = tmp_path / "drive.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8")
    return path


def _drive_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
                *, pipeline: _FakeEvalPipeline, free_gb: float = _HEALTHY_GB,
                wait_for_fire: bool = False, draw_counts: tuple[int, int] = (0, 0),
                **train_overrides: Any) -> _Drive:
    """Run `mantis.run.main` end to end with the eval pipeline substituted at its construction
    site. `wait_for_fire` blocks the fake train step until the disk guard's latch is set, so
    the first-fire-wins arm measures the ORDER and never a race, and `request.addfinalizer`
    closes the REAL sink, which `close_out` never touches."""
    drive = _Drive()
    drive.pipeline = pipeline
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(free_gb))

    class _RecordedGuard(DiskGuard):
        """The REAL guard; every behaviour is `super()`'s. Recorded so the drive finds it."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            drive.guards.append(self)

    def _await_fire(_step: int) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if drive.guards and drive.guards[-1].critical_fired:
                return
            time.sleep(0.005)
        raise AssertionError(
            "the rigged filesystem never drove the guard's critical arm inside 10 s — the "
            "drive's premise is broken, so nothing below would be measuring the seam"
        )

    out_dir = tmp_path / "out"
    buffer = HexgBuffer(64, "gnn_axis_v1", 128)
    for i in range(8):
        buffer.push_graph_position([(0, 0, 1), (1, 0, -1)], [(2, 0, 0.6), (1, 1, 0.4)],
                                   1, 30, 2 + i, True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    collaborators = RunCollaborators(
        trainer=_Trainer(on_step=_await_fire if wait_for_fire else None),
        pool=_Pool(draw_counts=draw_counts), buffer=buffer,
        log_dir=out_dir / "logs", checkpoint_dir=out_dir / "checkpoints",
    )
    monkeypatch.setattr(mantis_run, "build_run_collaborators", lambda **_kw: collaborators)
    monkeypatch.setattr(mantis_run, "build_eval_pipeline", lambda **_kw: pipeline)
    monkeypatch.setattr(mantis_run, "DiskGuard", _RecordedGuard)

    # Registered at the SINK'S OWN construction, not after `compose_run` returns, because
    # O-10's drive raises from INSIDE `compose_run` and would skip a post-return registration.
    real_build_run_safety = mantis_run.build_run_safety

    def _recording_build_run_safety(**kwargs: Any):
        run_safety = real_build_run_safety(**kwargs)      # the REAL builder, unmodified
        request.addfinalizer(run_safety.sink.close)
        return run_safety

    monkeypatch.setattr(mantis_run, "build_run_safety", _recording_build_run_safety)

    real_compose = mantis_run.compose_run

    def _recording_compose(**kwargs: Any):
        drive.handles = real_compose(**kwargs)      # the REAL composer, unmodified
        return drive.handles

    monkeypatch.setattr(mantis_run, "compose_run", _recording_compose)
    config_path = _write_config(tmp_path, **train_overrides)
    drive.rc = mantis_run.main(["--config", str(config_path), "--out-dir", str(out_dir)])
    return drive


def _await_signal(state: ShutdownState) -> None:
    """CPython delivers a signal at a bytecode boundary, so the handler may still be pending
    when `compose_run` returns. Bounded wait: a race must fail loudly, never leave a SIGTERM
    pending into the next test."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and state.stop_count < 1:
        time.sleep(0.005)


def _tiny_model():
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    return net


class _FakeProcess:
    def __init__(self, *, target=None, args=(), kwargs=None, daemon=None) -> None:
        self.pid = 4242
        self.alive = False
        self.exitcode: int | None = None

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -15

    def kill(self) -> None:
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -9


class _FakeCtx:
    def __init__(self) -> None:
        self.last_process: _FakeProcess | None = None

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> _FakeProcess:
        proc = _FakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self.last_process = proc
        return proc


def _real_pipeline(tmp_path: Path, sink: _SpySink):
    rungs = [LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5,
                        opponent_sims=None, opening_book="book_v1_s20260625_p4",
                        deploy_matched=True, games_max=32)]
    gate = GateConfig(stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=150,
                      opening_book="book_v1_s20260625_p4", bootstrap_resamples=1000,
                      min_distinct_per_pair=10, seed_base=20260625)
    ladder = LadderConfig(rungs=rungs, round_games=64, min_games_per_active_rung=4,
                          graduation_wr_lower_ci=0.75, graduation_consec_rounds=3,
                          activation_wr_lower_ci=0.65, calibration_every_k_rounds=4,
                          calibration_games=8, bootstrap_resamples=1000,
                          bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234)
    eval_cfg = EvalConfig(random_model_sims=96, sealbot_model_sims=128, random_floor_games=4, worker_device="cpu",
                          round_timeout_sec=5.0, worker_kill_grace_sec=0.2,
                          ply_cap_adjudication=None, strength_floor=None, gate=gate,
                          ladder=ladder)
    spool = tmp_path / "spool"
    spool.mkdir(parents=True, exist_ok=True)
    pipeline = build_eval_pipeline(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128,
        eval_cfg=eval_cfg,
        coordinator_cfg_caps=DrainCaps(final_eval_drain_timeout_sec=0.05,
                                       eval_final_drain_safety_factor=1.0,
                                       eval_final_drain_hard_cap_sec=0.05,
                                       terminal_eval_hard_cap_sec=0.05),
        encoding="v6_live2_ls", run_id="oracle_test_run", spool_dir=spool, game_record_dir=str(spool) + "_games",
        ladder_state_path=tmp_path / "ladder_state.json",
        promotion=DeployTagHooks(
            anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
            best_model_path=tmp_path / "best_model.pt", run_id="oracle_test_run",
            encoding="v6_live2_ls", save_anchor=lambda *a, **k: None,
            guarded_load=lambda *a, **k: None),
        sink=sink,
        # The GRID arm (`v6_live2_ls` has no fused graph forward to bound), stated because the
        # parameter carries no default.
        fused_graph_caps=None,
        inference_batching=None,
    )
    # The persistent poller would finalize a round the drive is about to finalize itself; a
    # race here would make WHICH path minted the round id nondeterministic.
    pipeline._stop_event.set()          # noqa: SLF001 -- deliberate, test-only quiescing
    pipeline._poller.join(5.0)          # noqa: SLF001
    assert not pipeline._poller.is_alive()   # noqa: SLF001
    return pipeline


def test_neither_mid_run_route_writes_the_terminal_latch(tmp_path, monkeypatch) -> None:
    """O-05, node 1: neither mid-run route may reach the terminal recorder, and the PREMISE
    assertions stop this passing vacuously.

    MUTATION THAT REDS IT (M-O5): call `_record_terminal_outcome` from `flush_pending_eval`."""
    broken = _broken_round("killed", round_id="r000001_3", step=3)
    pipeline = _FakeEvalPipeline(poll_result=dict(broken), drain_result=dict(broken))
    harness = _make_coordinator(eval_pipeline=pipeline, sink=_SpySink())
    coord = harness.coord

    assert coord.terminal_eval_reason is None, (
        "PREMISE — the coordinator must EXPOSE the terminal latch, unset. Without the "
        "attribute this row asserts nothing about the split"
    )
    assert callable(coord.record_terminal_eval_reason), (
        "PREMISE — …and its set-once writer, so 'the mid-run route did not call it' is a "
        "statement about a mechanism that exists"
    )

    coord.step()                          # route 1: step()'s async poll
    assert pipeline.polled >= 1, "premise: step() polled the pipeline"
    drain.flush_pending_eval(coord)       # route 2: the teardown flush

    assert coord.terminal_eval_reason is None, (
        "a MID-RUN broken round wrote the terminal latch — R133's split says mid-run "
        f"breakage stays non-fatal; got {coord.terminal_eval_reason!r}"
    )
    assert harness.shutdown.abort_rule is None, (
        f"…and no abort rule may be recorded for it; got {harness.shutdown.abort_rule!r}"
    )


def test_a_run_whose_MID_RUN_round_broke_still_exits_zero(tmp_path, monkeypatch, request) -> None:
    """O-05, node 2 — the same claim at the process boundary: a mid-run break with a clean
    terminal round must exit 0, with the latch observably present-and-None.

    MUTATION THAT REDS IT (M-O5): as above — this run would exit 48."""
    pipeline = _FakeEvalPipeline(
        terminal_result=_clean_round(), poll_result=_broken_round("result_invalid",
                                                                 round_id="r000001_1", step=1),
    )
    drive = _drive_main(tmp_path, monkeypatch, request, pipeline=pipeline)

    assert pipeline.terminal_calls == [True], (
        "premise: the terminal battery ran exactly once, with the stride ignored; got "
        f"{pipeline.terminal_calls}"
    )
    assert drive.handles.coordinator.terminal_eval_reason is None, (
        "PREMISE + claim: the latch exists and a CLEAN terminal round left it unset, so the "
        "rc below is about the mid-run break and not about a missing mechanism; got "
        f"{drive.handles.coordinator.terminal_eval_reason!r}"
    )
    assert drive.handles.shutdown.abort_rule is None and drive.rc == 0, (
        "a run whose mid-run eval round broke and whose terminal battery was clean is a "
        f"CLEAN run: rounds recur. got rule={drive.handles.shutdown.abort_rule!r} "
        f"rc={drive.rc!r}"
    )


def test_the_latched_reason_is_the_routed_results_own_value(tmp_path) -> None:
    """O-06, all seven reasons: the latch is a COPY of `result["eval_broken_reason"]` made in
    ONE expression off the routed mapping, so it cannot disagree with its round.

    MUTATION THAT REDS IT (M-O6): latch a constant. rc 48 is still 48 and O-08 stays green."""
    for reason in _SEVEN_REASONS:
        pipeline = _FakeEvalPipeline(terminal_result=_broken_round(reason))
        harness = _make_coordinator(eval_pipeline=pipeline, sink=_SpySink())
        routed = drain.run_terminal_eval(harness.coord)

        assert routed["eval_broken_reason"] == reason, "premise: the rigged round routed"
        assert harness.coord.terminal_eval_reason == routed["eval_broken_reason"], (
            f"reason {reason!r}: the latch says "
            f"{harness.coord.terminal_eval_reason!r} while the routed round says "
            f"{routed['eval_broken_reason']!r} — one fact, read once"
        )


def _calls_named(name: str) -> list[tuple[str, str, str]]:
    """Every call to `name` under `src/mantis`, as (module, enclosing function, receiver).

    AST, not grep: grep cannot tell a call from a `def`, from a docstring mention, or from the
    RECEIVER — and the receiver is the whole of conjunct (ii)."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        enclosing: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    enclosing.setdefault(id(child), node.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == name:
                receiver = ast.unparse(func.value)
            elif isinstance(func, ast.Name) and func.id == name:
                receiver = ""
            else:
                continue
            found.append((str(path.relative_to(_SRC.parent)),
                          enclosing[id(node)] if id(node) in enclosing else "<module>",
                          receiver))
    return found


def test_the_terminal_latch_has_one_writer_and_the_terminal_route_has_no_third_caller() -> None:
    """O-07, TWO conjuncts, both scoped to `src/`.

    (i) `record_terminal_eval_reason(` has exactly ONE production call site, REACHABLE ONLY
    from `drain.run_terminal_eval`. Reachability and not lexical containment: the write lives
    in a one-line private helper, which must itself have one call site inside
    `run_terminal_eval`.

    (ii) `drain.run_terminal_eval` has a SECOND entry point, the public delegate
    `StepCoordinator.run_terminal_eval`, with zero callers in `src/`; a future mid-loop caller
    would write the latch mid-run while (i) stayed green. Scoped to `src/` because `tests/`
    legitimately drives `drain.run_terminal_eval`.

    MUTATIONS: M-O7 reds conjunct (i); M-O7b reds conjunct (ii) and leaves (i) untouched."""
    writers = _calls_named("record_terminal_eval_reason")
    assert len(writers) == 1, (
        "the terminal latch must have EXACTLY ONE writer in `src/` — a second writer is a "
        f"second route into the rc that nothing else in this file can see. Found: {writers}"
    )
    module, enclosing, _receiver = writers[0]
    assert module == "mantis/train/coordinator/drain.py", (
        "…and it must live in the terminal-eval slice, which is the only module that passes "
        f"`ignore_stride=True`. Found it in {module}::{enclosing}"
    )
    if enclosing != "run_terminal_eval":
        helper_calls = _calls_named(enclosing)
        assert len(helper_calls) == 1, (
            f"the writer sits in the helper {enclosing!r}, and that helper has "
            f"{len(helper_calls)} call sites in `src/` — every one of them is a route into "
            f"the terminal latch. Found: {helper_calls}"
        )
        helper_module, helper_enclosing, _ = helper_calls[0]
        assert (helper_module == "mantis/train/coordinator/drain.py"
                and helper_enclosing == "run_terminal_eval"), (
            f"…and the helper {enclosing!r} must be reachable ONLY from "
            f"`drain.run_terminal_eval`; it is called from {helper_module}::{helper_enclosing}"
        )

    terminal_calls = _calls_named("run_terminal_eval")
    assert len(terminal_calls) == 2, (
        "conjunct (ii): `run_terminal_eval` is called from exactly two places in `src/` — "
        "`drain.close_out` and the `StepCoordinator` delegate that forwards to `drain`. A "
        f"third site is a new route into the terminal battery. Found: {terminal_calls}"
    )
    # The two legitimate shapes and only these: `drain.py`'s own bare-name call, and the
    # delegate forwarding to the module. Any other receiver is the mid-loop route (ii) forbids.
    delegate_callers = [
        site for site in terminal_calls
        if not (site[2] == "drain"
                or (site[2] == "" and site[0] == "mantis/train/coordinator/drain.py"))
    ]
    assert delegate_callers == [], (
        "conjunct (ii): the public delegate `StepCoordinator.run_terminal_eval` must have "
        "ZERO callers in `src/`. A mid-loop caller writes the terminal latch during "
        f"training, which is R133's split broken through the back door. Found: {delegate_callers}"
    )


@pytest.mark.parametrize("reason", _SEVEN_REASONS)
def test_a_broken_terminal_round_exits_48(reason, tmp_path, monkeypatch, request) -> None:
    """O-08 — a broken terminal round exits 48; measured at HEAD it returned 0, so a run with
    no promotion decision at all was indistinguishable to the supervisor. The rc is asserted
    THROUGH the manifest row and the ONE resolver, never against a literal.

    MUTATION THAT REDS IT (M-O8): delete `shutdown.record_abort(TERMINAL_EVAL_BROKEN_ABORT_
    RULE)` from `run.py`'s teardown; every other row stays green."""
    pipeline = _FakeEvalPipeline(terminal_result=_broken_round(reason))
    drive = _drive_main(tmp_path, monkeypatch, request, pipeline=pipeline)

    assert pipeline.terminal_calls == [True], (
        f"premise: the terminal battery ran; got {pipeline.terminal_calls}"
    )
    assert drive.handles.coordinator.terminal_eval_reason == reason, (
        f"premise: the round's reason {reason!r} reached the coordinator's latch; got "
        f"{drive.handles.coordinator.terminal_eval_reason!r}"
    )
    assert drive.handles.shutdown.abort_rule == _TERMINAL_RULE, (
        "the composition root must RECORD which rule stopped the run — `abort_rule is None` "
        f"is the only thing that means a clean run. Got {drive.handles.shutdown.abort_rule!r}"
    )
    row = next((r for r in MANIFEST if r.name == _TERMINAL_RULE), None)
    assert row is not None, (
        f"the armed-abort manifest authors no {_TERMINAL_RULE!r} row, so the resolver below "
        "answers None and an aborted run reports rc 1 UnregisteredAbortExitError"
    )
    assert drive.rc == row.exit_code == exit_code_for_abort(_TERMINAL_RULE) == 48, (
        f"reason {reason!r}: the rc IS the row's `exit_code`, resolved through the ONE "
        f"resolver; got rc={drive.rc!r} against row={row.exit_code!r}"
    )


def test_a_clean_terminal_round_exits_zero(tmp_path, monkeypatch, request) -> None:
    """O-08's CONTROL: a latch that fired on EVERY terminal round would pass all seven rows
    above and ship a run that always exits 48. The in-file twin of the integration-tier
    `tests/test_run_launcher.py` drive over a real bounded `launch_run`."""
    pipeline = _FakeEvalPipeline(terminal_result=_clean_round())
    drive = _drive_main(tmp_path, monkeypatch, request, pipeline=pipeline)

    assert pipeline.terminal_calls == [True], "premise: the terminal battery ran"
    assert drive.handles.coordinator.terminal_eval_reason is None, (
        "a CLEAN terminal round leaves the latch unset — 'clean' is the absence of a reason, "
        f"not a second boolean; got {drive.handles.coordinator.terminal_eval_reason!r}"
    )
    assert drive.handles.shutdown.abort_rule is None and drive.rc == 0, (
        f"…and the run exits 0. got rule={drive.handles.shutdown.abort_rule!r} rc={drive.rc!r}"
    )


def test_a_disk_full_run_whose_terminal_eval_also_broke_reports_47(
    tmp_path, monkeypatch, request
) -> None:
    """O-09 arm (a). ORDER is the argument: the volume kills the run, the terminal battery then
    breaks BECAUSE of it, and `record_abort` is set-once.

    MUTATION THAT REDS IT (M-O9): move the terminal read before the disk-guard read."""
    pipeline = _FakeEvalPipeline(terminal_result=_broken_round("join_timeout"))
    drive = _drive_main(tmp_path, monkeypatch, request, pipeline=pipeline,
                        free_gb=_CRITICAL_GB, wait_for_fire=True)
    _await_signal(drive.handles.shutdown)

    assert drive.guards and drive.guards[-1].critical_fired, (
        "premise: the rigged volume drove the guard's critical arm"
    )
    assert drive.handles.coordinator.terminal_eval_reason == "join_timeout", (
        "PREMISE, and the whole of what makes this row non-vacuous: the terminal round DID "
        "break and DID latch, so the rc below is set-once winning rather than the terminal "
        f"seam being absent. Got {drive.handles.coordinator.terminal_eval_reason!r}"
    )
    assert drive.handles.shutdown.abort_rule == DISK_SPACE_ABORT_RULE, (
        "the FIRST fire wins and it is the root cause — a disk-full event that then breaks "
        f"the terminal eval is a disk-full run. Got {drive.handles.shutdown.abort_rule!r}"
    )
    assert drive.rc == exit_code_for_abort(DISK_SPACE_ABORT_RULE) == 47, (
        f"…so the supervisor reads 47, not 48. Got {drive.rc!r}"
    )


def test_a_draw_rate_collapse_that_precedes_a_broken_terminal_round_reports_46(
    tmp_path, monkeypatch, request
) -> None:
    """O-09 arm (b): the mid-loop fire is recorded BEFORE the epilogue runs, so a collapse is
    not re-labelled by the terminal battery that breaks with it.

    MUTATION THAT REDS IT: make `record_abort` last-writer-wins instead of set-once."""
    pipeline = _FakeEvalPipeline(terminal_result=_broken_round("exit_nonzero"))
    drive = _drive_main(
        tmp_path, monkeypatch, request, pipeline=pipeline, draw_counts=(50, 50),
        draw_rate_abort={"threshold": 0.5, "min_step": 1, "N_pool_min": 50, "consec": 1},
    )

    assert drive.handles.coordinator.terminal_eval_reason == "exit_nonzero", (
        "PREMISE: the terminal round broke and latched, so set-once really was asked to "
        f"overwrite. Got {drive.handles.coordinator.terminal_eval_reason!r}"
    )
    assert drive.handles.shutdown.abort_rule == "draw_rate_collapse", (
        f"the collapse keeps its name; got {drive.handles.shutdown.abort_rule!r}"
    )
    assert drive.rc == DRAW_RATE_COLLAPSE_EXIT_CODE == 46, (
        f"…and its number; got {drive.rc!r}"
    )


def test_an_unregistered_reason_spelling_raises_at_the_root(tmp_path, monkeypatch, request) -> None:
    """O-10 — the RUNTIME half of the unrepresentability claim: a `# type: ignore` slips past
    pyright and the result crosses a JSON boundary, so the root RE-PARSES the latched string.

    MUTATION THAT REDS IT (M-O10): replace `EvalBrokenReason(raw)` with `if raw:`."""
    pipeline = _FakeEvalPipeline(terminal_result=_broken_round("not_a_registered_reason"))

    with pytest.raises(ValueError, match="not_a_registered_reason"):
        _drive_main(tmp_path, monkeypatch, request, pipeline=pipeline)


def test_a_terminal_round_is_marked_terminal_in_the_stream(tmp_path, monkeypatch) -> None:
    """O-32, node 1: both stream discriminators exist at HEAD and nothing reads either, so
    their absence would be invisible. Driven through the REAL `EvalPipeline` over one sink.

    MUTATION THAT REDS IT (M-O32): drop the `_terminal` suffix from the round-id format."""
    ctx = _FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    sink = _SpySink()
    pipeline = _real_pipeline(tmp_path, sink)
    harness = _make_coordinator(eval_pipeline=pipeline, sink=sink)
    try:
        drain.run_terminal_eval(harness.coord)
    finally:
        pipeline.stop()

    broken = sink.named("eval_broken")
    assert broken, "premise: the rigged worker never exited, so the terminal round broke"
    round_id = broken[-1]["round_id"]
    assert round_id.endswith("_terminal"), (
        "a TERMINAL round must be self-identifying in the ONE channel: without the suffix a "
        "reader cannot tell the final battery from any other round, and rc 48 gives no round "
        f"id at all. Got {round_id!r}"
    )
    order = sink.order()
    assert "terminal_eval" in order, (
        f"…and the terminal battery announces itself BEFORE it runs; stream: {order}"
    )
    assert order.index("terminal_eval") < order.index("eval_broken"), (
        f"the announcement must precede the round's own events; stream: {order}"
    )


def test_a_mid_run_round_carries_neither_terminal_discriminator(tmp_path, monkeypatch) -> None:
    """O-32, node 2 — the contrast arm, without which node 1 is satisfied by stamping
    `_terminal` on every round. A mid-run round must carry neither discriminator."""
    ctx = _FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    sink = _SpySink()
    pipeline = _real_pipeline(tmp_path, sink)
    harness = _make_coordinator(eval_pipeline=pipeline, sink=sink)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={},
                                      best_model_step=None)
        assert ack["kicked"] is True, "premise: a mid-run round was kicked"
        drain.flush_pending_eval(harness.coord)
    finally:
        pipeline.stop()

    broken = sink.named("eval_broken")
    assert broken, "premise: the rigged worker never exited, so the mid-run round broke too"
    round_id = broken[-1]["round_id"]
    assert not round_id.endswith("_terminal"), (
        f"a mid-run round must not wear the terminal marker; got {round_id!r}"
    )
    order = sink.order()
    assert "terminal_eval" not in order, (
        f"…and the terminal announcement must not be emitted for it; stream: {order}"
    )
    assert "flush_pending_eval" in order, (
        f"premise: this WAS the teardown flush route; stream: {order}"
    )
