"""⊕ WP11-A — the `eval_round` heartbeat source (build_run_safety caller + poller beat).

RED-at-import until IMPL writes `mantis.eval.pipeline`. ORACLE-FIRST (⊕): the top-level
`import mantis.eval.pipeline` raises ModuleNotFoundError before any port code exists — that
import failure carries the whole file (`HEARTBEAT_SOURCES`, `MonitorConfig`,
`HeartbeatWatchdog`, `build_run_safety` all ALREADY EXIST at HEAD, WP13-A; only the
`"eval_round"` 4th source + its deadline field are new).

Today `HEARTBEAT_SOURCES == ("train_step", "inference_dispatch", "selfplay_drain")` (3
sources; `src/mantis/monitor/heartbeat.py`, already read). IMPL adds `"eval_round"` as the
4th source, a `heartbeat_deadline_eval_round_sec` field on `MonitorConfig`
(`src/mantis/monitor/config.py`), and the matching entry in `build_run_safety`'s deadlines
dict (`src/mantis/train/subsystems.py:239-243`). The pipeline's persistent poller/keepalive
thread (§c.3) beats `"eval_round"` on EVERY tick, idle or active — a between-round gap can
never false-fire the watchdog (round PROGRESS is bounded separately, by `round_timeout_sec`).
"""
from __future__ import annotations

import mantis.eval.pipeline  # noqa: F401 — RED-at-import anchor
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import (
    HEARTBEAT_SOURCES,
    WATCHDOG_STALL_EXIT_CODE,
    HeartbeatRegistry,
)
from mantis.train.lifecycle.heartbeat_watchdog import HeartbeatWatchdog
from mantis.train.subsystems import build_run_safety


class _ExitSpy:
    def __init__(self) -> None:
        self.codes: list[int] = []

    def __call__(self, code: int) -> None:
        self.codes.append(int(code))


class SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


class FakeBuffer:
    def save_to_path(self, p) -> None:
        return None


def test_eval_round_is_a_registered_heartbeat_source() -> None:
    """`"eval_round"` must join the 3 shipped sources as the 4th (§c.8)."""
    assert "eval_round" in HEARTBEAT_SOURCES, (
        f"HEARTBEAT_SOURCES must gain eval_round (4th source): {HEARTBEAT_SOURCES}"
    )
    assert len(HEARTBEAT_SOURCES) == 4, (
        f"expected exactly 4 heartbeat sources post-WP11-A: {HEARTBEAT_SOURCES}"
    )


def test_monitor_config_carries_eval_round_deadline() -> None:
    """`MonitorConfig.heartbeat_deadline_eval_round_sec` — mint: 1800.0 (§a.4 row)."""
    cfg = MonitorConfig()
    assert hasattr(cfg, "heartbeat_deadline_eval_round_sec"), (
        "MonitorConfig must gain heartbeat_deadline_eval_round_sec"
    )
    assert cfg.heartbeat_deadline_eval_round_sec == 1800.0


def test_poller_thread_beats_eval_round() -> None:
    """The pipeline's persistent poller thread beats the LITERAL `"eval_round"` heartbeat
    source on every tick, WITH or WITHOUT an in-flight round (idle beats too — the source
    proves the enforcement thread is alive; round progress is bounded separately)."""
    registry = HeartbeatRegistry()
    beats: list[str] = []
    real_beat = registry.beat

    def _spy_beat(source: str) -> None:
        beats.append(source)
        real_beat(source)

    pipeline = mantis.eval.pipeline.build_eval_pipeline(
        leaf_batch_size=1,
        eval_cfg=object(), coordinator_cfg_caps=object(), encoding="gnn_axis_v1",
        run_id="test-run", spool_dir="/tmp/mantis-eval-heartbeat-test",
        ladder_state_path="/tmp/mantis-eval-heartbeat-test/ladder.json",
        promotion=object(), sink=None, heartbeat=_spy_beat,
        # F-816-10 D-1: resolved once in the parent, carried on every RoundSpec. This
        # drive never builds an engine, so the value is inert here — but the parameter
        # carries no default, so the decision is written rather than omitted.
        fused_graph_caps=None,
    )
    try:
        import time

        time.sleep(0.05)  # let the poller thread tick at least once while IDLE
        assert "eval_round" in beats, (
            "the poller thread must beat 'eval_round' even with no round in flight"
        )
    finally:
        pipeline.stop()


def test_build_run_safety_arms_eval_round_deadline(tmp_path) -> None:
    """`build_run_safety(..., wired_sources=[..., "eval_round"])` must construct cleanly (no
    missing-deadline ValueError) and the arm event must name the `eval_round` source — i.e.
    once `eval_round` is a real HEARTBEAT_SOURCES member with a deadline wired into
    `build_run_safety`'s deadlines dict, arming succeeds and `eval_round` appears in the
    `heartbeat_watchdog_armed` event's `sources`, deadlines, and `wired_sources`."""
    sink = SpySink()
    run_safety = build_run_safety(
        log_dir=tmp_path, run_id="test-run", buffer=FakeBuffer(),
        buffer_persist_path=tmp_path / "replay.bin",
        wired_sources=["train_step", "inference_dispatch", "selfplay_drain", "eval_round"],
        # WP-UNFREEZE E36 fallout: the two lag-fn kwargs are REQUIRED (no defaults).
        # WPAX RED-TEAM F-2: `monitor_cfg` is REQUIRED too. This test asserts the
        # `eval_round` deadline arms at the MonitorConfig DEFAULT 1800.0, so the default
        # instance is now passed explicitly rather than reached through an absent-kwarg
        # fallback that also silently disarmed the actor-lag abort.
        monitor_cfg=MonitorConfig(),
        actor_ckpt_step_fn=lambda: 0, learner_step_fn=lambda: 0,
    )
    run_safety.watchdog._sink = sink  # route the arm-log through our spy without a real JSONL
    run_safety.watchdog.arm()

    armed = sink.named("heartbeat_watchdog_armed")
    assert armed, "arm() must emit exactly one heartbeat_watchdog_armed event"
    event = armed[-1]
    assert "eval_round" in event["sources"], f"eval_round missing from armed sources: {event}"
    assert event["deadlines"].get("eval_round") == 1800.0, (
        f"eval_round must arm with the MonitorConfig default deadline 1800.0: {event}"
    )
    assert "eval_round" in event["wired_sources"], (
        f"eval_round must be recorded as a wired source when declared: {event}"
    )


def test_stale_eval_poller_fires_42_under_fake_clock(tmp_path) -> None:
    """A wedged `eval_round` poller (never beats past its deadline) fires
    `WATCHDOG_STALL_EXIT_CODE` (42) via the injected `exit_fn`, exactly like the 3 shipped
    sources — under a FAKE clock, no real sleeping."""
    sink = SpySink()
    exit_spy = _ExitSpy()
    registry = HeartbeatRegistry(sources=("train_step", "inference_dispatch",
                                          "selfplay_drain", "eval_round"))
    fake_time = {"t": 0.0}

    def _clock() -> float:
        return fake_time["t"]

    registry_clock_patchable = HeartbeatRegistry(
        clock=_clock, sources=("train_step", "inference_dispatch", "selfplay_drain",
                               "eval_round"),
    )
    watchdog = HeartbeatWatchdog(
        registry=registry_clock_patchable,
        deadlines={"train_step": 100.0, "inference_dispatch": 100.0,
                  "selfplay_drain": 100.0, "eval_round": 5.0},
        sink=sink, counters_fn=lambda: 0, heartbeat_file=tmp_path / "hb.json",
        file_interval_sec=1.0, poll_interval_sec=1.0, clock=_clock,
        save_snapshot=lambda: None, exit_fn=exit_spy,
        wired_sources=["train_step", "inference_dispatch", "selfplay_drain", "eval_round"],
    )
    watchdog.arm()
    fake_time["t"] = 10.0  # past the eval_round 5.0 s deadline; others (100.0) still fine
    watchdog.poll_once()

    assert WATCHDOG_STALL_EXIT_CODE in exit_spy.codes, (
        "a stale eval_round source must fire the shared 42 stall exit code"
    )


def test_headless_launch_without_pipeline_is_unwired_loud_not_fatal(tmp_path) -> None:
    """When `eval_round` is a known+deadlined source but nothing ever beat it (a headless
    launch with no eval pipeline wired), the watchdog must emit `heartbeat_source_unwired`
    for it and NEVER fire 42 for it (killing a healthy pipeline-less run is the worse
    failure)."""
    sink = SpySink()
    exit_spy = _ExitSpy()
    fake_time = {"t": 0.0}

    def _clock() -> float:
        return fake_time["t"]

    registry = HeartbeatRegistry(
        clock=_clock, sources=("train_step", "inference_dispatch", "selfplay_drain",
                               "eval_round"),
    )
    watchdog = HeartbeatWatchdog(
        registry=registry,
        deadlines={"train_step": 5.0, "inference_dispatch": 5.0,
                  "selfplay_drain": 5.0, "eval_round": 5.0},
        sink=sink, counters_fn=lambda: 0, heartbeat_file=tmp_path / "hb.json",
        file_interval_sec=1.0, poll_interval_sec=1.0, clock=_clock,
        save_snapshot=lambda: None, exit_fn=exit_spy,
        # eval_round deliberately NOT declared wired (headless launch, no pipeline built).
        wired_sources=["train_step", "inference_dispatch", "selfplay_drain"],
    )
    watchdog.arm()
    registry.beat("train_step")
    registry.beat("inference_dispatch")
    registry.beat("selfplay_drain")
    fake_time["t"] = 10.0  # past every deadline, but eval_round is undeclared/never-beaten
    # Beat the three WIRED sources again at the jumped clock so only eval_round is stale —
    # otherwise a wired-but-unbeaten-since-t0 source (e.g. train_step) would fire the shared
    # 42 first and the eval_round-unwired path under test would never be reached (the
    # fixture bug this correction fixes: staleness is checked source-by-source in order, and
    # `_check_source_staleness` returns on the FIRST fire it finds).
    registry.beat("train_step")
    registry.beat("inference_dispatch")
    registry.beat("selfplay_drain")
    watchdog.poll_once()

    assert WATCHDOG_STALL_EXIT_CODE not in exit_spy.codes, (
        "an unwired eval_round must NEVER fire a stall abort on a healthy pipeline-less run"
    )
    unwired = sink.named("heartbeat_source_unwired")
    assert any(e.get("source") == "eval_round" for e in unwired), (
        "an unwired eval_round must emit a LOUD heartbeat_source_unwired event"
    )
