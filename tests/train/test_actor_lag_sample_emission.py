"""Oracle for the `actor_lag_sample` emission on the heartbeat watchdog's lag check.

RED-at-EXECUTION, not at collection: at HEAD every drive observes ZERO samples, because the lag
check emits only on `lag < 0` or `lag > threshold`. Without the sample a healthy run emits
nothing from the lag check, so no observer can tell a live reading from a frozen 0.

R8 >300 justify: ONE unit — the emission, the two structural gates it inherits and the
constructor CENSUS that keeps its interval derived rather than parameterised are one negotiated
remedy; the census read alone looks like an arbitrary signature freeze.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from mantis.monitor.sink import JsonlEventSink
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog

# The event name is the contract surface: the event manifest gains a row for it in the SAME
# commit as this file.
SAMPLE_EVENT = "actor_lag_sample"

# The four keys `_check_actor_lag` already builds into `detail` and hands to the fire path. The
# sample must carry the same four, because it is the SAME dict.
DETAIL_KEYS = ("learner_step", "actor_ckpt_step", "lag_steps", "threshold_steps")


def _registry():
    return SimpleNamespace(
        sources=("train_step",), ages=lambda: {"train_step": 0.0},
        beaten_sources=lambda: frozenset({"train_step"}), arm=lambda: None,
    )


def _watchdog(tmp_path: Path, *, spec, sink, file_interval_sec=0.0, clock=None, codes=None):
    """Silence staleness structurally (`deadline <= 0`) so every event comes from the lag check;
    the watchdog THREAD is never started, `poll_once()` is driven directly."""
    return HeartbeatWatchdog(
        registry=_registry(), deadlines={"train_step": 0.0}, sink=sink,
        counters_fn=lambda: 0, heartbeat_file=tmp_path / "hb.json",
        file_interval_sec=file_interval_sec, poll_interval_sec=0.0,
        clock=clock if clock is not None else (lambda: 0.0),
        save_snapshot=lambda: None,
        exit_fn=(codes.append if codes is not None else (lambda code: None)),
        snapshot_timeout_sec=2.0, wired_sources=["train_step"], actor_lag=spec,
    )


def _spec(*, learner, actor, threshold=100, armed=False):
    return ActorLagSpec(learner_step_fn=learner, actor_ckpt_step_fn=actor,
                        threshold_steps=threshold, abort_enabled=armed)


def _read(sink: JsonlEventSink, name: str) -> list[dict]:
    """Decode the REAL segment file the sink wrote and filter by event name — this file IS the
    preflight's observation transport, so an emission that never reaches it is invisible."""
    lines = [ln for ln in sink.path.read_text().splitlines() if ln.strip()]
    return [e for e in (json.loads(ln) for ln in lines) if e.get("event") == name]


def test_a_healthy_poll_emits_an_actor_lag_sample_carrying_the_live_reading(tmp_path) -> None:
    """A HEALTHY poll puts the reading on the wire. The values are rigged to distinct non-zero
    integers so a stub emitting a constant or echoing the threshold back fails."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5a")
    wd = _watchdog(tmp_path, sink=sink, spec=_spec(learner=lambda: 37, actor=lambda: 31,
                                                   threshold=100, armed=True))
    wd.poll_once()

    samples = _read(sink, SAMPLE_EVENT)
    assert len(samples) == 1, (
        f"a healthy poll must emit exactly one {SAMPLE_EVENT}; got {len(samples)}. At HEAD "
        "the lag check emits ONLY on `lag < 0` or `lag > threshold`, so assertion (b) of "
        "the mint preflight has no subject (DESIGN_P §3.3 TD-6)"
    )
    sample = samples[0]
    assert {k: sample.get(k) for k in DETAIL_KEYS} == {
        "learner_step": 37, "actor_ckpt_step": 31, "lag_steps": 6, "threshold_steps": 100,
    }, f"the sample must carry the LIVE reading verbatim; got {sample!r}"
    assert sample.get("seq") == 0, (
        "the sample carries the watchdog's own poll sequence so a reader can place it in "
        f"the mirror's timeline; got {sample.get('seq')!r}"
    )
    assert _read(sink, "heartbeat_watchdog_fired") == [], "a healthy poll must not fire"


def test_the_sample_and_the_exceedance_event_carry_THE_SAME_detail(tmp_path) -> None:
    """The sample IS the fire path's own `detail` dict, so it can never disagree with the
    reading that fires. Driven over threshold but DISARMED, so both events land on one poll."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5b")
    wd = _watchdog(tmp_path, sink=sink, spec=_spec(learner=lambda: 900, actor=lambda: 100,
                                                   threshold=500, armed=False))
    wd.poll_once()

    samples, exceeded = _read(sink, SAMPLE_EVENT), _read(sink, "actor_lag_exceeded")
    assert len(samples) == 1 and len(exceeded) == 1, (
        f"one poll over a disarmed threshold must produce both events; got "
        f"{len(samples)} sample(s) and {len(exceeded)} exceedance event(s)"
    )
    assert {k: samples[0].get(k) for k in DETAIL_KEYS} == \
           {k: exceeded[0].get(k) for k in DETAIL_KEYS} == {
        "learner_step": 900, "actor_ckpt_step": 100, "lag_steps": 800, "threshold_steps": 500,
    }, (
        "the sample and the exceedance must carry IDENTICAL detail — they are the same "
        f"dict. sample={samples[0]!r} exceeded={exceeded[0]!r}"
    )


def test_a_negative_lag_still_emits_a_sample_beside_the_wiring_bug_event(tmp_path) -> None:
    """A negative lag is still SAMPLED: the preflight reads `lag_steps < 0` off the sample too,
    so an emission placed after the `lag < 0` arm would silence exactly the wiring defect."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5c")
    wd = _watchdog(tmp_path, sink=sink, spec=_spec(learner=lambda: 10, actor=lambda: 50,
                                                   threshold=5, armed=True))
    wd.poll_once()

    samples = _read(sink, SAMPLE_EVENT)
    assert len(samples) == 1, f"a negative lag must still be SAMPLED; got {len(samples)}"
    assert samples[0].get("lag_steps") == -40, (
        f"the sample must report the negative reading honestly; got {samples[0]!r}"
    )
    assert len(_read(sink, "actor_lag_negative")) == 1, (
        "the existing wiring-bug report must be unchanged by the new emission"
    )


def test_the_sample_reads_the_callables_live_across_polls(tmp_path) -> None:
    """A reading captured at ctor — or a constant — reproduces the first row forever; ordered
    equality over three polls, so a stream that merely *moves* fails too."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5d")
    state = {"learner": 5, "actor": 5}
    wd = _watchdog(tmp_path, sink=sink,
                   spec=_spec(learner=lambda: state["learner"], actor=lambda: state["actor"]))
    driven = [(5, 5), (40, 12), (91, 90)]
    for learner, actor in driven:
        state["learner"], state["actor"] = learner, actor
        wd.poll_once()

    samples = _read(sink, SAMPLE_EVENT)
    observed = [(s.get("learner_step"), s.get("actor_ckpt_step"), s.get("lag_steps"))
                for s in samples]
    assert observed == [(learner, actor, learner - actor) for learner, actor in driven], (
        "the samples must track BOTH callables poll by poll, in order — a captured or "
        f"constant reading cannot produce this sequence. driven={driven} observed={observed}"
    )


def test_the_sample_is_gated_by_the_file_interval_ALREADY_in_the_object(tmp_path) -> None:
    """The sample interval is DERIVED from `self._file_interval` — the value already passed as
    `file_interval_sec` — so one config fact never enters this ctor twice under two names. Both
    arms are asserted, because only the pair pins a derivation."""
    now = {"t": 0.0}
    gated_sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5e_gated")
    gated = _watchdog(tmp_path, sink=gated_sink, file_interval_sec=10.0,
                      clock=lambda: now["t"],
                      spec=_spec(learner=lambda: int(now["t"]) + 1, actor=lambda: 0))
    for t in (0.0, 1.0, 9.9, 10.0, 10.1, 20.0):
        now["t"] = t
        gated.poll_once()
    gated_ts = [s.get("learner_step") for s in _read(gated_sink, SAMPLE_EVENT)]
    assert gated_ts == [1, 11, 21], (
        "at `file_interval_sec=10.0` the emission must fire on the first poll and then "
        f"only once per 10 s of the INJECTED clock; got learner_steps {gated_ts}"
    )

    now["t"] = 0.0
    open_sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5e_open")
    ungated = _watchdog(tmp_path, sink=open_sink, file_interval_sec=0.0,
                        clock=lambda: now["t"],
                        spec=_spec(learner=lambda: int(now["t"]) + 1, actor=lambda: 0))
    for t in (0.0, 1.0, 2.0):
        now["t"] = t
        ungated.poll_once()
    assert [s.get("learner_step") for s in _read(open_sink, SAMPLE_EVENT)] == [1, 2, 3], (
        "at `file_interval_sec=0.0` the emission must fire on EVERY poll — the same rule "
        "`_mirror_file` already follows (`:408-421`); one rule, two consumers. This is the "
        "arm that keeps byte-frozen tests/train/test_actor_lag_watchdog.py:64 (which passes "
        "`file_interval_sec=0.0`) behaving as DESIGN_P §7.2 measured it"
    )


def test_heartbeat_watchdog_init_gains_NO_new_parameter() -> None:
    """The constructor gains NO new parameter, as a signature CENSUS rather than a spot check.

    The rejected shape was a required `lag_sample_interval_sec` kwarg, measured to turn 37 tests
    red across five files. `monitor_liveness` is admitted against this census because it is
    OPTIONAL, turned exactly one test red, and derives its interval from `file_interval_sec` too.
    """
    params = inspect.signature(HeartbeatWatchdog.__init__).parameters
    assert tuple(params) == (
        "self", "registry", "deadlines", "sink", "counters_fn", "heartbeat_file",
        "file_interval_sec", "poll_interval_sec", "clock", "save_snapshot", "exit_fn",
        "close_out_deadline_sec", "snapshot_timeout_sec", "wired_sources", "actor_lag",
        "monitor_liveness",
    ), (
        "HeartbeatWatchdog.__init__ must be UNCHANGED by TD-6 (MF-1, Remedy A): the sample "
        f"interval derives from the existing file_interval_sec. Got {tuple(params)}"
    )
    assert params["monitor_liveness"].default == (), (
        "monitor_liveness must stay OPTIONAL: a required kwarg here is exactly the shape "
        "TD-6 measured at 37 red tests, and this parameter's whole admissibility rests on "
        "not being that"
    )
    for name in ("registry", "deadlines", "sink", "counters_fn", "heartbeat_file",
                 "file_interval_sec", "poll_interval_sec", "save_snapshot"):
        assert params[name].default is inspect.Parameter.empty, (
            f"{name} must stay REQUIRED — a default here is a code-side default authority "
            "for a config fact (R1)"
        )


def test_no_sample_during_close_out_and_none_without_a_spec(tmp_path) -> None:
    """The emission inherits the two gates that already bound `_check_actor_lag` instead of
    becoming a third, which is what keeps it out of the close-out window."""
    closing_sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5f_closeout")
    wd = _watchdog(tmp_path, sink=closing_sink,
                   spec=_spec(learner=lambda: 1000, actor=lambda: 0, threshold=10, armed=True))
    wd.disarm_staleness()
    wd.poll_once()
    assert _read(closing_sink, SAMPLE_EVENT) == [], (
        "close-out freezes both counters; a sample there would be a stale reading the "
        "preflight would then have to defend against"
    )

    bare_sink = JsonlEventSink(log_dir=tmp_path, run_id="oracle_p_c5f_nospec")
    _watchdog(tmp_path, sink=bare_sink, spec=None).poll_once()
    assert _read(bare_sink, SAMPLE_EVENT) == [], (
        "a watchdog with no ActorLagSpec has nothing to sample — direct-ctor and non-run "
        "contexts must stay silent (the `absent, LOUD, never silent` posture at `:220` is "
        "the arm event's job, not the sample's)"
    )
