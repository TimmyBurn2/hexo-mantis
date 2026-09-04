"""⊕ WPAX Phase P ORACLE — C-5: TD-6's `actor_lag_sample` emission (DESIGN_P §7.2, M-1).

RED-at-EXECUTION, not at collection: TD-6 is a MODIFICATION to a module that already
imports (`mantis.train.lifecycle.heartbeat_watchdog`), so there is no import anchor to
hang a collection error on. At HEAD every drive below observes ZERO `actor_lag_sample`
events, because `_check_actor_lag` emits only on `lag < 0` (`:306`) or `lag > threshold`
(`:310`/`:316`) — that measured absence IS the RED, and it is stated here so a reader does
not mistake an execution failure for a broken test.

What this file exists to stop, in one sentence: assertion (b) of the mint preflight has NO
SUBJECT at HEAD — a healthy run emits nothing at all from the lag check, so no observer can
tell a live reading from a frozen 0, which is the exact discrimination R61 sent Phase P to
make (LAW-18: a lever under test logs its own fire-rate in-run).

The oracles, and the defect each one is the ONLY witness to:

- `..._emits_an_actor_lag_sample_on_a_healthy_poll` — the emission exists at all, and it
  carries a READING (learner/actor/lag/threshold), not an arming flag. Sole witness to
  "(b) has a subject". `heartbeat_watchdog_armed` (`:217-221`) carries `{armed,
  threshold_steps}` — the ARMING, never the reading — and V-2 recorded that mis-framing.
- `..._sample_and_the_fire_path_carry_THE_SAME_detail` — sole witness to the
  single-arithmetic-authority property (§7.2: "a sample that disagreed with the fire would
  be a second authority"). A reimplementation that recomputes the reading passes every
  other test in this file and fails only this one.
- `..._reads_the_callables_live_across_polls` — the LAW-07 mutation self-test. Sole witness
  against a captured-at-ctor reading; ordered-list equality, so a stream that merely
  *changes* does not satisfy it.
- `..._is_gated_by_the_file_interval_already_in_the_object` — sole witness to Remedy A's
  derivation. A private timer, a new module constant or a hardcoded interval all pass the
  three tests above and fail this one.
- `test_heartbeat_watchdog_init_gains_NO_new_parameter` — sole witness against MF-1's
  rejected shape. A required `lag_sample_interval_sec` kwarg turns 7 of the 9 tests in
  byte-frozen `tests/train/test_actor_lag_watchdog.py` (`5638b90db43866e6`) red, which is an
  R43 event outside every authorization this run holds (R67 lands AFTER Phase P). This
  oracle is the only thing in the repo that makes that shape un-reintroducible.
- `..._no_sample_during_close_out_and_none_without_a_spec` — sole witness that the emission
  inherits the existing structural gates rather than becoming a fourth one.
- `..._a_negative_lag_still_samples` — sole witness that the sample is not suppressed by the
  `lag < 0` arm; the preflight's b5a reads `lag_steps < 0` off the SAMPLE, not only off the
  `actor_lag_negative` event, and an emission placed after that arm would silence it.

R9 posture: every drive uses the REAL `HeartbeatWatchdog`, the REAL `ActorLagSpec` and the
REAL `JsonlEventSink` under `tmp_path`. No `*.jsonl` fixture is committed (R7 / gate 6,
DESIGN §9.3) — the streams are built by driving the real objects and read back off disk.

R8 >300 justify: ONE unit — the sample emission, the two structural gates it inherits (`_staleness_armed` and `spec is None`) and the constructor CENSUS that keeps its interval derived rather than parameterised are one negotiated remedy (TD-6 / MF-1, Remedy A). The census is only meaningful beside the rows it protects: read alone it looks like an arbitrary signature freeze, and read alone the emission rows give no reason the interval could not simply have been a new kwarg.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from mantis.monitor.sink import JsonlEventSink
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog

# The event name is the contract surface: `docs/contracts/event_manifest.md` gains a row for
# it in the SAME commit as this file (M-3 + C-5, gate 10's TOKEN_RE — DESIGN §10.4/SF-3).
SAMPLE_EVENT = "actor_lag_sample"

# The four keys `_check_actor_lag` already builds into `detail` (`heartbeat_watchdog.py:299
# -300`) and hands to the fire path. The sample must carry the same four — no more, no
# fewer arithmetic — because it is the SAME dict.
DETAIL_KEYS = ("learner_step", "actor_ckpt_step", "lag_steps", "threshold_steps")


def _registry():
    return SimpleNamespace(
        sources=("train_step",), ages=lambda: {"train_step": 0.0},
        beaten_sources=lambda: frozenset({"train_step"}), arm=lambda: None,
    )


def _watchdog(tmp_path: Path, *, spec, sink, file_interval_sec=0.0, clock=None, codes=None):
    """Staleness structurally silenced (`deadline <= 0` disables that source's fire), so
    every event observed below comes from the lag check itself. The watchdog THREAD is
    never started — `poll_once()` is driven directly, zero sleeps."""
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
    """Decode the REAL segment file the sink wrote and filter by event name.

    Reading back off disk rather than off a spy is deliberate: the mint preflight's whole
    observation transport is this file (§7.1), so an emission that never reaches it is
    invisible to assertion (b) however loudly it was 'emitted'.
    """
    lines = [ln for ln in sink.path.read_text().splitlines() if ln.strip()]
    return [e for e in (json.loads(ln) for ln in lines) if e.get("event") == name]


# ── the producer ──────────────────────────────────────────────────────────────────────
def test_a_healthy_poll_emits_an_actor_lag_sample_carrying_the_live_reading(tmp_path) -> None:
    """LAW-18 / TD-6. A HEALTHY poll — lag far under threshold, nothing firing — must put
    the reading on the wire. The values are rigged to distinct non-zero integers so that a
    stub emitting a constant (`lag_steps: 0`, or the threshold echoed back) fails."""
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
    """§7.2's load-bearing property: the sample IS the fire path's own `detail` dict, so a
    sample can never disagree with the reading that fires. Driven at a lag that is over
    threshold but DISARMED, so both events land on one poll from one `detail`."""
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
    """b5a reads `lag_steps < 0` off the SAMPLE as well as off `actor_lag_negative`
    (§7.4). An emission placed AFTER the `lag < 0` arm would silence the sample on exactly
    the wiring defect the preflight exists to catch, and every other test here would pass."""
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


# ── the LAW-07 mutation self-test ─────────────────────────────────────────────────────
def test_the_sample_reads_the_callables_live_across_polls(tmp_path) -> None:
    """The O-28 discipline as a MUTATION self-test: a reading captured at ctor/arm — or a
    constant — reproduces the first row forever. Ordered-list equality over three polls, so
    a stream that merely *moves* does not satisfy it either (Phase S's `assert lag > 0`
    trap, DESIGN §7.4)."""
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
    """MF-1 / Remedy A: the sample interval is DERIVED from `self._file_interval` — the
    value `subsystems.py:270` already passes as `file_interval_sec` — so one config fact
    never enters this ctor twice under two names (LAW-08). A private timer, a new module
    constant or a hardcoded seconds literal all fail here and nowhere else.

    Both arms are asserted, because only the pair pins a derivation: interval 10 must
    THIN the stream on the same clock that interval 0 must not.
    """
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
    """MF-1's closure, and the reason it is a signature CENSUS rather than a spot check.

    The rejected shape for TD-6 was a required `lag_sample_interval_sec` ctor kwarg. It was
    measured to turn 37 tests red across five files — 7 of the 9 in byte-frozen
    `tests/train/test_actor_lag_watchdog.py` (`5638b90db43866e6`), whose renegotiation R67
    schedules for AFTER Phase P. A census (not `assert "lag_sample_interval_sec" not in
    params`) is what makes ANY re-shaping of this constructor visible, including the
    module-constant-default variant §7.2 also rejects.

    `monitor_liveness` (AUDIT-1 F-11 / R334(b)) IS such a re-shaping and the census DID make
    it visible — which is the pin working, not the pin being routed around. It is admitted
    against the rule this census encodes, not despite it, and the grounds are measured rather
    than argued: TD-6's shape was rejected because a REQUIRED kwarg turned 37 tests red across
    five files, and this one is OPTIONAL (`()`) and turned exactly ONE test red — this census.
    It also does not touch TD-6's subject: the lag sample interval still derives from
    `file_interval_sec`, and so does the liveness sample, which is the same one rule with a
    third consumer rather than a second authority. Neither of the two files this docstring
    calls byte-frozen appears in ANY `wp/*/ORACLE_FREEZE*.sha256` row (grepped at the change,
    rc 1, zero hits), so no frozen-file grant is owed; the "byte-frozen" wording is a WP12-R
    working-tree convention and is left standing as the record of what it meant.
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
    """The emission inherits the two structural gates that already bound `_check_actor_lag`
    — `_staleness_armed` (`poll_once`, `:276-277`) and `spec is None` (`:294`) — instead of
    becoming a third, independently-wrong one. This is also what keeps the emission out of
    the close-out window that MF-5's b4c reasoning depends on."""
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
