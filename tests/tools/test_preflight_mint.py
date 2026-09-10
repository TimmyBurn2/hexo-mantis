"""The mint preflight's corpus + structural pins.

What this file exists to stop: run5 minting on a preflight that returned rc 0 while the sync
cadence, the lag transport or the arming was broken. Three predicates exist only because
executing the design's earlier versions found them satisfiable by a broken run.

"Exactly one predicate flips per mutation" is FALSE — a thinned stream flips a1 AND a2, and a
frozen actor is also a stale one — so the cross-check pins the EXACT declared flip-set per row
plus pairwise-distinct signatures, and pins predicate order so a report is deterministic. Every
stream's `ts` is re-based onto a modelled clock, which makes b4c's window HARDER, never easier.

>300 justify (R8): one gate, one corpus. Every row is a mutation of the SAME event stream against
the SAME predicate set, and the load-bearing assertion is the cross-check that each mutation
flips exactly its declared set — a property that exists only while all of them are collected in
one place. Splitting by predicate deletes the cross-check; splitting by mutation duplicates the
stream builders, which are the thing the mutations mutate.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path
from types import SimpleNamespace

from mantis.config.armed_aborts import MANIFEST, audit_arming  # RED anchor #2
from mantis.config.loader import load_config
from mantis.monitor.sink import JsonlEventSink
from mantis.train.actor_sync import ActorSync
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"

#: The four new test files. The sys.path census scans the tool AND its tests.
PHASE_P_TEST_FILES = (
    "tests/tools/test_preflight_mint.py",
    "tests/config/test_armed_abort_manifest.py",
    "tests/train/test_actor_lag_sample_emission.py",
    "tests/model/test_build_net_arch_handle.py",
)


def _load_tool():
    """Load the gate script by absolute path, with ZERO `sys.path` mutation: `tools/` is not a
    package."""
    if not TOOL_PATH.is_file():
        raise ModuleNotFoundError(
            f"RED anchor: {TOOL_PATH.relative_to(REPO_ROOT)} does not exist at HEAD — "
            "IMPL owes C-2 (DESIGN_P §13). This file is the mint preflight's producer "
            "suite; it cannot be green before the gate it produces evidence for exists."
        )
    spec = importlib.util.spec_from_file_location("preflight_mint", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()  # RED-at-import anchor #1

# The modelled run: every number below is a MEASURED repo fact, none invented.
_N = 101              # §5.5 — the minimum legal burst on all five minted configs
_C = 1                # train.actor_sync_cadence_steps, all five minted configs
_P = 5.0              # monitor.heartbeat_poll_interval_sec — configs/run6.yaml:198
_STEP_SEC = 0.5       # §7.4's modelled step duration (§14 item 17: the real ratio is unmeasured)
_SAMPLE_TS = (0.0, 15.0, 30.0, 45.0)   # heartbeat_file_interval_sec 15.0 — run6.yaml:199
_THRESHOLD = 100      # monitor.actor_lag_threshold_steps — run6.yaml:202

#: (learner_step, actor_ckpt_step) at each of the four sample instants. The third pair is the
#: one poll that lands INSIDE the sync window, which is what makes
#: `inversion_discrimination == "proven"` reachable at cadence 1 — and rc 0 requires `proven`.
_HEALTHY_READINGS = ((0, 0), (30, 30), (60, 59), (90, 90))
_FROZEN_ACTOR_READINGS = ((0, 0), (30, 0), (60, 0), (90, 0))
_STALE_MIRROR_READINGS = ((0, 0), (30, 0), (60, 20), (90, 50))

A_KEYS = ("a1", "a2", "a3", "a4")
B_KEYS = ("b0", "b1", "b2", "b3", "b4a", "b4b", "b4c", "b5a")
SAMPLE_EVENT = "actor_lag_sample"


class _SyncTarget:
    """Stand-in for the injected sync target, which `ActorSync.__init__` requires. Legitimate
    here: stand-ins are banned in the TOOL, not in a unit test."""

    def __init__(self) -> None:
        self.pushes: list = []
        self.steps: list[int] = []

    def sync_inference_weights(self, state_dict) -> None:
        self.pushes.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        self.steps.append(int(step))


def _read(sink: JsonlEventSink, name: str) -> list[dict]:
    lines = [ln for ln in sink.path.read_text().splitlines() if ln.strip()]
    return [event for event in (json.loads(ln) for ln in lines)
            if event.get("event") == name]


def _real_syncs(tmp_path: Path, tag: str, steps, *, cadence: int = _C) -> list[dict]:
    """Drive a REAL `ActorSync` at `cadence` over `steps` through a REAL `JsonlEventSink` and
    read the events back off disk, with `ts` re-based onto the modelled clock."""
    sink = JsonlEventSink(log_dir=tmp_path / f"sync_{tag}", run_id=f"oracle_p_{tag}")
    learner = {"v": 0}
    sync = ActorSync(target=_SyncTarget(), state_dict_fn=lambda: {},
                     step_fn=lambda: learner["v"], cadence_steps=cadence, sink=sink,
                     run_id=f"oracle_p_{tag}")
    for step in steps:
        learner["v"] = step
        sync.maybe_sync(step)
    events = _read(sink, "actor_sync")
    for event in events:
        event["ts"] = _STEP_SEC * float(event["step"])
    return events


def _real_samples(tmp_path: Path, tag: str, readings, *, swapped: bool = False) -> list[dict]:
    """Drive a REAL `HeartbeatWatchdog` with a REAL `ActorLagSpec` through a REAL sink, one poll
    per reading, and read the `actor_lag_sample` events back off disk. RED at HEAD by
    construction: the emission does not exist yet, so this returns `[]`."""
    sink = JsonlEventSink(log_dir=tmp_path / f"lag_{tag}", run_id=f"oracle_p_{tag}")
    state = {"learner": 0, "actor": 0}
    learner_fn, actor_fn = (lambda: state["learner"]), (lambda: state["actor"])
    if swapped:
        learner_fn, actor_fn = actor_fn, learner_fn
    watchdog = HeartbeatWatchdog(
        registry=SimpleNamespace(sources=("train_step",), ages=lambda: {"train_step": 0.0},
                                 beaten_sources=lambda: frozenset({"train_step"}),
                                 arm=lambda: None),
        deadlines={"train_step": 0.0}, sink=sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / f"hb_{tag}.json", file_interval_sec=0.0,
        poll_interval_sec=0.0, clock=lambda: 0.0, save_snapshot=lambda: None,
        exit_fn=lambda code: None, snapshot_timeout_sec=2.0, wired_sources=["train_step"],
        actor_lag=ActorLagSpec(learner_step_fn=learner_fn, actor_ckpt_step_fn=actor_fn,
                               threshold_steps=_THRESHOLD, abort_enabled=False),
    )
    for learner, actor in readings:
        state["learner"], state["actor"] = learner, actor
        watchdog.poll_once()
    events = _read(sink, SAMPLE_EVENT)
    for index, event in enumerate(events):
        event["ts"] = _SAMPLE_TS[index] if index < len(_SAMPLE_TS) else float(index)
    negatives = _read(sink, "actor_lag_negative")
    for event in negatives:
        event["ts"] = _SAMPLE_TS[0]
    return events + negatives


def _model_samples(readings) -> list[dict]:
    """The same stream, constructed rather than driven, and licensed by the bridge test — without
    it this is a second authority for the sample's shape."""
    return [{"event": "actor_lag_sample", "seq": index, "ts": _SAMPLE_TS[index],
             "learner_step": learner, "actor_ckpt_step": actor,
             "lag_steps": learner - actor, "threshold_steps": _THRESHOLD}
            for index, (learner, actor) in enumerate(readings)]


def _stream(syncs: list[dict], samples: list[dict], *, final_step: int = _N) -> list[dict]:
    """One JSONL segment, in emission order. `shutdown_save` is the step ground truth: a
    101-step burst emits no per-step event at `log_interval=1000`."""
    save = [{"event": "shutdown_save", "step": final_step,
             "ts": _STEP_SEC * float(final_step) + 0.5}]
    return sorted(syncs + samples + save, key=lambda event: float(event["ts"]))


def _assertions(events: list[dict], *, cadence: int = _C, burst: int = _N):
    """THE seam. Return shape is the design's `assertions` object."""
    return TOOL.evaluate_assertions(events, cadence_steps=cadence, burst_steps=burst,
                                    poll_interval_sec=_P)


def _vector(block, keys) -> tuple[frozenset, frozenset]:
    """Return (affirmatively-False predicates, not-evaluated predicates). Three-valued on purpose:
    `b0` and the zero-sync arm GATE the predicates behind them, and collapsing "did not hold" into
    "was not evaluated" stops a cross-check discriminating."""
    missing = [key for key in keys if key not in block]
    assert not missing, f"the report block must carry every predicate key; missing {missing}"
    return (frozenset(k for k in keys if block[k] is False),
            frozenset(k for k in keys if block[k] is None))


# The corpus: one healthy stream, mutated in exactly one place per row.
def _row_healthy(tmp_path):
    return {"events": _stream(_real_syncs(tmp_path, "healthy", range(1, _N + 1)),
                              _real_samples(tmp_path, "healthy", _HEALTHY_READINGS))}


def _row_m2(tmp_path):
    """The `actor_lag_sample` emission removed: the samples are driven for real and then
    FILTERED OUT, so the row is deterministic both before and after the emission lands."""
    return {"events": _stream(_real_syncs(tmp_path, "m2", range(1, _N + 1)), [])}


def _row_m3(tmp_path):
    return {"events": _stream(_real_syncs(tmp_path, "m3", range(1, _N + 1)),
                              _real_samples(tmp_path, "m3", _FROZEN_ACTOR_READINGS))}


def _row_m4(tmp_path):
    return {"events": _stream(
        _real_syncs(tmp_path, "m4", range(1, _N + 1)),
        _real_samples(tmp_path, "m4", ((0, 0), (30, 29), (60, 59), (90, 89)), swapped=True))}


def _row_m5(tmp_path):
    samples = _real_samples(tmp_path, "m5", _HEALTHY_READINGS)
    for sample in samples:
        if sample.get("learner_step") == 60:
            sample["lag_steps"] = 0          # inconsistent with its OWN operands (60 − 59)
    return {"events": _stream(_real_syncs(tmp_path, "m5", range(1, _N + 1)), samples)}


def _row_m6(tmp_path):
    samples = _real_samples(tmp_path, "m6", _HEALTHY_READINGS)
    for sample in samples:
        if sample.get("learner_step") == 90:  # moves, self-consistent, NEVER a sync step
            sample.update(learner_step=150, actor_ckpt_step=150, lag_steps=0)
    return {"events": _stream(_real_syncs(tmp_path, "m6", range(1, _N + 1)), samples)}


def _row_m7(tmp_path):
    steps = [step for step in range(1, _N + 1) if step != 50]
    return {"events": _stream(_real_syncs(tmp_path, "m7", steps),
                              _real_samples(tmp_path, "m7", _HEALTHY_READINGS))}


def _row_m8(tmp_path):
    """Zero syncs, with the physically consistent lag stream: an actor never synced never
    advances."""
    return {"events": _stream(_real_syncs(tmp_path, "m8", []),
                              _real_samples(tmp_path, "m8", _FROZEN_ACTOR_READINGS))}


def _row_m14(tmp_path):
    return {"events": _stream(_real_syncs(tmp_path, "m14", [1, *range(1, _N + 1)]),
                              _real_samples(tmp_path, "m14", _HEALTHY_READINGS))}


def _row_m16(tmp_path):
    return {"events": _stream(_real_syncs(tmp_path, "m16", range(1, _N + 1)),
                              _model_samples(_STALE_MIRROR_READINGS))}


def _row_m1(tmp_path):
    row = _row_healthy(tmp_path)
    row["config"] = "dev_example.yaml"       # real committed file, arming false at :200
    return row


def _row_m10(tmp_path):
    row = _row_healthy(tmp_path)
    root = tmp_path / "tampered_root"
    for entry in MANIFEST:
        if entry.source_pin is None:
            continue
        rel, text = entry.source_pin
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text((REPO_ROOT / rel).read_text().replace(text, "# deleted\n"))
    row["pin_root"] = root
    return row


#: name -> (builder, a_failure, a_false, a_none, a_sub, b_failure, b_false, b_none, b_sub,
#:          c_fails, manifest_fails).
_CORPUS = {
    "M2": (_row_m2, None, (), (), None,
           "PreflightLagUnobservableError", ("b0",), B_KEYS[1:], None, False, False),
    "M3": (_row_m3, None, (), (), None,
           "PreflightLagFrozenError", ("b3", "b4c"), (), "actor", False, False),
    "M4": (_row_m4, None, (), (), None,
           "PreflightLagInvertedError", ("b5a",), (), None, False, False),
    "M5": (_row_m5, None, (), (), None,
           "PreflightLagArithmeticError", ("b1",), (), None, False, False),
    "M6": (_row_m6, None, (), (), None,
           "PreflightLagSourceMismatchError", ("b4a",), (), "foreign", False, False),
    "M7": (_row_m7, "PreflightSyncCadenceError", ("a1", "a2"), (), "missed",
           None, (), (), None, False, False),
    "M8": (_row_m8, "PreflightSyncAbsentError", (), A_KEYS, None,
           "PreflightLagFrozenError", ("b3",), (), "actor", False, False),
    "M14": (_row_m14, "PreflightSyncCadenceError", ("a1", "a2"), (), "extra",
            None, (), (), None, False, False),
    "M16": (_row_m16, None, (), (), None,
            "PreflightLagSourceMismatchError", ("b4c",), (), "stale", False, False),
    "M1": (_row_m1, None, (), (), None, None, (), (), None, True, False),
    "M10": (_row_m10, None, (), (), None, None, (), (), None, False, True),
}


def _observe(row):
    """Evaluate all four assertion axes over one corpus row."""
    blocks = _assertions(row["events"])
    config = load_config(REPO_ROOT / "configs" / row.get("config", "run6.yaml"))
    return {
        "a": blocks["a_sync"],
        "b": blocks["b_lag"],
        "c_fails": bool(audit_arming(config).disarmed),
        "manifest_fails": bool(TOOL.verify_source_pins(
            MANIFEST, repo_root=row.get("pin_root", REPO_ROOT))),
    }


def _signature(observed) -> tuple:
    a_false, a_none = _vector(observed["a"], A_KEYS)
    b_false, b_none = _vector(observed["b"], B_KEYS)
    return (observed["a"].get("failure"), tuple(sorted(a_false)), tuple(sorted(a_none)),
            observed["a"].get("sub_reason"),
            observed["b"].get("failure"), tuple(sorted(b_false)), tuple(sorted(b_none)),
            observed["b"].get("sub_reason"),
            observed["c_fails"], observed["manifest_fails"])


def _declared(name) -> tuple:
    (_build, a_failure, a_false, a_none, a_sub,
     b_failure, b_false, b_none, b_sub, c_fails, manifest_fails) = _CORPUS[name]
    return (a_failure, tuple(sorted(a_false)), tuple(sorted(a_none)), a_sub,
            b_failure, tuple(sorted(b_false)), tuple(sorted(b_none)), b_sub,
            c_fails, manifest_fails)


# The healthy inverse — a gate that cannot go green is as useless as one that cannot go red.
def test_a_healthy_stream_passes_every_predicate(tmp_path) -> None:
    """Real `ActorSync` over 101 steps at cadence 1 + real watchdog samples: nothing flips."""
    observed = _observe(_row_healthy(tmp_path))
    assert observed["a"]["verdict"] == "pass", f"a_sync must be green: {observed['a']!r}"
    assert observed["b"]["verdict"] == "pass", f"b_lag must be green: {observed['b']!r}"
    assert all(observed["a"][key] is True for key in A_KEYS)
    assert all(observed["b"][key] is True for key in B_KEYS)
    assert observed["a"]["expected_syncs"] == observed["a"]["observed_syncs"] == _N
    assert observed["a"]["step_ground_truth"] == {"source": "shutdown_save", "value": _N}, (
        "§7.3: N comes from an independent witness, and the report must NAME which one "
        f"spoke; got {observed['a'].get('step_ground_truth')!r}"
    )
    assert observed["b"]["inversion_discrimination"] == "proven", (
        "MF-3 / ADJ-12: rc 0 in mode PREFLIGHT REQUIRES `proven`. This stream carries one "
        "sample inside the sync window (learner 60 / actor 59), so the axis IS "
        f"discriminated; got {observed['b'].get('inversion_discrimination')!r}"
    )
    assert observed["b"]["discriminating_samples"] == 1
    assert observed["a"]["a4_pins"] == "single-producer / no sink line loss", (
        "MF-4: the report must say what `sync_count` actually pins, so no reader banks it "
        "as cadence coverage"
    )
    assert observed["b"]["b1_scope"] == (
        "source-mutation detector; vacuous against an unmodified watchdog"
    ), "§7.4: b1's honest scope is STATED in the report, not implied"


def test_the_inversion_axis_is_proven_only_by_a_discriminating_sample(tmp_path) -> None:
    """With the lag operands exchanged, a `C == 1` config produces no negative lag and no
    discriminating sample. Both arms are asserted, because only the pair shows the axis is a
    real COUNT."""
    equal_only = _stream(_real_syncs(tmp_path, "inv_a", range(1, _N + 1)),
                         _model_samples(((0, 0), (30, 30), (60, 60), (90, 90))))
    block = _assertions(equal_only)["b_lag"]
    assert all(block[key] is True for key in B_KEYS), (
        "every transport predicate holds on this stream — which is exactly why the "
        f"inversion axis needs its own non-green outcome; got {block!r}"
    )
    assert block["discriminating_samples"] == 0
    assert block["inversion_discrimination"] == "unproven"
    assert block["verdict"] != "pass", (
        "MF-3 / ADJ-12: `unproven` at cadence 1 must NOT be a green. A SWAPPED-OPERAND "
        "wiring produces this same stream, and rc 0 here is the precise failure R61 exists "
        f"to prevent; got verdict {block['verdict']!r}"
    )
    assert block["failure"] == "PreflightInversionUndiscriminatedError", (
        f"the named non-green outcome is rc 23; got {block.get('failure')!r}"
    )

    discriminating = _stream(_real_syncs(tmp_path, "inv_b", range(1, _N + 1)),
                             _model_samples(_HEALTHY_READINGS))
    proven = _assertions(discriminating)["b_lag"]
    assert (proven["inversion_discrimination"], proven["discriminating_samples"]) == \
           ("proven", 1), (
        "one sample with learner_step != actor_ckpt_step proves the axis — the count must "
        f"be a real count, not a constant; got {proven!r}"
    )


def test_at_cadence_above_one_an_undiscriminated_axis_is_a_FROZEN_reading(tmp_path) -> None:
    """At cadence > 1 the learner is STRUCTURALLY ahead between syncs, so a reading that never
    shows it is frozen (rc 26), not merely undiscriminated (rc 23)."""
    stream = _stream(_real_syncs(tmp_path, "cadence5", range(1, _N + 1)),
                     _model_samples(((0, 0), (30, 30), (60, 60), (90, 90))))
    block = TOOL.evaluate_assertions(stream, cadence_steps=5, burst_steps=_N,
                                     poll_interval_sec=_P)["b_lag"]
    assert block["failure"] == "PreflightLagFrozenError", (
        f"cadence > 1 + zero discriminating samples must name the FROZEN error, not the "
        f"undiscriminated one; got {block.get('failure')!r}"
    )
    assert block.get("sub_reason") == "both", (
        f"§6.3 rc 26 carries side ∈ {{learner, actor, both}}; got {block.get('sub_reason')!r}"
    )


def test_the_modelled_sample_stream_is_what_the_REAL_watchdog_emits(tmp_path) -> None:
    """The bridge that licenses `_model_samples`; without it every predicate-level row tests a
    fiction. Compared on payload, not on the re-based `ts`."""
    real = _real_samples(tmp_path, "bridge", _HEALTHY_READINGS)
    model = _model_samples(_HEALTHY_READINGS)
    def strip(events):
        return [{k: v for k, v in event.items() if k != "ts"} for event in events]

    assert strip(real) == strip(model), (
        "the modelled sample stream must be byte-equal to what a real HeartbeatWatchdog "
        f"emits for the same readings.\nreal  = {strip(real)}\nmodel = {strip(model)}"
    )


# Assertion (a) — sync presence and cadence-consistency.
def test_a_missed_cadence_boundary_fails_the_sync_check(tmp_path) -> None:
    """A real `ActorSync` with one boundary skipped fails a1/a2 with sub-reason `missed`."""
    block = _observe(_row_m7(tmp_path))["a"]
    assert (block["a1"], block["a2"]) == (False, False)
    assert (block["a3"], block["a4"]) == (True, True), (
        "a3 and a4 must SURVIVE a thinned stream — a4 stays True because the run really "
        "did miss a sync (as opposed to the sink losing a line), and that discrimination "
        f"is a4's whole value (MF-4); got a3={block['a3']} a4={block['a4']}"
    )
    assert block["sub_reason"] == "missed" and block["failure"] == "PreflightSyncCadenceError"
    assert block["observed_syncs"] == _N - 1 and block["expected_syncs"] == _N


def test_an_over_firing_sync_stream_fails_by_name(tmp_path) -> None:
    """Step 1 driven twice: the over-firing stream the design's earlier SET-equality predicate
    PASSED, re-produced by execution at 102 events."""
    block = _observe(_row_m14(tmp_path))["a"]
    assert block["observed_syncs"] == _N + 1, (
        "the stream must really carry a duplicate — otherwise this oracle is testing "
        f"nothing; got {block['observed_syncs']} events"
    )
    assert (block["a1"], block["a2"]) == (False, False), (
        "an over-firing stream must fail. A SET comparison collapses the duplicate and "
        "returns PASS; the predicate must be ORDERED-LIST equality with a length conjunct"
    )
    assert block["sub_reason"] == "extra", (
        f"a1 exists separately from a2 only to carry this sub-reason; got "
        f"{block.get('sub_reason')!r}"
    )
    assert block["a4"] is True, (
        "`sync_count` is contiguous for ANY sequence one ActorSync produces — including "
        "this one. Pinning it True here is what stops a reader banking a4 as cadence cover"
    )


def test_zero_syncs_fails_by_name(tmp_path) -> None:
    """Zero syncs is named separately from a cadence mismatch: "never synced" and "synced at
    the wrong steps" are different diagnoses and get different codes (rc 20 vs rc 21)."""
    block = _observe(_row_m8(tmp_path))["a"]
    assert block["failure"] == "PreflightSyncAbsentError", (
        f"zero actor_sync events is its own named outcome; got {block.get('failure')!r}"
    )
    assert block["observed_syncs"] == 0
    assert all(block[key] is None for key in A_KEYS), (
        "with no syncs at all the four sub-predicates were not evaluated — reporting them "
        f"as False would claim a cadence measurement nobody made; got {block!r}"
    )


def test_a_burst_shorter_than_one_cadence_still_expects_the_unconditional_first_sync(
    tmp_path,
) -> None:
    """A burst too short to produce a sync still produces one: `ActorSync` syncs unconditionally
    on the FIRST call. Pinned so a future `expected` that drops the `{1} u` is caught."""
    syncs = _real_syncs(tmp_path, "shortburst", [1, 2, 3], cadence=7)
    assert [event["step"] for event in syncs] == [1], (
        "at cadence 7 over steps 1..3 a real ActorSync syncs exactly once, on the first "
        f"call; got {[event['step'] for event in syncs]}"
    )
    block = TOOL.evaluate_assertions(_stream(syncs, [], final_step=3), cadence_steps=7,
                                     burst_steps=3, poll_interval_sec=_P)["a_sync"]
    assert block["expected_syncs"] == 1 and block["a1"] is True and block["a2"] is True, (
        f"expected == {{1}} when N < C; got {block!r}"
    )


# Assertion (b) — the lag transport.
def test_a_watchdog_that_never_samples_makes_the_lag_unobservable(tmp_path) -> None:
    """The state of the tree at HEAD: `_check_actor_lag` emits only on `lag < 0` or
    `lag > threshold`, so a healthy run publishes NOTHING about the reading."""
    block = _observe(_row_m2(tmp_path))["b"]
    assert block["b0"] is False and block["failure"] == "PreflightLagUnobservableError"
    assert block["samples"] == 0
    assert all(block[key] is None for key in B_KEYS[1:]), (
        "b0 GATES the rest: with fewer than two samples, b1…b5a were not evaluated, and "
        f"reporting them True would be a green over an absent measurement; got {block!r}"
    )


def test_a_frozen_actor_callable_fails_the_live_lag_check(tmp_path) -> None:
    """`actor_ckpt_step_fn = lambda: 0`, driven through a REAL watchdog."""
    block = _observe(_row_m3(tmp_path))["b"]
    assert block["b3"] is False, f"a constant actor source must fail b3; got {block!r}"
    assert block["b2"] is True, "the learner side is healthy here — only the actor froze"
    assert block["failure"] == "PreflightLagFrozenError" and block["sub_reason"] == "actor"


def test_swapped_lag_callables_emit_actor_lag_negative(tmp_path) -> None:
    """The only DETERMINISTIC closure of the inversion axis in this phase, driven through a real
    watchdog whose two lag callables are exchanged rather than a synthetic negative stream: the
    thing under test is that the exchange is OBSERVABLE at all."""
    row = _row_m4(tmp_path)
    negatives = [e for e in row["events"] if e.get("event") == "actor_lag_negative"]
    assert len(negatives) == 1, (
        "the real watchdog must report the inverted wiring loudly, exactly once per episode "
        f"(`heartbeat_watchdog.py:306`); got {len(negatives)}"
    )
    block = _observe(row)["b"]
    assert block["b5a"] is False and block["failure"] == "PreflightLagInvertedError"
    assert all(block[key] is True for key in ("b1", "b2", "b3", "b4a", "b4b", "b4c")), (
        "an operand swap leaves every OTHER predicate satisfied — which is why b5a has to "
        f"exist as its own arm; got {block!r}"
    )


def test_a_hardcoded_lag_value_fails_the_arithmetic_check(tmp_path) -> None:
    """b1's only producer. b1 is a source-mutation detector, vacuous against an unmodified
    watchdog, and the report SAYS so."""
    block = _observe(_row_m5(tmp_path))["b"]
    assert block["b1"] is False and block["failure"] == "PreflightLagArithmeticError"
    assert block["b5a"] is True, (
        "the planted constant is 0, which is non-negative — so b5a cannot catch it and b1 "
        "is genuinely the only witness"
    )


def test_a_moving_but_foreign_actor_source_is_rejected(tmp_path) -> None:
    """The membership half of b4: the planted value MOVES, is self-consistent, is
    non-decreasing and is never late, so only membership catches it."""
    block = _observe(_row_m6(tmp_path))["b"]
    assert block["b4a"] is False and block["sub_reason"] == "foreign"
    assert block["failure"] == "PreflightLagSourceMismatchError"
    assert (block["b2"], block["b3"], block["b4b"], block["b4c"], block["b1"]) == \
           (True, True, True, True, True), (
        f"every other predicate must hold, or this row is not testing membership: {block!r}"
    )


def test_a_healthy_run_whose_last_sample_predates_the_final_syncs_is_not_a_source_mismatch(
    tmp_path,
) -> None:
    """The earlier b4 required `max(l.actor_ckpt_step) == max(s.actor_ckpt_step)`, and on a
    healthy run sampling stops at close-out while syncs happen every step, so rc 28 fired on a
    healthy run. The first assertion proves this stream really is inside the hazard window."""
    row = _row_healthy(tmp_path)
    samples = [e for e in row["events"] if e.get("event") == "actor_lag_sample"]
    syncs = [e for e in row["events"] if e.get("event") == "actor_sync"]
    assert max(s["actor_ckpt_step"] for s in samples) < max(s["step"] for s in syncs), (
        "this stream must exhibit the MF-5 hazard (last reading strictly behind the last "
        "sync) or the oracle is vacuous"
    )
    block = _assertions(row["events"])["b_lag"]
    assert (block["b4a"], block["b4b"], block["b4c"]) == (True, True, True), (
        f"a healthy tail is NOT a source mismatch; got {block!r}"
    )
    assert block["verdict"] == "pass"


def test_a_stale_but_legitimate_actor_mirror_is_rejected(tmp_path) -> None:
    """A stale-but-legitimate mirror passes b1/b2/b3/b4a/b4b/b5a, so only the `ts`-bounded
    staleness conjunct kills it — the property b4 existed for."""
    block = _observe(_row_m16(tmp_path))["b"]
    assert block["b4c"] is False and block["sub_reason"] == "stale"
    assert block["failure"] == "PreflightLagSourceMismatchError"
    assert (block["b4a"], block["b4b"]) == (True, True), (
        "a stale MIRROR reports only real sync values in order — that is what makes it "
        f"invisible to everything except b4c; got {block!r}"
    )


def test_a_regressing_actor_source_is_rejected(tmp_path) -> None:
    """A regressed mirror (a re-read of a rotated file, a reset mirror) is neither foreign nor
    merely late, so it needs its own arm."""
    stream = _stream(_real_syncs(tmp_path, "regress", range(1, _N + 1)),
                     _model_samples(((0, 0), (30, 30), (60, 20), (90, 90))))
    block = _assertions(stream)["b_lag"]
    assert block["b4b"] is False and block["sub_reason"] == "regressed"
    assert block["b4a"] is True, "every value is still a legitimate sync step"


# The CLI contract.
def _run_tool(*args, timeout: int = 300):
    return subprocess.run([sys.executable, str(TOOL_PATH), *args], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=timeout)


def test_an_out_dir_inside_the_repo_is_refused(tmp_path) -> None:
    """An `--out-dir` inside the repo is REFUSED before anything is created: the child writes
    `*.jsonl` and gate 6 rejects those, so the gate would manufacture its own violation."""
    inside = REPO_ROOT / "_preflight_oracle_outdir"
    assert not inside.exists(), "the oracle's probe path must not pre-exist"
    # When the guard under test FAILS, the tool creates this path inside the repo; the finally
    # removes what the failure created so one red assertion does not also litter the tree.
    try:
        result = _run_tool("--config", "configs/run6.yaml", "--burst-steps", str(_N),
                           "--out-dir", str(inside), "--timeout-sec", "60")
        assert result.returncode == 13, (
            "§6.3 rc 13 PreflightOutDirInsideRepoError; got "
            f"{result.returncode}\nstdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
        )
        assert not inside.exists(), (
            "the refusal must precede creation — an out-dir the tool made and then rejected is "
            "still an untracked artifact directory inside the tree (R7)"
        )
        assert "PreflightOutDirInsideRepoError" in (result.stdout + result.stderr)
    finally:
        if inside.is_dir() and not inside.is_symlink():
            shutil.rmtree(inside)


def test_a_burst_below_the_lag_threshold_is_refused_by_name(tmp_path) -> None:
    """The minimum legal burst is 101 on every minted config, and the refusal must TEACH that —
    quote the binding validator, state the minimum — not merely reject."""
    result = _run_tool("--config", "configs/run6.yaml", "--burst-steps", "50",
                       "--out-dir", str(tmp_path), "--timeout-sec", "60")
    output = result.stdout + result.stderr
    assert result.returncode == 11, (
        f"§6.3 rc 11 PreflightBurstTooShortError; got {result.returncode}\n{output[-2000:]}"
    )
    assert "PreflightBurstTooShortError" in output
    assert "101" in output, "the message must state the MINIMUM, not just that 50 is wrong"
    assert "actor_lag_threshold_steps" in output, (
        "the message must quote the BINDING validator's own subject, so the operator knows "
        "which cross-field rule to read"
    )


def test_the_preflight_args_carry_no_defaults_and_are_enforced_per_mode(tmp_path) -> None:
    """Requiredness is pinned BEHAVIOURALLY, per mode: argparse cannot express "required in mode
    PREFLIGHT only" and the gate-12 step invokes `--audit-only` alone."""
    full = {"--config": "configs/run6.yaml", "--burst-steps": str(_N),
            "--out-dir": str(tmp_path), "--timeout-sec": "60"}
    for omitted in full:
        argv = [token for key, value in full.items() if key != omitted
                for token in (key, value)]
        result = _run_tool(*argv)
        assert result.returncode == 2, (
            f"omitting {omitted} must be a USAGE error (§6.3 rc 2) — no code-side default "
            f"may stand in for it (R1); got rc {result.returncode}\n"
            f"{(result.stdout + result.stderr)[-1500:]}"
        )


def test_audit_only_is_green_on_the_real_tree() -> None:
    """The gate-12 invocation verbatim: rc 0, AND it says out loud that rc 0 covers assertion
    (c) alone — a CI log reading `gate 12 ... exit 0` is what a later reader cites."""
    result = _run_tool("--audit-only")
    assert result.returncode == 0, (
        "configs/run6.yaml arms the one required row (the R59 flip at :203), so mode AUDIT "
        f"is green TODAY; got rc {result.returncode}\n"
        f"{(result.stdout + result.stderr)[-3000:]}"
    )
    for needle in ("mode=AUDIT", "NOT RUN", "rc 0 covers assertion (c) ONLY"):
        assert needle in result.stdout, (
            f"the mandatory AUDIT stdout line (§6.3b) must carry {needle!r}; got "
            f"stdout={result.stdout!r}"
        )
    # The shipped manifest holds ZERO deferred rows, so gate-12 stdout can no longer carry one;
    # what must still hold is that the loud-debt MECHANISM works, driven here on a SYNTHETIC row
    # through the `manifest=` keyword.
    import contextlib
    import io

    from mantis.config.armed_aborts import ArmedAbort, Mechanism, Status

    spec = importlib.util.spec_from_file_location("_pfm_deferred_probe", TOOL_PATH)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    probe = ArmedAbort(
        name="_synthetic_deferred_probe", config_path="train.does_not_exist",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.DEFERRED, exit_code=None,
        owner="CARD-COORD-KNOBS (R78)", source_pin=("src/mantis/run.py", "def compose_run"),
        note="synthetic subject for R56's loud-debt mechanism; not a shipped row.",
    )
    loud = io.StringIO()
    with contextlib.redirect_stdout(loud):
        tool._print_deferred_rows(manifest=(probe,))
    for needle in ("DEFERRED", probe.name, probe.owner, probe.config_path, probe.note):
        assert needle in loud.getvalue(), (
            "R56's loud print on EVERY run that carries debt — registered debt that stops "
            f"being visible stops being debt; missing {needle!r} from {loud.getvalue()!r}"
        )
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        tool._print_deferred_rows(manifest=())
    assert quiet.getvalue() == "", (
        "the empty arm (R72 — `if not deferred: return`): with no deferred row the loud "
        "print says NOTHING, which is exactly why the shipped-manifest assertion this "
        f"replaces had to move; got {quiet.getvalue()!r}"
    )


def test_an_audit_report_can_never_read_as_a_green_for_the_dynamic_assertions(
    tmp_path,
) -> None:
    """Behavioural, not AST: drives the tool in AUDIT mode and reads the JSON it actually
    emitted."""
    result = _run_tool("--audit-only", "--out-dir", str(tmp_path))
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    reports = sorted(tmp_path.glob("preflight_*.json"))
    assert len(reports) == 1, (
        f"the evidence report is written ALWAYS, in a finally (§9.1); found {reports}"
    )
    report = json.loads(reports[0].read_text())
    assert report["schema"] == "preflight-mint-v1" and report["mode"] == "audit"
    assert report["child"] is None, "mode AUDIT spawns no child (§9.1)"
    for name in ("a_sync", "b_lag"):
        block = report["assertions"][name]
        assert block["verdict"] == "not_run", (
            f"{name} must be PRESENT and explicitly not_run — omitting the key is the "
            f"absent-key hazard MF-7 names; got {block!r}"
        )
        assert block["verdict"] != "pass"
        assert block.get("reason"), f"{name}'s not_run verdict must carry a reason"
    assert report["assertions"]["c_arming"]["verdict"] == "pass"
    assert report["manifest"]["source_pins_ok"] is True


# Structural pins on the TOOL.
def _code_text(path: Path) -> str:
    """Return source with COMMENT / STRING / f-string tokens removed: a raw-text census would
    flag the tool's own prose, teaching people to word comments around a gate."""
    # FSTRING_MIDDLE is 3.12+; on the 3.11 floor f-strings lex as STRING.
    skip = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        return "\n".join(
            tok.string for tok in tokenize.tokenize(handle.readline) if tok.type not in skip
        )


TOOL_SOURCE = TOOL_PATH.read_text()
TOOL_TREE = ast.parse(TOOL_SOURCE)
TOOL_CODE = _code_text(TOOL_PATH)


def _add_argument_calls():
    return [node for node in ast.walk(TOOL_TREE)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"]


def test_the_parser_declares_no_defaults() -> None:
    """A `default=` in the parser is a code-side default authority for a run input, which is
    the defect this repo is arranged against."""
    calls = _add_argument_calls()
    # `--device` DIED on both callers, so the surface is FIVE documented inputs plus the
    # suppressed `--_boot`. Pinned rather than floored: `>= 6` kept passing NUMERICALLY across
    # the removal. The successor is the device-flag BAN over BOTH parsers.
    assert len(calls) == 6, (
        f"the tool declares exactly the five documented CLI inputs plus the suppressed "
        f"--_boot; found {len(calls)}"
    )
    for call in calls:
        names = [arg.value for arg in call.args if isinstance(arg, ast.Constant)]
        defaults = [kw for kw in call.keywords if kw.arg == "default"]
        assert not defaults, (
            f"add_argument{names} passes default= — R1: no code-side defaults, a default "
            "lives only in the schema field"
        )
    declared = {arg.value for call in calls for arg in call.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    for option in ("--config", "--burst-steps", "--out-dir", "--timeout-sec",
                   "--audit-only"):
        assert option in declared, f"the tool must declare {option}; got {sorted(declared)}"


def test_the_tool_contains_no_stand_in_for_a_production_object() -> None:
    """The tool contains no stand-in for a production object: a missing collaborator method is
    not supplied, and the AttributeError reaches it as `PreflightTreeDefectError`."""
    for token in ("monkeypatch", "unittest.mock", "SimpleNamespace", "MagicMock",
                  "mock.patch", "setattr(", "pytest"):
        assert token not in TOOL_CODE, (
            f"the preflight must contain no {token!r}: R61's whole point is that it varies "
            "the two axes no test can, and R64 forbids designing around a wall in the tool"
        )


def test_no_sys_path_mutation_in_the_tool_or_its_tests() -> None:
    """Zero `sys.path` writes. The tool re-execs ITSELF as the boot child and the child
    inherits the venv, which is the one place the ban is tempting."""
    for rel in (str(TOOL_PATH.relative_to(REPO_ROOT)), *PHASE_P_TEST_FILES):
        code = _code_text(REPO_ROOT / rel)
        for token in ("sys.path.append", "sys.path.insert", "sys.path.extend",
                      "sys.path=", "sys.path+=", "PYTHONPATH"):
            assert token not in code.replace(" ", ""), (
                f"{rel} mutates sys.path ({token!r}) — R5 / LAW-17 admit ZERO exceptions; "
                "load scripts by absolute path with importlib.util.spec_from_file_location"
            )


def test_the_override_map_carries_exactly_one_key() -> None:
    """`stop_step` keeps exactly ONE source; a second entry in the override tuple would make the
    preflight a second run-length authority. `override.keys` is emitted from this same
    constant."""
    assert tuple(TOOL.OVERRIDE_KEYS) == ("train.max_train_steps",), (
        "the burst override writes exactly one dotted key and reads nothing; got "
        f"{TOOL.OVERRIDE_KEYS!r}"
    )


def test_the_tool_never_constructs_a_config_by_a_validator_skipping_route() -> None:
    """The override's route is `dump -> mutate -> model_validate`, which IS the validator, never
    a construction that SKIPS the cross-field validators."""
    for token in ("model_copy", "model_construct"):
        assert token not in TOOL_CODE, (
            f"{token} skips every @model_validator — the preflight must construct configs "
            "only via load_config / RunConfig.model_validate"
        )
    assert "load_config" in TOOL_CODE, "the tool must use the ONE loader"
    assert "model_validate" in TOOL_CODE, "…and the loader's own final step for the override"


def test_compose_run_is_driven_with_the_four_real_collaborators() -> None:
    """Exactly one `compose_run` call site with the six explicit kwargs, and nothing the
    composition root builds for itself is passed in or patched. The builder tokens are BANNED
    here rather than required, a CI gate owning the only real collaborator build having been the
    one-authority violation."""
    calls = [node for node in ast.walk(TOOL_TREE) if isinstance(node, ast.Call)
             and getattr(node.func, "id", getattr(node.func, "attr", None)) == "compose_run"]
    assert len(calls) == 1, f"exactly one compose_run call site; found {len(calls)}"
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    for name in ("config", "trainer", "pool", "buffer", "log_dir", "checkpoint_dir"):
        assert name in keywords, (
            f"compose_run must be called with {name}= explicitly; got {sorted(keywords)}"
        )
    for name in ("trainer", "pool", "buffer"):
        assert isinstance(keywords[name], ast.Attribute), (
            f"{name}= must be the collaborator the ONE builder made, read off its result; "
            f"got {ast.dump(keywords[name])[:120]}"
        )
    for token in ("init_trainer", "WorkerPool", "HexgBuffer", "ReplayBuffer",
                  "build_run_safety(", "StepCoordinatorConfig("):
        assert token not in TOOL_CODE, (
            f"{token} is the composition root's to build (§4.1 / D-1); a CI gate that builds "
            "its own collaborators preflights a run nobody will launch"
        )
    assert "build_run_collaborators" in TOOL_CODE, (
        "…and the child must reach them through the ONE builder at `mantis.run`"
    )


def test_eval_enabled_is_the_literal_True_and_is_not_derived() -> None:
    """The tool declares NO eval posture at all, which strictly dominates the old "literal True"
    pin: the parameter is DELETED, so the child passes nothing and CAN pass nothing. The
    no-local-assignment ban survives, because a local `eval_enabled = ...` was how the escape
    came back quietly."""
    calls = [node for node in ast.walk(TOOL_TREE) if isinstance(node, ast.Call)
             and getattr(node.func, "id", getattr(node.func, "attr", None)) == "compose_run"]
    keywords = {kw.arg for kw in calls[0].keywords}
    assert "eval_enabled" not in keywords, (
        "the child must pass NO eval posture — the CONFIG governs it (R120/R64); got "
        f"{sorted(keywords)}"
    )
    assigned = [target.id for node in ast.walk(TOOL_TREE) if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)]
    assert "eval_enabled" not in assigned, (
        "a local `eval_enabled = …` is how the banned escape gets reintroduced quietly"
    )
    declared = {arg.value for call in _add_argument_calls() for arg in call.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    assert not any("eval" in option for option in declared), (
        f"no CLI switch may reach eval_enabled; got {sorted(declared)}"
    )


# The cross-check: "each mutation dies alone", measured.
def test_each_mutation_flips_exactly_its_declared_predicates(tmp_path) -> None:
    """Every mutation flips EXACTLY its declared predicate set, and the healthy stream flips
    nothing: no cross-talk, no silent death, the not-evaluated set declared so a predicate that
    stops being computed is not read as a pass, signatures UNIQUE, and the healthy row catching a
    predicate gone unconditionally red."""
    healthy = _signature(_observe(_row_healthy(tmp_path)))
    assert healthy == (None, (), (), None, None, (), (), None, False, False), (
        f"the healthy stream must flip nothing on any of the four axes; got {healthy}"
    )

    observed, declared = {}, {}
    for name in _CORPUS:
        observed[name] = _signature(_observe(_CORPUS[name][0](tmp_path / name)))
        declared[name] = _declared(name)
    mismatched = {name: (declared[name], observed[name])
                  for name in _CORPUS if declared[name] != observed[name]}
    assert not mismatched, (
        "every mutation must flip exactly its declared predicates.\n" + "\n".join(
            f"  {name}: declared={pair[0]}\n       observed={pair[1]}"
            for name, pair in mismatched.items())
    )

    collisions = {sig: sorted(n for n in declared if declared[n] == sig)
                  for sig in set(declared.values())}
    assert all(len(names) == 1 for names in collisions.values()), (
        "two mutations sharing one signature cannot be told apart from the report, which "
        "is what 'dies alone' has to mean operationally: "
        + str({str(sig): names for sig, names in collisions.items() if len(names) > 1})
    )
    assert len(_CORPUS) >= 11, (
        f"the corpus floor (§12 rows M1–M16 minus the CLI-level rows M11/M12/M13, which "
        f"are process-exit oracles rather than predicate rows); got {len(_CORPUS)}"
    )


def test_every_corpus_row_is_actually_caught(tmp_path) -> None:
    """The vacuity floor for the cross-check: a row whose builder silently produced the healthy
    stream would still satisfy `declared == observed` if its declaration were also empty."""
    uncaught = []
    for name in sorted(_CORPUS):
        observed = _observe(_CORPUS[name][0](tmp_path / f"vac_{name}"))
        if not (observed["a"]["verdict"] != "pass" or observed["b"]["verdict"] != "pass"
                or observed["c_fails"] or observed["manifest_fails"]):
            uncaught.append((name, observed["a"], observed["b"]))
    assert not uncaught, (
        "these corpus rows produced a fully GREEN result — the mutation is not being "
        f"caught by anything: {uncaught}"
    )
