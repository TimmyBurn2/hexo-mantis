# >300 justify (R8), stated at this file's MEASURED size of 538 lines (`wc -l` at
# ORACLE-WRITE, never transcribed — SF-7; IMPL re-measures rather than carrying this number
# forward). The eight rows are ONE claim — the three Phase-T target-integrity counters reach the
# run's OWN event stream with a fire-rate over a published denominator (LAW-18) — over ONE
# harness: a real `StepCoordinator` driven past its `log_interval` boundary with a riggable
# `runner_stats` snapshot. Every row needs the whole rig, R5 bars cross-test imports, and a
# split would fork the rig into copies that drift while both stay green. Executable content
# is a minority; the rest is the per-row mutation and the "what defect is this the only
# witness to" rationale LAW-07 asks each row to carry.
"""⊕ WP12-R Phase O / O-20..O-26, O-28 (R164/LAW-18) — the Phase-T counters reach the ONE
channel, in-run, with their rate and their denominator.

RED-at-HEAD on its own mechanism (⊕): no import anchor — every row below fails because the
`iteration_complete` payload does not carry `target_integrity`, which is the defect itself.

R164's finding, stated: `PREREG_T §0b` names `export_offwindow_mass_moves` as "the in-run
witness attributing" the expected game-shape drift. At HEAD that counter is readable ONLY by
a test that calls `runner_stats(pool)` — it is NOT in the run's own stream, so a live run
cannot attribute its own drift, and LAW-18's text ("a lever under test must log its own
fire-rate IN-RUN; a post-hoc offline probe cannot distinguish 'starved' from 'ineffective'")
is unmet. The seam already exists end to end (Rust atomics → snapshot → bridge getters →
`pool_hooks.runner_stats` → `events.py:266`, which ALREADY calls it); only the last stage is
missing.

The precedent this landing must not repeat is in the same function: `solver_deltas`
(`events.py:202,259-260`) is a DEFAULTED parameter with zero callers passing it, no manifest
row and no producer test — so eight solver fire-rate counters silently never reach the
stream and no test can tell. This file is two of the three legs that make that impossible
here (emission and signature); the third — resolution — is the producer-manifest row, pinned
in `tests/monitor/test_manifest_contract.py`.

The rows, and the defect each is the ONLY witness to:

- O-20 `test_iteration_complete_carries_the_target_integrity_fire_rates` — the emission leg,
  and the node the `target_integrity_counters` manifest row cites as its producer test
  (LAW-07). Sole witness that the key reaches the stream at all. MUTATION (M-O20): make
  `_target_integrity_report` return `{}` → the key vanishes.
- O-21 — the §0b WITNESS specifically: an advance in `export_offwindow_mass_moves` is
  READABLE in the stream within one `log_interval`. A payload that carries the key but a
  frozen number satisfies O-20 and tells a live run nothing. MUTATION (M-O21): snapshot the
  counters once at construction and reuse.
- O-22 — `delta` is the INTERVAL change and `total` is cumulative. They are different
  numbers and a consumer reads them differently; publishing one in the other's slot is
  invisible to every other row. MUTATION (M-O22).
- O-23 — `per_position` is `None`, never `0.0`, when nothing was recorded.
  `event_manifest.md:230-234` verbatim: "An unproduced field carries `None`, never a
  fabricated value. A constant `0` in the ONE channel reads as a real measurement and is the
  F-10 class in miniature." MUTATION (M-O23): `delta / max(1, positions_delta)`.
- O-24 — an IDLE lever stays VISIBLE at 0 (the `chain_loss_with_fire_rate` posture,
  `losses.py:224-238`). This is what makes "starved" distinguishable from "not firing", and
  it is the row `target_integrity_defects` depends on: that latch is run-fatal, so it reads
  0 in every run that lives to emit. MUTATION (M-O24): omit zero-valued counters.
- O-25 — no crosswiring, proven with three DISTINCT values. Two counters that reached the
  right payload in the wrong slots pass O-20, O-22 and O-24. MUTATION (M-O25): swap two.
- O-26 — the anti-`solver_deltas` SIGNATURE leg: `target_integrity` is a required
  keyword-only parameter with NO default. A parameter default is a MIGRATED authority, not
  an absent one (`run.py:366-372`, MF-2 Attack B); with `= None` a caller that forgets it
  emits nothing and no test can see the difference. MUTATION (M-O26): give it `= None` in
  the CALLEE's signature — and that is the ONLY mutation that can red this row, because a
  call-site edit cannot move a callee's signature.
- O-28 — a DECREASE is emitted as measured, never clamped. The atomics are monotonic so it
  cannot happen; a `max(0, …)` would therefore hide a wiring bug rather than a real event,
  which is the `actor_lag_negative` precedent (`event_manifest.md:101`: "a negative lag is a
  wiring bug reported loudly once, never a fire"). MUTATION (M-O28).

**What is real here and what is not.** Real: the shipped `StepCoordinator`, its real
`_run_log_interval` boundary, the real `emit_training_events`, the real event payloads and
the real `RunnerStats` snapshot dataclass (so a field rename in `pool_hooks` reds this file
rather than being papered over by a hand-shaped double). Fake: the pool, trainer and buffer —
the injected seam every coordinator drive in this suite stands in — and the counter VALUES,
which are rigged: a real advance would need a live Rust runner playing real games, which
`tests/selfplay/test_target_law18_counters.py` already owns for the SURFACE half. This file's
subject is the last stage, from the snapshot to the stream.
"""
from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.selfplay.pool_hooks import RunnerStats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.events import emit_training_events
from mantis.train.lifecycle.signals import ShutdownState
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEV_CONFIG = load_config(_REPO / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)

#: The three Phase-T counters, in the order `IMPL_NOTES_T §3.6` names them, plus the
#: denominator the rate is taken over. Transcribed rather than derived from the payload under
#: test: an oracle that read its own expectation off the subject would be satisfied by any
#: consistent renaming (R81).
_COUNTERS = ("export_offwindow_mass_moves", "gridls_zero_policy_rows",
             "target_integrity_defects")
_DENOMINATOR = "positions_delta"
_SLOTS = ("total", "delta", "per_position")
_PAYLOAD_KEY = "target_integrity"


def _stats(*, positions: int, export_offwindow: int, gridls_zero: int, defects: int) -> RunnerStats:
    """A REAL `RunnerStats` snapshot with the four load-bearing numbers supplied EXPLICITLY.

    Every parameter is required and none has a default: the three counters and their
    denominator are exactly what each assertion below rides on, so a row that forgot to state
    one must fail loudly here rather than inherit a zero from the dataclass and then assert
    against it. The remaining fields are scalars this payload does not read.
    """
    return RunnerStats(
        games_completed=0, positions_generated=positions, x_wins=0, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=5.0,
        mcts_mean_root_concentration=0.1, cluster_value_std_mean=0.0,
        cluster_policy_disagreement_mean=0.0, cluster_variance_sample_count=0,
        export_offwindow_mass_moves=export_offwindow, gridls_zero_policy_rows=gridls_zero,
        target_integrity_defects=defects,
    )


class _Pool:
    """A pool whose `runner_stats()` answer the drive sets EXPLICITLY before each step. No
    internal call counter decides which snapshot is returned: `events.py` is free to read the
    snapshot once or twice per emit, and an oracle whose rigging depended on that would be
    measuring the reader's call pattern instead of the payload."""

    gumbel_mcts = True
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    def __init__(self, stats: RunnerStats) -> None:
        self._games = 0
        self.recent_move_histories: list = []
        self.current = stats

    @property
    def games_completed(self) -> int:
        # A step only runs when new games have arrived since the last one, so a CONSTANT
        # game count drives exactly one `log_interval` boundary and every two-emit row here
        # would silently degrade into a one-emit row. Incrementing on read is the house rig
        # (`tests/test_run_disk_guard_abort_rc.py::_Pool`).
        self._games += 1
        return self._games

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> RunnerStats:
        return self.current

    def sync_inference_weights(self, state_dict: Any) -> None:
        return None

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        return self.train_step_from_tensors()

    def save_checkpoint(self, loss_info: Any) -> None:
        return None


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, path: Any) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        # `event` is subscripted, not `.get`-ed: a payload without it is a producer defect
        # and must be loud rather than silently filtered out of every assertion below.
        return [e for e in self.events if e["event"] == name]


def _drive(*snapshots: RunnerStats) -> list[dict]:
    """Drive a REAL `StepCoordinator` once per snapshot, at `log_interval=1`, and return the
    `iteration_complete` payloads it emitted — one per step, in order.

    `log_interval=1` is the smallest boundary that produces an emit per step; the production
    cadence (run5 mints 1000) is the SAME guard line (`step.py:369`) with a bigger modulus,
    so nothing about the payload under test depends on the number.
    """
    assert snapshots, "a drive with no snapshot measures nothing"
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, knobs=_KNOBS),
        **{"eval_interval": 10**9, "log_interval": 1, "min_buf_size": 10},
    )
    pool = _Pool(snapshots[0])
    sink = _SpySink()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=config,
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}},
        train_cfg={}, mixing_cfg={}, sink=sink, monitor_cfg=MonitorConfig(),
    )
    for snapshot in snapshots:
        pool.current = snapshot
        coord.step()
    payloads = sink.named("iteration_complete")
    assert len(payloads) == len(snapshots), (
        f"premise: one `iteration_complete` per driven step at log_interval=1; drove "
        f"{len(snapshots)} and saw {len(payloads)}"
    )
    return payloads


def _integrity(payload: dict) -> dict:
    """The nested block, subscripted so its absence is the loud failure this file exists to
    produce."""
    return payload[_PAYLOAD_KEY]


# ══ O-20 — the emission leg (this node id is cited by the producer-manifest row) ═══════
def test_iteration_complete_carries_the_target_integrity_fire_rates() -> None:
    """O-20 — the LAW-07 producer test the `target_integrity_counters` manifest row cites.

    A live run's own stream must carry the three Phase-T counters, each with its cumulative
    `total`, its interval `delta` and a `per_position` RATE, beside the `positions_delta`
    denominator that rate is taken over. LAW-03: the unit is fires per RECORDED POSITION —
    not per game, not per ply — which is why the denominator is published rather than left
    for a consumer to guess.

    MUTATION THAT REDS IT (M-O20): `_target_integrity_report` returns `{}` → the key is
    absent from the emitted payload. O-26 stays green (the signature is untouched) and O-27
    stays green (the producer symbol still resolves), which is why those two rows exist
    separately."""
    payload = _drive(
        _stats(positions=1200, export_offwindow=17, gridls_zero=3, defects=0),
        _stats(positions=2400, export_offwindow=41, gridls_zero=9, defects=0),
    )[-1]

    block = _integrity(payload)
    assert _DENOMINATOR in block, (
        "the RATE's denominator must travel beside it — a rate whose denominator a consumer "
        f"has to guess is not a measurement (LAW-03). Keys: {sorted(block)}"
    )
    missing = [name for name in _COUNTERS if name not in block]
    assert missing == [], (
        f"the three Phase-T counters must ALL reach the stream; missing {missing}. "
        f"Keys: {sorted(block)}"
    )
    for name in _COUNTERS:
        absent_slots = [slot for slot in _SLOTS if slot not in block[name]]
        assert absent_slots == [], (
            f"{name} is missing {absent_slots} — `total` alone is a cumulative number nobody "
            f"can attribute to an interval, which is what LAW-18 asks for. Got {block[name]}"
        )


# ══ O-21 — the §0b drift witness is READABLE in-run ════════════════════════════════════
def test_the_offwindow_witness_advance_is_readable_within_one_log_interval() -> None:
    """O-21 — R164's premise, closed or not closed.

    `PREREG_T §0b` names `export_offwindow_mass_moves` as THE in-run witness attributing the
    expected game-shape drift. A witness a live run cannot read is not a witness. So an
    advance between two `log_interval` boundaries must be visible as a NONZERO `delta` in the
    stream at the next boundary — not merely as a bigger `total` a reader would have to
    difference by hand across two log files.

    MUTATION THAT REDS IT (M-O21): read the counters ONCE at coordinator construction and
    reuse the snapshot — `total` freezes, `delta` stalls at 0, and O-24 (the idle case) stays
    green throughout, which is exactly why this row rigs an ADVANCE."""
    first, second = _drive(
        _stats(positions=1000, export_offwindow=100, gridls_zero=0, defects=0),
        _stats(positions=2000, export_offwindow=175, gridls_zero=0, defects=0),
    )

    witness = _integrity(second)["export_offwindow_mass_moves"]
    assert witness["total"] == 175, (
        f"the cumulative counter must track the live snapshot; got {witness['total']!r}"
    )
    assert witness["delta"] == 75, (
        "the advance between the two boundaries must be READABLE as the interval delta — a "
        f"frozen or stalled read is R164's defect verbatim; got {witness['delta']!r}"
    )
    assert witness["per_position"] == 75 / 1000, (
        "…and the rate is that advance over the positions recorded in the SAME interval "
        f"(75 / 1000); got {witness['per_position']!r}"
    )
    assert _integrity(first)["export_offwindow_mass_moves"]["total"] == 100, (
        "premise: the first boundary saw the pre-advance value, so the delta above is a real "
        "interval and not an artefact of a single emit"
    )


# ══ O-22 — delta is the interval, total is cumulative ══════════════════════════════════
def test_the_delta_is_the_interval_change_and_the_total_is_cumulative() -> None:
    """O-22. Two numbers with two meanings, read by consumers who need them apart: `total`
    answers "how much has this lever fired all run", `delta` answers "is it firing NOW".

    Asserted on the SECOND emit so the claim is independent of whatever baseline the first
    emit uses — a first-emit-only oracle would be measuring the constructor's convention
    rather than the arithmetic.

    MUTATION THAT REDS IT (M-O22): publish `total` in the `delta` slot. O-20 stays green (the
    key and all three slots are still there), which is why this row is separate from it."""
    payloads = _drive(
        _stats(positions=500, export_offwindow=10, gridls_zero=200, defects=0),
        _stats(positions=1500, export_offwindow=10, gridls_zero=260, defects=0),
    )
    block = _integrity(payloads[-1])

    assert block["gridls_zero_policy_rows"]["total"] == 260, (
        f"total is the cumulative counter; got {block['gridls_zero_policy_rows']['total']!r}"
    )
    assert block["gridls_zero_policy_rows"]["delta"] == 60, (
        "delta is t2 − t1 over the interval, not the total again; got "
        f"{block['gridls_zero_policy_rows']['delta']!r}"
    )
    assert block[_DENOMINATOR] == 1000, (
        f"…and the denominator is the interval's own recorded positions; got "
        f"{block[_DENOMINATOR]!r}"
    )
    assert block["export_offwindow_mass_moves"]["delta"] == 0, (
        "a counter that did NOT advance over the interval has delta 0 while its total stays "
        f"10 — the two really are different numbers; got {block['export_offwindow_mass_moves']}"
    )


# ══ O-23 — an unmeasurable rate is None, never a fabricated 0.0 ════════════════════════
def test_per_position_is_None_when_no_position_was_recorded() -> None:
    """O-23 — the F-10 class in miniature, and the repo's own stated convention.

    `event_manifest.md:230-234` verbatim: "An unproduced field carries `None`, never a
    fabricated value. A constant `0` in the ONE channel reads as a real measurement." With
    zero positions recorded in the interval there is NO rate to publish: `0.0` would tell a
    reader "this lever did not fire per position", which is a claim nobody measured.

    MUTATION THAT REDS IT (M-O23): `per_position = delta / max(1, positions_delta)` — the
    tempting divide-by-zero guard, which fabricates exactly the reading this row forbids."""
    payloads = _drive(
        _stats(positions=800, export_offwindow=5, gridls_zero=5, defects=0),
        _stats(positions=800, export_offwindow=9, gridls_zero=5, defects=0),
    )
    block = _integrity(payloads[-1])

    assert block[_DENOMINATOR] == 0, (
        f"premise: no position was recorded in this interval; got {block[_DENOMINATOR]!r}"
    )
    for name in _COUNTERS:
        assert block[name]["per_position"] is None, (
            f"{name}.per_position must be None with a zero denominator — a fabricated "
            f"{block[name]['per_position']!r} reads as a real measurement"
        )
    assert block["export_offwindow_mass_moves"]["delta"] == 4, (
        "…and the delta is still MEASURED and published: it is the RATE that is unavailable, "
        f"not the count; got {block['export_offwindow_mass_moves']['delta']!r}"
    )


# ══ O-24 — an idle lever stays visible ═════════════════════════════════════════════════
def test_an_idle_lever_stays_visible_at_zero() -> None:
    """O-24 — LAW-18's "starved vs ineffective" distinction, which only a VISIBLE zero can
    make. The `chain_loss_with_fire_rate` posture (`losses.py:224-238`) publishes
    `fired: False, fire_rate: 0.0` rather than omitting the block, and
    `tests/selfplay/test_target_law18_counters.py` already names that as the Phase-T posture
    for these three counters on the surface half.

    It is `target_integrity_defects` that most depends on this: that latch is run-FATAL
    (`search_drive.rs:765-767` stores the typed message and breaks; the bridge drain face
    raises), so it reads 0 in every run that survives to emit an `iteration_complete`. A
    payload that omitted zero-valued counters would make that permanent 0 indistinguishable
    from a field with no producer — which is F-10 exactly.

    MUTATION THAT REDS IT (M-O24): omit counters whose `total == 0`."""
    payloads = _drive(
        _stats(positions=1000, export_offwindow=0, gridls_zero=0, defects=0),
        _stats(positions=3000, export_offwindow=0, gridls_zero=0, defects=0),
    )
    block = _integrity(payloads[-1])

    assert block[_DENOMINATOR] == 2000, (
        f"premise: positions WERE recorded, so a rate is available; got {block[_DENOMINATOR]!r}"
    )
    for name in _COUNTERS:
        assert name in block, (
            f"{name} vanished from the payload because it never fired — an idle lever that "
            f"goes invisible is indistinguishable from one with no producer. Keys: {sorted(block)}"
        )
        assert block[name]["total"] == 0 and block[name]["delta"] == 0, (
            f"{name} must report its zeros as measurements; got {block[name]}"
        )
        assert block[name]["per_position"] == 0.0, (
            f"{name}.per_position is a MEASURED 0.0 here (the denominator is nonzero), which "
            f"is a different statement from the None of O-23; got {block[name]['per_position']!r}"
        )


# ══ O-25 — no crosswiring ══════════════════════════════════════════════════════════════
def test_the_three_counters_do_not_crosswire() -> None:
    """O-25. Three distinct rigged values thread 1:1 into three distinct slots.

    Two counters that reached the right payload in each other's slots satisfy O-20 (the key
    and the slots are there), O-22 (the arithmetic is still right, just about the wrong
    lever) and O-24 (nothing is omitted). Only distinct values can see it, and the values
    are chosen distinct-in-both-total-and-delta so a swap cannot alias.

    MUTATION THAT REDS IT (M-O25): swap `gridls_zero_policy_rows` and
    `target_integrity_defects` in the report builder."""
    payloads = _drive(
        _stats(positions=1000, export_offwindow=11, gridls_zero=22, defects=33),
        _stats(positions=2000, export_offwindow=111, gridls_zero=222, defects=333),
    )
    block = _integrity(payloads[-1])

    expected_total = {"export_offwindow_mass_moves": 111, "gridls_zero_policy_rows": 222,
                      "target_integrity_defects": 333}
    expected_delta = {"export_offwindow_mass_moves": 100, "gridls_zero_policy_rows": 200,
                      "target_integrity_defects": 300}
    observed_total = {name: block[name]["total"] for name in _COUNTERS}
    observed_delta = {name: block[name]["delta"] for name in _COUNTERS}
    assert observed_total == expected_total, (
        f"the counters are crosswired at the total slot.\n  expected: {expected_total}\n"
        f"  observed: {observed_total}"
    )
    assert observed_delta == expected_delta, (
        f"the counters are crosswired at the delta slot.\n  expected: {expected_delta}\n"
        f"  observed: {observed_delta}"
    )


# ══ O-26 — the signature leg (the anti-`solver_deltas` pin) ════════════════════════════
def test_the_target_integrity_parameter_has_no_default() -> None:
    """O-26 — the leg no emission oracle can see, and the reason `solver_deltas` rotted.

    `solver_deltas: dict | None = None` (`events.py:202`) has ZERO callers passing it. The
    payload key silently never appears and no test can tell, because every emission assertion
    in the repo is written against payloads that were emitted WITHOUT it. A parameter default
    is a MIGRATED authority, not an absent one (`run.py:366-372`, MF-2 Attack B): with no
    default, a caller that omits `target_integrity` is a `TypeError` the first time the
    coordinator crosses a `log_interval` boundary, in a test, loudly.

    MUTATION THAT REDS IT (M-O26): give the parameter `= None` in the CALLEE's signature —
    and this is the ONLY mutation that can red this row. Deleting the ARGUMENT at the call
    site cannot move a callee's signature; it reds the emission rows instead. The two legs
    are independent and each needs its own killer."""
    parameters = inspect.signature(emit_training_events).parameters
    assert _PAYLOAD_KEY in parameters, (
        f"`emit_training_events` must take `{_PAYLOAD_KEY}` — the payload cannot carry what "
        f"the builder was never handed. Parameters: {list(parameters)}"
    )
    parameter = parameters[_PAYLOAD_KEY]
    assert parameter.default is inspect.Parameter.empty, (
        f"`{_PAYLOAD_KEY}` carries a default ({parameter.default!r}). That is the "
        "`solver_deltas` shape verbatim: the one caller can then stop passing it and the "
        "counters leave the stream with every test still green"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"…and it must be KEYWORD-ONLY, so it cannot be supplied positionally by accident "
        f"from the caller's long argument list; got {parameter.kind}"
    )
    solver = parameters["solver_deltas"]
    assert solver.default is None, (
        "premise (the CONTRAST this row is defined against): `solver_deltas` is still the "
        "defaulted, uncalled parameter — Phase O leaves it byte-untouched and queues it "
        f"(`Q-O-SOLVERDELTAS`) rather than tidying it inside a taxonomy commit. Got "
        f"{solver.default!r}; if this moved, the queue row was taken and this premise must "
        "be re-pointed, never deleted"
    )


# ══ O-28 — a decrease is emitted as measured ═══════════════════════════════════════════
def test_a_counter_decrease_is_emitted_as_measured_and_never_clamped() -> None:
    """O-28. The atomics are monotonic and the pool is not restarted mid-run, so a negative
    delta CANNOT happen. That is precisely why it must not be clamped: a `max(0, …)` would
    hide a wiring bug — a swapped snapshot, a re-created pool, a counter read off the wrong
    runner — rather than suppress a real event. The repo already took this position once:
    `event_manifest.md:101`, "a negative lag is a wiring bug reported loudly once, never a
    fire".

    MUTATION THAT REDS IT (M-O28): `delta = max(0, t2 - t1)`. Every other row here stays
    green, because no other row ever drives a decrease."""
    payloads = _drive(
        _stats(positions=1000, export_offwindow=100, gridls_zero=0, defects=0),
        _stats(positions=2000, export_offwindow=40, gridls_zero=0, defects=0),
    )
    block = _integrity(payloads[-1])
    witness = block["export_offwindow_mass_moves"]

    assert witness["total"] == 40, (
        f"premise: the rigged snapshot really did go backwards; got {witness['total']!r}"
    )
    assert witness["delta"] == -60, (
        "a decrease must reach the stream AS MEASURED — clamping it to 0 hides a wiring bug "
        f"behind a plausible-looking reading; got {witness['delta']!r}"
    )
    assert witness["per_position"] == -60 / 1000, (
        f"…and the rate follows the measured delta; got {witness['per_position']!r}"
    )
