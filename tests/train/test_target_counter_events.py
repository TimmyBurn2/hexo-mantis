# >300 justify (R8): the eight rows are ONE claim — the three target-integrity counters reach
# the run's OWN event stream with a fire-rate over a published denominator — over ONE harness,
# a real `StepCoordinator` driven past its `log_interval` boundary with a riggable
# `runner_stats` snapshot. Cross-test imports are barred, so a split would fork the rig into
# copies that drift while both stay green.
"""The target-integrity counters reach the ONE channel, in-run, with their rate and denominator.

RED at HEAD on its own mechanism: every row fails because the `iteration_complete` payload does
not carry `target_integrity`. The seam exists end to end and only the last stage is missing, so
a live run cannot attribute its own game-shape drift.

The precedent this must not repeat is `solver_deltas` in the same function: a DEFAULTED
parameter with zero callers passing it, no manifest row and no producer test, so eight fire-rate
counters silently never reach the stream. This file is the emission and signature legs that make
that impossible here; the third leg is the producer-manifest row.

Each row is the only witness to one defect and names the mutation that reds it: the key reaching
the stream at all; an advance being READABLE within one `log_interval`; `delta` and `total` not
swapping slots; `per_position` being `None` rather than a fabricated `0.0`; an IDLE lever staying
VISIBLE at 0; no crosswiring, proven with three DISTINCT values; the parameter being required
with NO default, since a default is a MIGRATED authority rather than an absent one; and a
DECREASE emitted as measured, since the atomics are monotonic and a clamp would hide a wiring
bug rather than a real event.

Real here: the shipped `StepCoordinator`, its `_run_log_interval` boundary, the real emit
builders and payloads, and the real `RunnerStats` dataclass. Fake: the pool, trainer, buffer and
the counter VALUES.
"""
from __future__ import annotations

from mantis._engine import HexgBuffer

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
from mantis.train.events import emit_iteration_complete_event, emit_training_step_event
from mantis.train.lifecycle.signals import ShutdownState
from pathlib import Path

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


_REPO = Path(__file__).resolve().parents[2]
_DEV_CONFIG = load_config(_REPO / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)
#: Harnesses that set `log_interval` MIRROR it onto `gate_interval` — the shipped posture, since
#: every committed config mints the two equal.
_GATE_INTERVAL = _DEV_CONFIG.monitor.gate_interval

#: The counters carried in the `target_integrity` block plus the denominator the rate is taken
#: over. Transcribed rather than derived from the payload under test: an oracle that read its
#: own expectation off the subject would be satisfied by any consistent renaming. Three live
#: counters are needed for the crosswire proof below to be a proof at all.
_COUNTERS = ("export_offwindow_mass_moves", "target_integrity_defects",
             "inference_failures_total")
_DENOMINATOR = "positions_delta"
_SLOTS = ("total", "delta", "per_position")
_PAYLOAD_KEY = "target_integrity"


def _stats(*, positions: int, export_offwindow: int, seam: int, defects: int) -> RunnerStats:
    """Build a REAL `RunnerStats` snapshot with the four load-bearing numbers EXPLICIT: none has
    a default, so a row that forgot one fails loudly rather than inheriting a zero."""
    return RunnerStats(
        games_completed=0, positions_generated=positions, x_wins=0, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=5.0,
        mcts_mean_root_concentration=0.1,
        export_offwindow_mass_moves=export_offwindow,
        target_integrity_defects=defects, inference_failures_total=seam,
    )


class _Pool:
    """A pool whose `runner_stats()` answer the drive sets EXPLICITLY before each step. No
    internal call counter decides which snapshot is returned, so the oracle measures the payload
    rather than the reader's call pattern."""

    search_kind = "gumbel"
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # the third outcome share.
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    def __init__(self, stats: RunnerStats) -> None:
        self._games = 0
        self.recent_move_histories: list = []
        self.current = stats

    @property
    def games_completed(self) -> int:
        # A step only runs when new games have arrived, so a CONSTANT game count would drive
        # exactly one boundary and silently degrade every two-emit row into a one-emit row.
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
        # `event` is subscripted, not `.get`-ed: a payload without it is a producer defect.
        return [e for e in self.events if e["event"] == name]


def _drive(*snapshots: RunnerStats) -> list[dict]:
    """Drive a REAL `StepCoordinator` once per snapshot at `log_interval=1` and return the
    `iteration_complete` payloads, one per step, in order. The production cadence is the SAME
    guard line with a bigger modulus."""
    assert snapshots, "a drive with no snapshot measures nothing"
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        # Gate cadence mirrors narration cadence, the shipped posture.
        **{"eval_interval": 10**9, "log_interval": 1, "gate_interval": 1,
           "min_buf_size": 10},
    )
    pool = _Pool(snapshots[0])
    sink = _SpySink()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=config,
        full_config=_GRAPH_FULL_CONFIG,
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
    """Return the nested block, subscripted so its absence is loud."""
    return payload[_PAYLOAD_KEY]


# O-20 — the emission leg; this node id is cited by the producer-manifest row.
def test_iteration_complete_carries_the_target_integrity_fire_rates() -> None:
    """The stream carries the three counters, each with `total`, `delta` and a `per_position`
    rate, beside the `positions_delta` denominator that rate is taken over — fires per RECORDED
    POSITION, published rather than left for a consumer to guess."""
    payload = _drive(
        _stats(positions=1200, export_offwindow=17, seam=3, defects=0),
        _stats(positions=2400, export_offwindow=41, seam=9, defects=0),
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


# O-21 — the drift witness is READABLE in-run.
def test_the_offwindow_witness_advance_is_readable_within_one_log_interval() -> None:
    """An advance between two boundaries is visible as a NONZERO `delta` at the next one, not
    merely as a bigger `total` a reader would have to difference by hand across two log files."""
    first, second = _drive(
        _stats(positions=1000, export_offwindow=100, seam=0, defects=0),
        _stats(positions=2000, export_offwindow=175, seam=0, defects=0),
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


# O-22 — delta is the interval, total is cumulative.
def test_the_delta_is_the_interval_change_and_the_total_is_cumulative() -> None:
    """`total` answers "how much has this lever fired all run", `delta` answers "is it firing
    NOW". Asserted on the SECOND emit, so the claim is independent of the first emit's baseline
    convention."""
    payloads = _drive(
        _stats(positions=500, export_offwindow=10, seam=200, defects=0),
        _stats(positions=1500, export_offwindow=10, seam=260, defects=0),
    )
    block = _integrity(payloads[-1])

    assert block["inference_failures_total"]["total"] == 260, (
        f"total is the cumulative counter; got {block['inference_failures_total']['total']!r}"
    )
    assert block["inference_failures_total"]["delta"] == 60, (
        "delta is t2 − t1 over the interval, not the total again; got "
        f"{block['inference_failures_total']['delta']!r}"
    )
    assert block[_DENOMINATOR] == 1000, (
        f"…and the denominator is the interval's own recorded positions; got "
        f"{block[_DENOMINATOR]!r}"
    )
    assert block["export_offwindow_mass_moves"]["delta"] == 0, (
        "a counter that did NOT advance over the interval has delta 0 while its total stays "
        f"10 — the two really are different numbers; got {block['export_offwindow_mass_moves']}"
    )


# O-23 — an unmeasurable rate is None, never a fabricated 0.0.
def test_per_position_is_None_when_no_position_was_recorded() -> None:
    """With zero positions recorded there is NO rate to publish: an unproduced field carries
    `None`, because a constant `0` in the ONE channel reads as a real measurement. The tempting
    `delta / max(1, positions_delta)` guard fabricates exactly that reading."""
    payloads = _drive(
        _stats(positions=800, export_offwindow=5, seam=5, defects=0),
        _stats(positions=800, export_offwindow=9, seam=5, defects=0),
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


# O-24 — an idle lever stays visible.
def test_an_idle_lever_stays_visible_at_zero() -> None:
    """A counter at zero is PUBLISHED at zero, which distinguishes "starved" from "not firing".
    `target_integrity_defects` most depends on it: that latch is run-FATAL, so it reads 0 in
    every run that survives to emit."""
    payloads = _drive(
        _stats(positions=1000, export_offwindow=0, seam=0, defects=0),
        _stats(positions=3000, export_offwindow=0, seam=0, defects=0),
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


# O-25 — no crosswiring.
def test_the_three_counters_do_not_crosswire() -> None:
    """Three distinct rigged values thread 1:1 into three distinct slots — distinct in both
    `total` and `delta`, so a swap cannot alias. Counters in each other's slots satisfy the
    emission, arithmetic and idle rows alike."""
    payloads = _drive(
        _stats(positions=1000, export_offwindow=11, seam=22, defects=33),
        _stats(positions=2000, export_offwindow=111, seam=222, defects=333),
    )
    block = _integrity(payloads[-1])

    expected_total = {"export_offwindow_mass_moves": 111, "inference_failures_total": 222,
                      "target_integrity_defects": 333}
    expected_delta = {"export_offwindow_mass_moves": 100, "inference_failures_total": 200,
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


# O-26 — the signature leg.
def test_the_target_integrity_parameter_has_no_default() -> None:
    """`target_integrity` is a required keyword-only parameter with NO default.

    The leg no emission oracle can see, and the reason `solver_deltas` rotted: with a default, a
    caller that omits it emits nothing and every assertion is written against payloads emitted
    without it. With none, omitting it is a loud `TypeError` at the first boundary crossing.
    """
    # The pin reads the builder that actually emits the block, not the retired pre-split
    # wrapper, whose own docstring said it existed to keep this assertion green.
    parameters = inspect.signature(emit_iteration_complete_event).parameters
    assert _PAYLOAD_KEY in parameters, (
        f"`emit_iteration_complete_event` must take `{_PAYLOAD_KEY}` — the payload cannot "
        f"carry what the builder was never handed. Parameters: {list(parameters)}"
    )
    parameter = parameters[_PAYLOAD_KEY]
    assert parameter.default is inspect.Parameter.empty, (
        f"`{_PAYLOAD_KEY}` carries a default ({parameter.default!r}). That is the "
        "`solver_deltas` shape verbatim: the one caller can then stop passing it and the "
        "counters leave the stream with every test still green"
    )
    # THE KEYWORD-ONLY HALF IS BANKED, not silently dropped: it held on the retired WRAPPER and
    # does not hold on the live builder, whose eight call sites each pass fourteen positionals —
    # precisely the hazard the clause named, and a signature change rather than an assertion.
    solver = inspect.signature(emit_training_step_event).parameters["solver_deltas"]
    assert solver.default is None, (
        "premise (the CONTRAST this row is defined against): `solver_deltas` — which lives on "
        "`emit_training_step_event`, the OTHER half of the R210 split — is still the "
        "defaulted, uncalled parameter; Phase O leaves it byte-untouched and queues it "
        f"(`Q-O-SOLVERDELTAS`) rather than tidying it inside a taxonomy commit. Got "
        f"{solver.default!r}; if this moved, the queue row was taken and this premise must "
        "be re-pointed, never deleted"
    )


# O-28 — a decrease is emitted as measured.
def test_a_counter_decrease_is_emitted_as_measured_and_never_clamped() -> None:
    """A negative delta is published, never clamped: the atomics are monotonic, so a
    `max(0, ...)` would hide a wiring bug — a swapped snapshot, a re-created pool, a counter read
    off the wrong runner — rather than suppress a real event."""
    payloads = _drive(
        _stats(positions=1000, export_offwindow=100, seam=0, defects=0),
        _stats(positions=2000, export_offwindow=40, seam=0, defects=0),
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
