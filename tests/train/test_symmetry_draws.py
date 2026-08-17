# >300 justify (R8): one instrument, pinned along its whole length — the block-function
# absence/presence unit tests, the real-FFI producer checks and the production
# `StepCoordinator` wiring rig share ONE claim ("the compact/spread counters reach
# `training_step` on the dense arm and are absent on graph") and the same config/buffer
# constants. Splitting the rig from the pins that drive it would fork the coordinator
# harness the k_cluster_histogram/uncovered_forced_win precedents already warn against
# duplicating (R5 bars cross-test imports).
"""R266/F-P1/N1 + LAW-18 — the R245(c) compact/spread symmetry gate gets its in-run
fire-rate counter, owed by ruling R266 ("before any dense-arm training").

THE MECHANISM. `fdc6f09` gated `ReplayBuffer::sample_batch_core` / `sample_batch_with_pos_core`
per record: a record that is window-lossless under every D6 element draws from the full
12-element group, one that is not draws only from `sym::WINDOW_PRESERVING_SYMS` (4 elements)
— a silent restriction with no in-run reading of how often each arm actually fires. The
counter ticks in `sample.rs::record_symmetry_draw`, the ONE call site both sample cores route
through (the be0637a precedent: the instrument attaches to the mechanism's measured live
path, not the encoding family it was first described under), and ONLY under `augment=True` —
an unaugmented draw never consults `compact` at all (`sym_idx` is unconditionally 0), so
counting it would fabricate a reading for a lever that was never exercised (the b349ec4/R249
disarmed-lever posture).

THE MAPPING. DENSE-only mechanism: the graph arm has no window and keeps the full D6 group
unconditionally (`sym.rs::WINDOW_PRESERVING_SYMS`'s own doc), so `symmetry_draw_block` is
keyed the SAME way as the K histogram (item 10(b)) — present on DENSE, ABSENT on GRAPH — on
the same `is_graph_run` authority two subtractions on the same grounds must not disagree
about.

SHAPE. Cumulative `{"compact": int, "spread": int, "compact_fraction": float | None}` — the
two raw counts (truthful at 0, R249) and the derived fraction, `None` when neither arm has
fired (a rate over zero samples is not a measurement). No producer (an engine build predating
the getters) -> keyed `None`, never a fabricated zero block.

NOT PINNED HERE, and deliberately: that the Rust counters FIRE, and that a compact-only /
spread-only buffer draws EXACTLY the group R245(c) promises. That burden is Rust-side
(`crates/mantis-selfplay/tests/replay_compact_gate.rs` —
`a_compact_only_buffer_draws_the_full_group`, `a_spread_only_buffer_draws_exactly_the_
window_preserving_subgroup`, `augment_false_ticks_neither_counter`,
`sample_batch_with_pos_core_ticks_the_same_counters`; killers: deleted `fetch_add`, swapped
compact/spread counters, hoisted tick above `if augment`, a second uncounted call site).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

from mantis import _engine
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.loader import load_config
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.events import SYMMETRY_DRAW_KEY, symmetry_draw_block
from mantis.train.lifecycle.signals import ShutdownState

_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
_DEV_CONFIG = load_config(_REPO / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)

GRID_CONFIG: dict[str, Any] = {"identity": {"encoding": "v6_live2_ls",
                                            "representation": "grid"}}
GRAPH_CONFIG: dict[str, Any] = {"identity": {"encoding": "gnn_axis_v1",
                                             "representation": "graph"}}


def _buffer(*, compact: Any, spread: Any = 0, omit_getter: bool = False) -> Any:
    if omit_getter:
        return SimpleNamespace(size=7, capacity=64)
    return SimpleNamespace(size=7, capacity=64, compact_draws=compact, spread_draws=spread)


# ═══ the block fn — the single spelling authority ═══════════════════════════════════
def test_a_graph_run_carries_no_symmetry_draws_key_at_all() -> None:
    """FALSIFYING MUTATION: delete the `if graph_run: return {}` arm of
    `symmetry_draw_block` — the key then ships on the graph arm, where the mechanism has
    no subject at all (the D37/K-histogram arm-(i) trap: a `{compact: 0, spread: 0}`
    reading zero on an arm nothing measures)."""
    block = symmetry_draw_block(_buffer(compact=5, spread=3), graph_run=True)
    assert block == {}, f"the graph arm must publish nothing; got {block}"


def test_a_dense_run_publishes_compact_spread_and_the_fraction() -> None:
    """FALSIFYING MUTATION: gate the block on `not graph_run` (copy-paste-inverted from
    the K histogram) — the instrument then vanishes from the ONLY arm the mechanism
    exists on."""
    block = symmetry_draw_block(_buffer(compact=30, spread=10), graph_run=False)
    assert block == {SYMMETRY_DRAW_KEY: {
        "compact": 30, "spread": 10, "compact_fraction": 0.75,
    }}


def test_a_truthful_all_zero_reading_is_published_not_dropped() -> None:
    """A raw counter is truthful at 0 (the R249 distinction: the COUNT is evidence, only a
    DERIVED rate over zero samples is fabrication) — a dense run that has not yet sampled
    with augmentation publishes real zeros for both counts."""
    block = symmetry_draw_block(_buffer(compact=0, spread=0), graph_run=False)
    assert block[SYMMETRY_DRAW_KEY]["compact"] == 0
    assert block[SYMMETRY_DRAW_KEY]["spread"] == 0


def test_zero_total_draws_yields_a_none_fraction_never_a_fabricated_one() -> None:
    block = symmetry_draw_block(_buffer(compact=0, spread=0), graph_run=False)
    assert block[SYMMETRY_DRAW_KEY]["compact_fraction"] is None, (
        "a fraction over zero draws is not a measurement (R249/b349ec4)"
    )


def test_an_engine_build_without_the_getters_publishes_none_not_zeros() -> None:
    """The event_manifest unproduced-field convention: keyed, `None`, never a fabricated
    `{compact: 0, spread: 0}` block."""
    block = symmetry_draw_block(_buffer(compact=0, omit_getter=True), graph_run=False)
    assert block == {SYMMETRY_DRAW_KEY: None}


def test_graph_omission_holds_even_when_the_buffer_reads_nonzero() -> None:
    """The gate is about the ARM (the run's declared identity), never the reading — a
    buffer double carrying nonzero counts must still be omitted on a graph-declared run."""
    block = symmetry_draw_block(_buffer(compact=99, spread=1), graph_run=True)
    assert block == {}


# ═══ the real FFI producer exists (wheel-compat caveat as for the Phase-T family) ════
def test_the_real_engine_getters_exist_and_read_zero_on_a_fresh_buffer() -> None:
    """LAW-07 producer leg at the seam: the shipped engine exposes both getters and a
    fresh buffer reads truthful zeros. (That the counters FIRE, and fire on the CORRECT
    arm, is pinned Rust-side in `replay_compact_gate.rs` — Python-side visibility alone
    is not a producer proof, the k_cluster_histogram/uncovered_forced_win posture.)"""
    buf = _engine.ReplayBuffer(capacity=8, encoding="v6")
    assert buf.compact_draws == 0
    assert buf.spread_draws == 0


def test_the_real_engine_getters_move_under_an_augmented_sample_drive() -> None:
    """A real drive across the FFI boundary: pushing rows and sampling with
    `augment=True` must move at least one of the two counters (which one depends on the
    rows' own compactness — the Rust suite pins the exact split; this is the Python-side
    confirmation that the hop from Rust atomic to bridge getter is live)."""
    import numpy as np

    buf = _engine.ReplayBuffer(capacity=8, encoding="v6")
    spec = buf.encoding
    n_cells = spec.trunk_size * spec.trunk_size
    n_chain = spec.chain_stride // n_cells
    for _ in range(8):
        buf.push(
            state=np.zeros((spec.n_planes, spec.trunk_size, spec.trunk_size), dtype=np.float16),
            chain_planes=np.zeros((n_chain, spec.trunk_size, spec.trunk_size),
                                  dtype=np.float16),
            policy=np.zeros(spec.policy_logit_count, dtype=np.float32),
            outcome=0.0,
            ownership=np.ones(n_cells, dtype=np.uint8),
            winning_line=np.zeros(n_cells, dtype=np.uint8),
        )
    buf.sample_batch(batch_size=8, augment=True)
    assert buf.compact_draws + buf.spread_draws == 8, (
        "8 augmented draws must tick the shared counters exactly 8 times between them; "
        f"got compact={buf.compact_draws} spread={buf.spread_draws}"
    )

    before_compact, before_spread = buf.compact_draws, buf.spread_draws
    buf.sample_batch(batch_size=4, augment=False)
    assert (buf.compact_draws, buf.spread_draws) == (before_compact, before_spread), (
        "an UNAUGMENTED draw must not move either counter — the lever was never exercised"
    )


# ═══ the production coordinator wiring — the hop `training_step` actually carries ═══
class _Pool:
    def __init__(self) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return SimpleNamespace(
            mcts_mean_depth=5.0, mcts_mean_root_concentration=0.1,
            cluster_value_std_mean=0.0, cluster_policy_disagreement_mean=0.0,
            cluster_variance_sample_count=0,
        )

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

    def save_checkpoint(self, loss_info) -> None:
        return None


class _Buffer:
    """A dense-arm buffer double carrying the R266 getters — the ONE thing this rig is
    for. `sample_batch_with_pos` is untouched by them (the counters are read AFTER the
    fact by the coordinator's own emitter, never derived from the sample call's return)."""

    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000
        self.compact_draws = 12
        self.spread_draws = 4

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _config(**overrides) -> Any:
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=1,
                                 knobs=_KNOBS),
        **{"eval_interval": 10**9, "min_buf_size": 10, "max_train_burst": 1,
           "training_steps_per_game": 4.0, "hard_gn_threshold": 1e9, "log_interval": 1,
           **overrides},
    )


def _coordinator(*, full_config: dict[str, Any]) -> Any:
    pool, trainer, buffer, sink = _Pool(), _Trainer(), _Buffer(), _Sink()
    shutdown = ShutdownState()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=_config(),
        full_config=full_config, train_cfg={}, mixing_cfg={}, sink=sink,
        monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer,
                           sink=sink, shutdown=shutdown)


def test_the_production_coordinator_carries_the_counters_into_training_step() -> None:
    """THE wiring pin: `_emit_training_step` must read `self.buffer.compact_draws` /
    `.spread_draws` and thread them into the SAME `training_step` event the 4 WARN rules
    read, on a dense-declared run. FALSIFYING MUTATION: delete the `symmetry_draws=`
    kwarg at the `emit_training_step_event` call site in `coordinator/step.py` — the
    hop breaks and `SYMMETRY_DRAW_KEY` vanishes from the production stream with every
    other pin in this file still green."""
    h = _coordinator(full_config=GRID_CONFIG)
    h.pool.games_completed = 5
    h.coord.step()

    events = h.sink.named("training_step")
    assert len(events) == 1, f"one log_interval=1 boundary must emit exactly one row; got {len(events)}"
    assert events[0][SYMMETRY_DRAW_KEY] == {
        "compact": 12, "spread": 4, "compact_fraction": 0.75,
    }, f"got {events[0].get(SYMMETRY_DRAW_KEY)!r}"


def test_the_production_coordinator_omits_the_key_on_a_graph_declared_run() -> None:
    """The SAME coordinator hop (`_emit_training_step`, driven directly — the graph
    ROUTE dispatch itself is `dispatch.py`'s subject, not this instrument's; a `_Buffer`
    fake with no `sample_graph_batch` would red on the wrong seam), on the arm the
    mechanism has no subject on — the key must not appear at all, not as a keyed zero."""
    h = _coordinator(full_config=GRAPH_CONFIG)
    loss_info = {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}
    h.coord._emit_training_step(h.coord.config, loss_info, h.sink)

    events = h.sink.named("training_step")
    assert len(events) == 1
    assert SYMMETRY_DRAW_KEY not in events[0], (
        f"graph run must omit {SYMMETRY_DRAW_KEY} entirely; got {events[0].get(SYMMETRY_DRAW_KEY)!r}"
    )
