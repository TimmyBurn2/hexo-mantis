"""⊕ WPAX Phase D ORACLE — `CARD-DRAWRATE-KEY`: the resolved value actually REACHES the
runtime, and all three of the block's keys reach their own call site (DESIGN_D §4, §5.3).

RED-at-import until IMPL lands the delta. Two anchors: `mantis.config.resolve.draw_rate`
(the ONE read path, R80) and `mantis.run._step_coordinator_config` (renamed from
`_default_step_coordinator_config` because it no longer supplies the config-authored values
— R73 name-truth).

The oracles, and the defect each is the ONLY witness to:

- O-D2 `test_the_audited_value_IS_the_value_the_coordinator_runs_on` — the construction site
  not threading the RESOLVED value: the config says one thing, the runtime uses another, and
  the audit reads the config and goes green over a disarmed run. **Sole witness for "pinned
  text present, wrong value flowing"** (SF-2): a `source_pin` proves a character sequence is
  still in `run.py`, and `dataclasses.fields()` proves no default survives — a threading line
  and defaultless fields can BOTH be true while the wrong value flows.
- O-D8 `test_a_disarmed_threshold_skip_counts_and_never_raises_TypeError` — `step.py:421`
  left as a numeric comparison. `cfg.draw_rate_abort > 0` on the `None` posture raises
  `TypeError` inside `_run_hard_abort_gates` ONCE PER `step()`, on every disarmed run, so
  `is not None` is REQUIRED BY THE TYPE CHANGE rather than a tidy-up. Every other oracle here
  uses an armed config or never reaches `_run_hard_abort_gates`.
- O-D10 `test_all_THREE_block_keys_reach_their_runtime_destination` — each key observed AT
  ITS OWN CALL SITE. (WPMINT Phase K-B: the block has FOUR keys now — `consec` joined it by
  call K-b — and the name is kept because THREE names the oracle, not the arity; every key
  the block carries is observed at its own destination, which is the invariant.) WPMINT DR-6 (R93) fixed `_leaf_paths` to descend through
  `Block | None`, so the three inner keys DO carry a consumer-registry obligation now (they
  did not when this oracle was written). That obligation is a key-set bijection against a
  registry STRING, though: it proves someone wrote down a consumer, never that the value
  arrives. This oracle is still the only thing that watches `threshold` / `min_step` land on
  `check_draw_rate_collapse(...)` and `N_pool_min` decides the OBSERVATION BOUNDARY
  (WPMINT Phase DS re-point, R92 — the bar no longer travels through the pool).
  Not caught by O-D2, which asserts the spec object rather than the three call sites.

R7 / gate 6: nothing here writes a `*.jsonl`; every drive writes under `tmp_path`.

>300 justify (R8): one seam, one set of drivable fakes. `_Pool` / `_Trainer` / `_Buffer` /
`_SpySink` are the shared harness that BOTH `compose_run` (O-D2) and a directly-built
`StepCoordinator` (O-D8, O-D10) drive, and `_Pool` is the one fake this delta's Protocol
widening actually changes — splitting the file would fork it into two copies free to drift
apart in exactly the direction the delta moves. The remaining length is the per-oracle LAW-07
rationale and the R69 producer citations, not logic.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run
import mantis.train.coordinator.step as step_module

# NOTE (ORACLE-WRITE): `ruff --fix` at HEAD re-sorts the `resolve.draw_rate` import into the
# third-party block, because the module it names does not exist yet. It sits here, with its
# `mantis.*` siblings, which is where it belongs the moment IMPL lands it.
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

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_DRIVE_STEPS = 4
#: Deliberately NOT run5's `{0.25, 25000, 50}`. O-D2 asserts transport, and a harness that
#: drives the production values cannot distinguish "the config reached the coordinator" from
#: "the builder hardcodes the same numbers the config happens to carry".
_OFF_PREREG = {"threshold": 0.37, "min_step": 2, "N_pool_min": 7, "consec": 2}


# ── fakes (the `tests/test_run_strict_composition.py:113-205` shapes) ─────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    """The pool surface `compose_run` and `StepCoordinator` touch, plus a recorder on the
    ONE method whose shape this delta changes.

    WPMINT Phase DS (R92): the producer is `pooled_draw_counts() -> (draws, completed)` and
    it takes NO bar — the evidence bar moved to the abort decision. The recorder therefore
    counts CALLS rather than the argument values it used to see; the bar's own transport is
    observed BEHAVIOURALLY in O-D10 (where the observation boundary falls), which is a
    stronger pin than reading back an argument the fake was handed.
    """

    def __init__(self, *, counts: tuple[int, int] = (0, 0),
                 fresh_game_per_read: bool = True) -> None:
        self._games = 0
        self._fresh = fresh_game_per_read
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate, self.o_winrate, self.draws = 0.5, 0.45, 1
        self.sims_per_sec, self.batch_fill_pct = 100.0, 0.9
        self.recent_move_histories: list = []
        self.started = self.stopped = False
        self.n_workers = 1
        self._counts = (int(counts[0]), int(counts[1]))
        #: how many times the producer was CALLED — O-D10's disarmed-arm observation.
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

    # WPTS/TD-1 re-point (R90a): the dead `train_step` fake is gone — the double
    # conforms to the DECLARED seam (typed entry points + `device`).
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
    size, capacity = 1000, 100_000

    def resize(self, n: int) -> None:
        return None

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        # The grid route's sampler (WPTS dispatcher); rows are opaque to _Trainer.
        return (None,) * 9


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


def _bounded(factory, *, block, name: str = "smoke_gnn.yaml", steps: int = _DRIVE_STEPS):
    """A REAL minted config, bounded so a `compose_run` drive terminates, carrying `block`.

    The three step-clock knobs are co-overridden together because the reachability validator
    spans them (`tests/test_run_strict_composition.py:240-244`), and `min_step` must stay
    strictly under `max_train_steps` for the same reason the actor-lag threshold must —
    the twin cross-validator this delta adds to `schema/core.py` (§6.2).
    """
    return factory(name,
                   train={"actor_sync_cadence_steps": 1, "max_train_steps": steps,
                          "draw_rate_abort": block,
                          # WPTS/TD-1: the compose drive runs the real graph route.
                          "batch_size": 8},
                   monitor={"actor_lag_threshold_steps": steps - 1,
                            # WPAX ADJ-18 (operator-authorized R43 event). `smoke_gnn.yaml`
                            # ships actor-lag DISARMED by R59's deliberate smoke allowance, so
                            # once Phase D flips draw-rate to `required` the audit correctly
                            # reports BOTH required rows disarmed and the `== ["draw_rate_
                            # collapse"]` assertion below could never hold. Arming actor-lag
                            # HERE makes draw-rate the only disarmed row, which keeps that
                            # assertion EXACT (it still catches over-reporting) instead of
                            # weakening it to membership. The oracle's subject is its own name
                            # — the audited value is the coordinator's value — not an
                            # incidental census of who else happens to be disarmed.
                            "actor_lag_abort_enabled": True},
                   # WPMAIN/R120: both compose drives on this config ran `eval_enabled=False`
                   # and that is now a CONFIG fact, declared here — byte-preserved posture.
                   eval_enabled=False)


def _coordinator(*, config, pool, trainer=None):
    """A `StepCoordinator` on fakes — the `tests/train/test_coordinator_gates.py:174-186`
    shape. Direct construction, because O-D8 and O-D10 are about what `step()` does with a
    config, not about how one is composed (that is O-D2's subject)."""
    shutdown, sink = ShutdownState(), _SpySink()
    coord = StepCoordinator(
        trainer=trainer or _Trainer(), buffer=_Buffer(), pretrained_buffer=None,
        recent_buffer=None, pool=pool, eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        # WPTS/TD-1: unit drives declare the grid identity their _Buffer fake serves.
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}}, train_cfg={}, mixing_cfg={}, sink=sink,
        heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, sink=sink)


def _coordinator_config(spec, **overrides) -> StepCoordinatorConfig:
    """The production builder's own output with `draw_rate_abort` set — never a hand-written
    census of the ~22 knobs CARD-COORD-KNOBS still owns (R78). `dataclasses.replace` supplies
    the harness-only cadence knobs so the builder stays the single source of the rest.

    WPMINT Phase K-A (R93): `drain_caps` is a third config-authored builder parameter with
    no default, so it arrives here from a MINTED `monitor.drain` block for the same reason
    `stop_step`/`draw_rate_abort` do — a literal would be a second authority.
    """
    base = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=spec,
        drain_caps=resolve_drain_caps(load_config(_CONFIGS / "dev_example.yaml").monitor),
        knobs=resolve_coordinator_knobs(load_config(_CONFIGS / "dev_example.yaml").train))
    return dataclasses.replace(base, log_interval=1, eval_interval=1, min_buf_size=1,
                               terminal_eval_enabled=False, **overrides)


# ── O-D2 — the audited value IS the value the coordinator runs on ─────────────────────
def test_the_audited_value_IS_the_value_the_coordinator_runs_on(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """R79(3)'s named RED, on its behavioural side. The config can say `0.25` while the
    runtime uses something else — the resolved value never reaching the construction site —
    and the audit reads the CONFIG, so gate 12 goes green over a disarmed run.

    Three readers, asserted to agree on one fact: the schema block, the resolver, and
    `StepCoordinator.config` after a real `compose_run` (`step.py:122` stores
    `self.config = config`, so the runtime value is directly readable). The driven values are
    deliberately NOT run5's pre-registered ones — a builder that hardcoded `0.25 / 25000 / 50`
    would satisfy an oracle driven on run5's own numbers while reading nothing at all.

    Second arm: the explicitly disarmed posture must survive the whole composition as `None`,
    AND the same config must be reported disarmed by `audit_arming`. The two halves are the
    R79 statement — arming is a property of the resolved value, and the manifest asserts a
    condition over that same value.
    """
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


# ── O-D8 — the disarmed posture skip-counts and never raises ──────────────────────────
def test_a_disarmed_threshold_skip_counts_and_never_raises_TypeError() -> None:
    """`step.py:421` is `if draw and cfg.draw_rate_threshold > 0` at HEAD. Under the type
    change `cfg.draw_rate_abort` is `None` on every disarmed run, and `None > 0` raises
    `TypeError` inside `_run_hard_abort_gates` — ONCE PER `step()`, on every disarmed run.
    So `is not None` is REQUIRED BY THE TYPE CHANGE, not a tidy-up, and this is its only
    witness.

    The skip accounting (LAW-18) is asserted too, and it is asserted with a NON-EMPTY
    producer: a coordinator that never reached the gate would satisfy "no TypeError" while
    witnessing nothing, so `checks` must have advanced as well as `skips`. WPMINT DR-1: the
    skip is `_sample`'s and always was — the `elif draw:` arm this prose used to credit was
    provably unreachable and has been deleted (R72). The EXACT per-run counts live in
    `tests/train/test_drawrate_gate_branch_flipset.py`; the `>=` here is deliberate, because
    this oracle drives a whole `step()` rather than one gate run.
    """
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


# ── O-D10 — all THREE keys reach their own call site ──────────────────────────────────
def test_all_THREE_block_keys_reach_their_runtime_destination(monkeypatch) -> None:
    """WHAT LAW-08's bijection gate can and cannot say about this block. When this oracle
    was written the gate said NOTHING: `_leaf_paths` recursed only when `isinstance(ann,
    type) and issubclass(ann, BaseModel)`, and `DrawRateAbortConfig | None` is a
    `types.UnionType`, so the whole block was ONE registry leaf. WPMINT DR-6 (R93) closed
    that — the three inner keys are separate registry leaves now. What the gate still cannot
    say is whether the value ARRIVES: it is a key-set bijection against a hand-written
    consumer STRING, so an `N_pool_min` that reached nothing would pass it with the string
    intact.

    Each key is therefore observed AT ITS OWN DESTINATION: `threshold` and `min_step` at
    `check_draw_rate_collapse(...)`, and `N_pool_min` at the OBSERVATION BOUNDARY it decides.
    The expected values are read off `configs/run5.yaml` through the resolver — never written
    as literals here — so a delta that threads a constant into any one of the three fails.

    WPMINT Phase DS (R92) RE-POINTS the third key's arm, and to a stronger observation. The
    bar used to be handed to `pool.per_worker_draw_rates(min_samples=)`, so the fake could
    read it back as an argument — which proves the value was PASSED, not that it DECIDED
    anything. R92 moves the bar off the pool entirely (the pool reports raw counts), so the
    arm below drives the boundary instead: at `N_pool_min - 1` completed games there is NO
    OBSERVATION (nothing appended, nothing compared, a skip counted) and at exactly
    `N_pool_min` there is one. A delta that threaded a different number would move that
    boundary; a delta that threaded none could not produce it at all.
    """
    cfg = load_config(_CONFIGS / "run5.yaml")
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
    # WPMINT Phase K-B (call K-b) RE-POINTS this arm — the DELIBERATE BOUNDARY MARKER stays,
    # read from the other side. It said "`consec` still comes from the coordinator's own
    # code-side default", which was true while R78/R80 kept that term with CARD-COORD-KNOBS.
    # K-B IS CARD-COORD-KNOBS, so `consec` is the block's FOURTH authored term now, and the
    # marker's job — "do not leave the reader to guess which of these the config authors" —
    # is served by asserting it at the same call site as the other two, against the config's
    # own value. The coordinator dataclass no longer carries a `draw_rate_consec` field at
    # all, so this arm would be unwritable in its old form.
    assert call["consec"] == spec.consec == cfg.train.draw_rate_abort.consec, (
        f"train.draw_rate_abort.consec must reach `check_draw_rate_collapse(consec=)`; "
        f"config says {cfg.train.draw_rate_abort.consec!r}, the rule saw "
        f"{call.get('consec')!r}. It is a CONFIG term since WPMINT Phase K-B, not the "
        "coordinator's own default — that boundary is what this arm has always marked"
    )
    # `N_pool_min` at ITS destination: the observation boundary. Both sides of the boundary
    # are driven, because "no observation ever" and "observation always" each satisfy one
    # side alone, and the config's own number is what separates them.
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
    """The unit statement O-D10's numbers only mean something against, and the reason `le=1`
    is the right ceiling (MF-1). WPMINT Phase DS (R92): `pooled_draw_rate` is
    `Sum(draws)/Sum(completed)` over the union of worker windows, i.e. a fraction in
    `[0, 1]`, and `check_draw_rate_collapse`'s predicate is `all(value >= threshold)` — an
    UPPER bound. A threshold above 1.0 is therefore unreachable BY ARITHMETIC, not by
    convention, which is what makes `le=1` a bound on the metric rather than a policy.

    The empty arm is where R92 changed the answer, and the change is asserted rather than
    described: the retired `recent_pool_draw_rate({})` returned the fail-safe `0.0`, which
    DR-4 measured being APPENDED to the abort history as a real healthy measurement. It is
    `None` now — a different TYPE, so the two cases can no longer be confused.
    """
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
    """The harness's own vacuity guard. O-D10's disarmed arm asserts `counts_calls == 0`, so
    if the fake still carried the RETIRED `per_worker_draw_rates` the coordinator would find
    that attribute by `getattr`, take the live path against it, and the failure would surface
    somewhere other than here. Asserted so the fake cannot drift into agreement with the
    defect it exists to detect."""
    pool = _Pool(counts=(1, 4))
    assert pool.pooled_draw_counts() == (1, 4)
    assert pool.counts_calls == 1
    assert not hasattr(pool, "per_worker_draw_rates"), (
        "the retired producer must be GONE from the fake, not merely unused: `step.py` reaches "
        "it by `getattr(self.pool, ...)`, so a surviving attribute is a live call site"
    )
    with pytest.raises(TypeError):
        pool.pooled_draw_counts(min_samples=9)  # type: ignore[call-arg]
