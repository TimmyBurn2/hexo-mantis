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
- O-D10 `test_all_THREE_block_keys_reach_their_runtime_destination` — the block's inner keys
  are INVISIBLE to the consumer-registry bijection: `_leaf_paths`
  (`test_every_key_has_consumer.py:206`) tests `isinstance(ann, type)` and
  `Optional[BlockModel]` is not a `type`, so `train.draw_rate_abort` is ONE leaf and a
  `min_samples` that reached nothing would still pass gate-level LAW-08. Not caught by O-D2,
  which asserts the spec object rather than the three call sites.

R7 / gate 6: nothing here writes a `*.jsonl`; every drive writes under `tmp_path`.

>300 justify (R8): one seam, one set of drivable fakes. `_Pool` / `_Trainer` / `_Buffer` /
`_SpySink` are ~90 lines of harness that BOTH `compose_run` (O-D2) and a directly-built
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
from mantis.config.resolve.draw_rate import (  # RED anchor (R80) — the ONE read path
    DrawRateAbortSpec,
    resolve_draw_rate_abort,
)
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config  # RED anchor — the renamed builder (R73)
from mantis.train.coordinator.config import StepCoordinatorConfig, recent_pool_draw_rate
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_DRIVE_STEPS = 4
#: Deliberately NOT run5's `{0.25, 25000, 50}`. O-D2 asserts transport, and a harness that
#: drives the production values cannot distinguish "the config reached the coordinator" from
#: "the builder hardcodes the same numbers the config happens to carry".
_OFF_PREREG = {"threshold": 0.37, "min_step": 2, "min_samples": 7}


# ── fakes (the `tests/test_run_strict_composition.py:113-205` shapes) ─────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    """The pool surface `compose_run` and `StepCoordinator` touch, plus a recorder on the
    ONE method whose signature this delta widens."""

    def __init__(self, *, draw_rates: dict[int, float] | None = None,
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
        self._draw_rates = dict(draw_rates or {})
        #: every `min_samples` this method was called with, in order — O-D10's observation.
        self.min_samples_seen: list[int] = []

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

    def per_worker_draw_rates(self, *, min_samples: int) -> dict[int, float]:
        self.min_samples_seen.append(int(min_samples))
        return dict(self._draw_rates)

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
        self.inference_sd = {"w": "SENTINEL"}

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

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
                          "draw_rate_abort": block},
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
                            "actor_lag_abort_enabled": True})


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
        full_config={}, train_cfg={}, mixing_cfg={}, sink=sink,
        heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, sink=sink)


def _coordinator_config(spec, **overrides) -> StepCoordinatorConfig:
    """The production builder's own output with `draw_rate_abort` set — never a hand-written
    census of the ~22 knobs CARD-COORD-KNOBS still owns (R78). `dataclasses.replace` supplies
    the harness-only cadence knobs so the builder stays the single source of the rest."""
    base = _step_coordinator_config(stop_step=10**9, draw_rate_abort=spec)
    return dataclasses.replace(base, log_interval=1, eval_interval=1, min_buf_size=1,
                               terminal_eval_enabled=False, **overrides)


# ── O-D2 — the audited value IS the value the coordinator runs on ─────────────────────
def test_the_audited_value_IS_the_value_the_coordinator_runs_on(
    tmp_path, monkeypatch, smoke_run_config,
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
        config=cfg, trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
    )
    runtime = handles.coordinator.config.draw_rate_abort
    assert runtime is not None, (
        "an ARMED config reached the coordinator as `None` — the composition root is not "
        "threading the resolved value, which is exactly the state where the audit reads "
        "0.37 from the config and the run aborts on nothing"
    )
    for key in ("threshold", "min_step", "min_samples"):
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
        config=disarmed_cfg, trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
        log_dir=str(tmp_path / "off"), checkpoint_dir=str(tmp_path / "off_ckpt"),
        eval_enabled=False,
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

    The `elif draw:` skip arm (LAW-18) is asserted too, and it is asserted with a NON-EMPTY
    producer: a coordinator that never reached the gate would satisfy "no TypeError" while
    witnessing nothing, so `checks` must have advanced as well as `skips`.
    """
    pool = _Pool(draw_rates={0: 0.99})
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
            DrawRateAbortSpec(threshold=0.4, min_step=0, min_samples=1)),
        pool=_Pool(draw_rates={0: 0.9, 1: 0.9}))
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
            DrawRateAbortSpec(threshold=0.4, min_step=10**9, min_samples=1)),
        pool=_Pool(draw_rates={0: 0.9, 1: 0.9}))
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
    """LAW-08 does not reach inside this block, and that is measured, not feared:
    `_leaf_paths` (`test_every_key_has_consumer.py:206`) recurses only when
    `isinstance(ann, type) and issubclass(ann, BaseModel)`, and `Optional[DrawRateAbortConfig]`
    is not a `type`. So `train.draw_rate_abort` is ONE registry leaf and a `min_samples` that
    reached nothing would pass the bijection gate.

    Each key is therefore observed AT ITS OWN CALL SITE: `threshold` and `min_step` at
    `check_draw_rate_collapse(...)`, `min_samples` at `pool.per_worker_draw_rates(...)`. The
    expected values are read off `configs/run5.yaml` through the resolver — never written as
    literals here — so a delta that threads a constant into any one of the three fails.
    """
    cfg = load_config(_CONFIGS / "run5.yaml")
    spec = resolve_draw_rate_abort(cfg.train)
    seen: list[dict] = []

    def _spy(history, current_step, **kwargs):
        seen.append(dict(kwargs))
        return None

    monkeypatch.setattr(step_module, "check_draw_rate_collapse", _spy)
    pool = _Pool(draw_rates={0: 0.9})
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
    assert call["consec"] == h.coord.config.draw_rate_consec, (
        "`consec` still comes from the coordinator's own code-side default — R78 keeps it "
        "with CARD-COORD-KNOBS, and this arm pins that boundary rather than leaving the "
        "reader to guess which of the four the config authors"
    )
    assert pool.min_samples_seen and set(pool.min_samples_seen) == {spec.min_samples}, (
        f"train.draw_rate_abort.min_samples must reach "
        f"`pool.per_worker_draw_rates(min_samples=)` on EVERY call; config says "
        f"{cfg.train.draw_rate_abort.min_samples!r}, the pool saw {pool.min_samples_seen!r}. "
        "A default at that seam would re-create the ADJ-14 saturation the moment any caller "
        "omitted it (R1, and MF-2's lesson one seam over)"
    )

    disarmed = _coordinator(config=_coordinator_config(None), pool=_Pool(draw_rates={0: 0.9}))
    disarmed.pool._games = 5
    disarmed.coord.step()
    assert disarmed.pool.min_samples_seen == [], (
        "on the disarmed posture `rates_fn` must never be CALLED: there is no `min_samples` "
        "to pass, and calling it with a stand-in would put a second authority on the axis "
        "the config just declined to arm (§4.5's disarmed path — `_sample` takes its "
        "producer-absent arm and SKIP-counts)"
    )


def test_the_pool_mean_is_the_thing_the_threshold_is_compared_against() -> None:
    """The unit statement O-D10's numbers only mean something against, and the reason `le=1`
    is the right ceiling (MF-1). `recent_pool_draw_rate` is an UNWEIGHTED MEAN of per-worker
    rates (`coordinator/config.py:141-146`), i.e. a fraction in `[0, 1]`, and
    `check_draw_rate_collapse`'s predicate is `all(value >= threshold)` — an UPPER bound
    (`rules.py:264`). A threshold above 1.0 is therefore unreachable BY ARITHMETIC, not by
    convention, which is what makes `le=1` a bound on the metric rather than a policy.
    """
    assert recent_pool_draw_rate({}) == 0.0, (
        "the empty arm is the fail-safe direction: while no worker qualifies the pool rate "
        "is BELOW every legal threshold, so the gate cannot fire on absent signal"
    )
    assert recent_pool_draw_rate({0: 1.0, 1: 0.0}) == 0.5
    assert recent_pool_draw_rate({w: 1.0 for w in range(8)}) == 1.0, (
        "1.0 is the metric's maximum at every worker count — no legal input can exceed it, "
        "so any threshold > 1.0 is 'armed in the config, absent in effect'"
    )


def test_the_harness_pool_refuses_the_call_shape_this_delta_retires() -> None:
    """The harness's own vacuity guard. O-D10 reads `min_samples` off `_Pool`, so if the fake
    silently accepted the OLD zero-argument call, a delta that never threaded `min_samples`
    would show up as `min_samples_seen == []` on the armed arm and could be mistaken for the
    disarmed arm's expected silence. Asserted here so the fake cannot drift into agreement
    with the defect it exists to detect."""
    pool = _Pool(draw_rates={3: 0.25})
    assert pool.per_worker_draw_rates(min_samples=9) == {3: 0.25}
    assert pool.min_samples_seen == [9]
    with pytest.raises(TypeError):
        pool.per_worker_draw_rates()  # type: ignore[call-arg]
