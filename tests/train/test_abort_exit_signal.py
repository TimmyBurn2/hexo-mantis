"""CARD-ABORT-EXIT (R84) — a fired hard abort is distinguishable from a clean run.

WPMINT Phase X. The defect the card names is narrow and exact: `shutdown.running = False` is
written by FOUR sites in `train/coordinator/step.py` — `stop()`, the O2 iteration limit
(clean completion), the O3 signal shutdown-save, and `_fire_hard_abort` — and until this card
landed `ShutdownState` carried no field that told the fourth apart from the second and third.
A run that collapsed on the draw-rate abort and a run that finished its last step left the
same state and exited with the same status, so no supervisor could tell them apart.

Six oracles, and each one is here because a different way of "fixing" that would still leave
it broken:

* **X-1** — a fired draw-rate abort records `abort_rule="draw_rate_collapse"`. Driven through
  the REAL `_fire_hard_abort` on a REAL `StepCoordinator` (R84 is explicit that a stand-in
  does not discharge this), and additionally through the whole armed gate, so the assertion
  covers the production path rather than a direct call nothing takes.
* **X-2** — `exit_code_for_abort("draw_rate_collapse") == 46` **and it reads the manifest**.
  The second half is the load-bearing one: the resolver is mutated against a synthetic
  manifest and must follow it. A resolver that answered `46` from its own literal would pass
  the first half forever and be a second authority.
* **X-3** — a CLEAN run leaves `abort_rule is None`, at ALL THREE clean stop sites. One of
  them is not enough: `stop()`, O2 and O3 are separate writes of `running = False`, and
  pinning one leaves the others free to grow an abort_rule they have no business having.
* **X-4** — an abort with no authored code resolves to `None`, never to a fabricated number.
  R84 refused to invent a code for `grad_norm_hard_abort` / `sealbot_wr_abort`; this is what
  stops the refusal being undone one layer down.
* **X-5** — the MUTATION R84 requires: with the `abort_rule` assignment suppressed, X-1 reds
  and the clean-run oracle does NOT (R81 under its R86 reading — not self-satisfying, no
  unrelated casualty). Without this the whole file could be satisfied by a field that is
  written unconditionally.
* **X-6** lives in `tests/tools/test_preflight_mint_process.py`, where the rc taxonomy it
  drives already lives (R5 bars cross-test imports, and the tool's loader is set up there).

The harness below is local by necessity — R5 bars importing `test_coordinator_gates`'s fakes —
and deliberately minimal: everything the card asserts on is real. The `StepCoordinator`, the
`ShutdownState`, `_fire_hard_abort`, the draw-rate gate and the resolver are all production
objects; only the trainer/buffer/pool collaborators are fakes, and none of them touches the
abort decision.

>300 justify (R8), at the file's MEASURED size of 440 lines (Phase X wrote 420; K-A's
coordinator-census consolidation restated 428; WPMINT Phase K-B's `resolve_coordinator_knobs`
harness threading is the 428 -> 440 delta, restated per SF-7). Two components, and neither is
splittable without losing what it is for. (1) ~110 lines are the local `StepCoordinator`
harness, which exists ONLY because R5 bars cross-test imports — the alternative is a shared
fixture module, which is the collection-shadowing shape R5 forbids. (2) The rest is one card's
six oracles held together on purpose: X-5's mutation is only meaningful beside the X-1 it must
red and the X-3 it must NOT, and splitting them across files puts "the proof" and "the thing
proved" where the next reader cannot see them in one screen. Roughly half the remaining volume
is the per-oracle "what wrong fix does this catch" rationale LAW-07 wants on a gate input.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mantis.config.armed_aborts import MANIFEST, ArmedAbort, Mechanism, Status, exit_code_for_abort
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import DRAW_RATE_COLLAPSE_EXIT_CODE
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

RULE = "draw_rate_collapse"

#: WPMINT Phase K-A stage 0: the four drain caps are `monitor.drain.*` (R93/DR-11) — read
#: from a MINTED config, never restated here.
_DRAIN_CAPS = resolve_drain_caps(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").monitor)
#: WPMINT Phase K-B: the builder's fourth config-authored parameter, from the same minted
#: config — the 19 coordinator knobs are `train.*` keys now, not builder literals.
_KNOBS = resolve_coordinator_knobs(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train)


# ── the minimum real-coordinator harness ──────────────────────────────────────────────
class _Pool:
    """A pool whose draw counts are the ONLY thing the card's gate reads off it."""

    def __init__(self, *, draws: int = 0, completed: int = 0) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = draws
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self._counts = (int(draws), int(completed))

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return self._counts

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return SimpleNamespace(mcts_mean_depth=5.0, mcts_mean_root_concentration=0.1,
                               cluster_value_std_mean=0.0,
                               cluster_policy_disagreement_mean=0.0,
                               cluster_variance_sample_count=0)

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.saves = 0

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def save_checkpoint(self, loss_info) -> None:
        self.saves += 1


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder (WPMINT Phase K-A stage 0) — this file's deltas
    only. `None` is the EXPLICIT disarmed draw-rate posture; the builder gives it no
    default and neither does this factory."""
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, knobs=_KNOBS),
        **{"eval_interval": 0, "log_interval": 1, "min_buf_size": 10, **overrides},
    )


def _coordinator(*, pool=None, config=None, shutdown=None):
    pool = pool or _Pool()
    trainer, buffer, sink = _Trainer(), _Buffer(), _Sink()
    shutdown = shutdown if shutdown is not None else ShutdownState()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None,
        config=config or _config(), full_config={}, train_cfg={}, mixing_cfg={},
        sink=sink, heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer,
                           shutdown=shutdown, sink=sink)


# ══ X-1 — a fired abort names itself ═══════════════════════════════════════════════════
def test_a_fired_hard_abort_records_the_rule_on_the_shutdown_state() -> None:
    """X-1 — the REAL `_fire_hard_abort` (R84: "not a stand-in"), on a REAL coordinator.

    Two claims in one drive, and the second is the card's: the fire stops the run (which it
    always did) AND records WHICH rule stopped it (which is the whole delta). A fresh
    `ShutdownState` starts at `abort_rule is None`, asserted here rather than assumed — a
    field that started life set would make every later assertion in this file vacuous.
    """
    h = _coordinator()
    assert h.shutdown.abort_rule is None, "a fresh ShutdownState carries no fired rule"

    fired = h.coord._fire_hard_abort(RULE, "pooled draw rate 0.91 over 3 consecutive checks")

    assert fired is True, "a fire with a message returns True (the shared contract)"
    assert h.shutdown.running is False, "the fire stops the run — unchanged behaviour"
    assert h.shutdown.abort_rule == RULE, (
        "R84: a fired abort must be distinguishable from a clean run, and the carrier is the "
        f"RULE NAME on the shutdown state; got {h.shutdown.abort_rule!r}"
    )


def test_the_armed_gate_fires_through_the_same_contract_and_names_the_rule() -> None:
    """X-1, through the PRODUCTION path rather than a direct method call.

    A direct `_fire_hard_abort` call proves the contract; it does not prove the draw-rate gate
    reaches it. This drives the armed gate off a real `DrawRateAbortSpec` and a pool reporting
    a collapse (900 draws / 1000 completed = 0.9, well over the 0.25 bar), so the assertion
    covers the chain a real run takes: `_check_draw_rate` -> `_sample` -> the rule ->
    `_fire_hard_abort` -> `ShutdownState`.
    """
    spec = DrawRateAbortSpec(threshold=0.25, min_step=0, N_pool_min=50, consec=3)
    h = _coordinator(pool=_Pool(draws=900, completed=1000),
                     config=_config(draw_rate_abort=spec, log_interval=1))
    for _ in range(12):
        if not h.shutdown.running:
            break
        h.pool.games_completed += 5
        h.coord.step()

    assert h.shutdown.running is False, "a 0.9 pooled draw rate must stop the run"
    assert h.shutdown.abort_rule == RULE, (
        f"the gate must reach the shared fire contract and name itself; got "
        f"{h.shutdown.abort_rule!r}"
    )
    assert [e["rule"] for e in h.sink.named("hard_abort")] == [RULE], (
        "and the event trail is unchanged — the card adds a PROCESS signal, it does not "
        "replace the event one"
    )


# ══ X-2 — the code comes from the manifest, not from a literal ═════════════════════════
def test_the_resolver_answers_46_for_the_draw_rate_rule() -> None:
    """X-2, first half. Pinned against the CONSTANT: an equality against a bare `46` would be
    satisfied just as well by a hand-typed number in the resolver, which is what X-2's second
    half exists to rule out."""
    assert exit_code_for_abort(RULE) == DRAW_RATE_COLLAPSE_EXIT_CODE == 46


def test_the_resolver_reads_the_manifest_and_has_no_literal_of_its_own() -> None:
    """X-2, second half — the one that can actually fail.

    The manifest row is replaced with one carrying a DIFFERENT code and the resolver must
    follow it. If `exit_code_for_abort` held its own `46` — or branched on the rule's name —
    this reds. That is the difference between "the resolver agrees with the manifest today"
    and "the manifest is the authority".
    """
    rewired = tuple(
        dataclasses.replace(row, exit_code=99) if row.name == RULE else row
        for row in MANIFEST
    )
    assert exit_code_for_abort(RULE, manifest=rewired) == 99, (
        "the resolver must ANSWER FROM THE ROW; a literal of its own would answer 46 here "
        "and the manifest would have stopped being the single authority"
    )
    assert exit_code_for_abort(RULE) == DRAW_RATE_COLLAPSE_EXIT_CODE, (
        "and the synthetic manifest must not have leaked into the shipped one"
    )


def test_the_resolver_never_branches_on_a_rules_identity() -> None:
    """X-2, third half — the shape, not just the answer.

    A synthetic row with a name the resolver has never heard of must resolve, because the
    lookup is by data. A resolver that special-cased `draw_rate_collapse` would pass both
    assertions above and fail this one, which is the same discrimination `Mechanism.is_armed`
    is written to survive.
    """
    invented = ArmedAbort(
        name="a_rule_this_repo_has_never_seen", config_path="monitor.actor_lag_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.REQUIRED, exit_code=77,
        owner=None, source_pin=None, note="oracle probe",
    )
    assert exit_code_for_abort(invented.name, manifest=(*MANIFEST, invented)) == 77


# ══ X-3 — a clean run leaves the field alone, at ALL THREE clean stop sites ════════════
def test_the_O2_iteration_limit_is_a_clean_stop() -> None:
    """X-3, site one (`step.py`'s O2 arm): the run reached `stop_step` and stopped. It set
    `running = False` exactly as an abort does, so it is the site an abort is most easily
    confused with — and it must leave `abort_rule` untouched."""
    h = _coordinator(config=_config(stop_step=0))
    h.coord.step()

    assert h.shutdown.running is False, "O2 stops the run when the step ceiling is reached"
    assert h.shutdown.abort_rule is None, (
        "a run that COMPLETED is not a run that aborted; `abort_rule` is the only thing that "
        f"says which happened, and it must stay None here (got {h.shutdown.abort_rule!r})"
    )


def test_the_O3_shutdown_save_is_a_clean_stop() -> None:
    """X-3, site two (`step.py`'s O3 arm): a SIGINT/SIGTERM flipped `shutdown_save`, the loop
    saved and stopped. Covered separately from O2 because it is a SEPARATE write of
    `running = False` — pinning one clean site and not the other leaves half the claim
    unmeasured, which is exactly what the card's own §5 calls out."""
    h = _coordinator(shutdown=ShutdownState(shutdown_save=True))
    outcome = h.coord.step()

    assert h.shutdown.running is False and outcome.checkpoint_saved is True, (
        "O3 saves a checkpoint and stops (LAW-16 save-then-exit)"
    )
    assert h.trainer.saves == 1, "and the save is the real one, not skipped past"
    assert h.shutdown.abort_rule is None, (
        "an operator-requested shutdown is a CLEAN stop; got "
        f"{h.shutdown.abort_rule!r}"
    )


def test_a_signal_handler_stop_is_a_clean_stop() -> None:
    """X-3, the third clean write (`stop()`), included because it is the site a future reader
    is likeliest to add a rule to by analogy. `stop()` takes a `reason` STRING, which looks
    exactly like a rule name and is not one."""
    h = _coordinator()
    h.coord.stop("operator asked for it")
    assert h.shutdown.running is False and h.shutdown.abort_rule is None


# ══ X-4 — no code is invented for an unregistered abort ════════════════════════════════
@pytest.mark.parametrize("rule", ["grad_norm_hard_abort", "sealbot_wr_abort"])
def test_an_abort_with_no_authored_code_resolves_to_None(rule: str) -> None:
    """X-4. Both rules share `_fire_hard_abort` with the draw-rate gate and NEITHER is
    pre-registered in the manifest. R84 ratified the refusal to invent a code for an abort
    nobody registered; `None` is that refusal, expressed as a value.

    Parametrized over both because they fail differently under the tempting wrong fix: a
    resolver with a `dict.get(rule, SOME_DEFAULT)` would hand both the same fabricated number,
    and a resolver that raised would turn an un-carded abort into a crash at the boundary.
    """
    # WPMINT Phase K-B RE-POINTS the premise, exactly as this assertion's own text
    # instructed. `grad_norm_hard_abort` HAS a manifest row now (a DEFERRED one, call K-c) —
    # what it still has not got is an authored exit code, and that is the subject. The two
    # were the same fact when this was written and are not any more, so the premise says the
    # narrower thing it always meant.
    row = [candidate for candidate in MANIFEST if candidate.name == rule]
    assert all(candidate.exit_code is None for candidate in row), (
        "the premise: this rule has NO AUTHORED EXIT CODE. A row may exist for it — "
        "grad_norm_hard_abort gained a DEFERRED one at WPMINT Phase K-B — but a code "
        "appearing without a card is the class R84 refused, and inventing one at the "
        f"resolver is that class one layer down; got {[c.exit_code for c in row]}"
    )
    assert exit_code_for_abort(rule) is None


def test_a_registered_row_carrying_no_code_also_resolves_to_None() -> None:
    """X-4's other source of `None`, kept distinct on purpose. `None` means "no authored exit
    code" — it never means "no abort fired". Only `ShutdownState.abort_rule is None` means
    that, and conflating the two would read every un-coded abort as a clean run."""
    uncoded = ArmedAbort(
        name="a_registered_rule_with_no_code", config_path="monitor.actor_lag_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.REQUIRED, exit_code=None,
        owner=None, source_pin=None, note="oracle probe",
    )
    assert exit_code_for_abort(uncoded.name, manifest=(*MANIFEST, uncoded)) is None


# ══ X-5 — the mutation R84 requires ════════════════════════════════════════════════════
def test_MUTATION_suppressing_the_assignment_makes_an_abort_look_like_a_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X-5 — R84's explicit requirement, driven rather than asserted.

    The mutation is the defect itself, restored: a `_fire_hard_abort` that stops the run and
    does NOT record the rule. Under it a fired abort and a clean run are the same observable,
    which is precisely what the card exists to make impossible — so this test measures BOTH
    directions in one place:

    * with the assignment suppressed, a fired abort is indistinguishable from a clean stop
      (`running is False`, `abort_rule is None`) — the state X-1 asserts against is gone;
    * with the real method, the two are distinguishable.

    R81 under its R86 reading: the mutation is not self-satisfying (it kills the production
    assignment, not a test helper), and its casualty is in-subject — the file's own X-1 nodes.
    The named-file evidence lives in IMPL_NOTES_X.md §7.
    """
    real = StepCoordinator._fire_hard_abort

    def _without_the_assignment(self, rule, message, step=None):
        rc = real(self, rule, message, step)
        self.shutdown.abort_rule = None      # the suppression
        return rc

    h = _coordinator()
    monkeypatch.setattr(StepCoordinator, "_fire_hard_abort", _without_the_assignment)
    assert h.coord._fire_hard_abort(RULE, "collapse") is True
    assert h.shutdown.running is False and h.shutdown.abort_rule is None, (
        "THE DEFECT: with the assignment suppressed a collapsed run leaves exactly the state "
        "a completed run leaves — no supervisor can tell them apart. This is what X-1 fails "
        "on, and it is the reason the assignment is not optional"
    )

    monkeypatch.undo()
    clean = _coordinator()
    clean.coord.stop("finished")
    fired = _coordinator()
    fired.coord._fire_hard_abort(RULE, "collapse")
    assert (clean.shutdown.running, clean.shutdown.abort_rule) == (False, None)
    assert (fired.shutdown.running, fired.shutdown.abort_rule) == (False, RULE), (
        "and with the real method the two runs are distinguishable — the card, discharged"
    )


def test_the_rule_name_carrier_keeps_the_train_layer_free_of_the_manifest() -> None:
    """The design's load-bearing choice, pinned where it can rot (DESIGN_X §3.2).

    `ShutdownState` carries a rule NAME and not an exit code, so that the coordinator never
    has to know which number an abort exits with — the mapping lives beside the manifest. If
    a later change imports `armed_aborts` into the step slice, the number acquires a second
    home in the layer that is meant not to know it, and gate 9's DAG check would not object
    because the edge is legal. This is what objects.
    """
    src = __import__("mantis.train.coordinator.step", fromlist=["x"]).__file__
    with open(src, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    # An IMPORT scan, at any depth, not a token scan over the text: the file's own comments
    # name `armed_aborts` (they have to — they explain why it is NOT imported), and a token
    # scan would be satisfied by deleting the explanation. Function-body imports are included
    # because a lazy import is still the coupling this pins against.
    imported = {
        name
        for node in ast.walk(tree)
        for name in (
            [a.name for a in node.names] if isinstance(node, ast.Import)
            else [node.module or ""] if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert not any("armed_aborts" in name for name in imported), (
        "train/coordinator/step.py must not import the armed-abort manifest: it records WHAT "
        f"fired, and the rule -> exit-code resolution belongs at the process boundary. Got "
        f"{sorted(n for n in imported if 'armed_aborts' in n)}"
    )
    assert dataclasses.fields(ShutdownState)[-1].name == "abort_rule", (
        "and the carrier is the LAST field, so it stays keyword-compatible with every "
        "positional `ShutdownState(...)` construction in the tree"
    )
