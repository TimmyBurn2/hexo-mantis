"""A fired hard abort is distinguishable from a clean run.

The defect is narrow and exact: `shutdown.running = False` is written by FOUR sites in
`train/coordinator/step.py` — `stop()`, the O2 iteration limit, the O3 signal shutdown-save, and
`_fire_hard_abort` — and `ShutdownState` carried no field telling the fourth apart from the other
three, so a collapsed run and a completed run left the same state and the same exit status.

Each oracle is here because a different way of "fixing" that would still leave it broken: a fired
abort must record its rule THROUGH the armed gate, not only through a direct call nothing takes;
the exit code must READ THE MANIFEST, since a resolver answering from its own literal would agree
forever and be a second authority; all THREE clean stop sites must leave `abort_rule is None`; an
abort with no authored code must resolve to `None`, never a fabricated number; and the mutation
that suppresses the assignment must red the fired-abort oracle and NOT the clean-run one.

Only the trainer/buffer/pool collaborators are fakes, and none of them touches the abort decision.

>300 justify (R8): the mutation rows are only meaningful beside the oracle each must red and
the one it must not, and they share this file's StepCoordinator harness.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from mantis.config.armed_aborts import MANIFEST, ArmedAbort, Mechanism, Status, exit_code_for_abort
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from _drivable import DrivablePoolStub
from _graph_drive import DEV_DRAIN_CAPS, DEV_GATE_INTERVAL, DEV_KNOBS, GRAPH_FULL_CONFIG, GraphSampleBuffer, mirrored
from _monitor_config import monitor_config
from mantis.monitor.heartbeat import DRAW_RATE_COLLAPSE_EXIT_CODE
from mantis.run import _step_coordinator_config
from mantis.train.resume_state import sidecar_path_for
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState


RULE = "draw_rate_collapse"


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self.saves = 0
        # A REAL directory, because R343(c) made the O3 arm write a resume sidecar beside the
        # checkpoint it saves. A fake `checkpoint_dir` would have made the O3 row assert a
        # clean stop while the resumable leg went unexercised. Owned TemporaryDirectory.
        self._ckpt_tmp = tempfile.TemporaryDirectory(prefix="mantis-abort-exit-")
        self.checkpoint_dir = Path(self._ckpt_tmp.name)

    # WPTS/TD-1 re-point (R90a): the dead `train_step` fake is gone — the double
    # conforms to the DECLARED seam (typed entry points + `device`).
    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def save_checkpoint(self, loss_info) -> Path:
        self.saves += 1
        path = self.checkpoint_dir / f"fake_{self.step:08d}_deadbeef.ckpt"
        path.write_bytes(b"fake-checkpoint")
        return path


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder — this file's deltas only. `None` is the EXPLICIT
    disarmed draw-rate posture; neither the builder nor this factory gives it a default."""
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
                                 drain_caps=DEV_DRAIN_CAPS, gate_interval=DEV_GATE_INTERVAL,
                                 knobs=DEV_KNOBS),
        **mirrored({"eval_interval": 0, "log_interval": 1, "min_buf_size": 10,
                    **overrides}),
    )


class _Buffer(GraphSampleBuffer):
    """The shared double whose `save_to_path` writes REAL bytes: the resume sidecar hashes this
    file, so a no-op would make the hash a hash of nothing and the identity witness vacuous."""

    def save_to_path(self, p) -> None:
        Path(p).write_bytes(b"fake-ring" * 8)


def _coordinator(*, pool=None, config=None, shutdown=None):
    pool = pool or DrivablePoolStub()
    trainer, buffer, sink = _Trainer(), _Buffer(), _Sink()
    shutdown = shutdown if shutdown is not None else ShutdownState()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(),
        config=config or _config(), full_config=GRAPH_FULL_CONFIG,
        sink=sink, heartbeat=None, monitor_cfg=monitor_config(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer,
                           shutdown=shutdown, sink=sink)


def test_a_fired_hard_abort_records_the_rule_on_the_shutdown_state() -> None:
    """The REAL `_fire_hard_abort` on a REAL coordinator: the fire stops the run AND records WHICH
    rule stopped it. A fresh `ShutdownState` starting at `abort_rule is None` is asserted rather
    than assumed — a field that started life set would make every later assertion vacuous."""
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
    """The PRODUCTION path, not a direct method call: a real spec and a pool reporting 0.9 against
    a 0.25 bar, so the assertion covers the chain a real run takes from `_check_draw_rate` through
    the rule to `ShutdownState`."""
    spec = DrawRateAbortSpec(threshold=0.25, min_step=0, N_pool_min=50, consec=3)
    h = _coordinator(pool=DrivablePoolStub(draw_counts=(900, 1000)),
                     config=_config(draw_rate_abort=spec, policy_loss_trough_abort=None, ply_cap_abort=None, log_interval=1))
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


def test_the_resolver_answers_46_for_the_draw_rate_rule() -> None:
    """Pinned against the CONSTANT: an equality against a bare `46` would be satisfied just as
    well by a hand-typed number in the resolver."""
    assert exit_code_for_abort(RULE) == DRAW_RATE_COLLAPSE_EXIT_CODE == 46


def test_the_resolver_reads_the_manifest_and_has_no_literal_of_its_own() -> None:
    """The manifest row is replaced with one carrying a DIFFERENT code and the resolver must follow
    it — the difference between "the resolver agrees with the manifest today" and "the manifest is
    the authority"."""
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
    """A synthetic row with a name the resolver has never heard of must resolve, because the lookup
    is by data. A resolver that special-cased the known rule would pass both assertions above."""
    invented = ArmedAbort(
        name="a_rule_this_repo_has_never_seen", config_path="monitor.actor_lag_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.REQUIRED, exit_code=77,
        owner=None, source_pin=None, note="oracle probe",
    )
    assert exit_code_for_abort(invented.name, manifest=(*MANIFEST, invented)) == 77


def test_the_O2_iteration_limit_is_a_clean_stop() -> None:
    """The iteration-limit stop sets `running = False` exactly as an abort does, so it is the site
    an abort is most easily confused with — and it must leave `abort_rule` untouched."""
    h = _coordinator(config=_config(stop_step=0))
    h.coord.step()

    assert h.shutdown.running is False, "O2 stops the run when the step ceiling is reached"
    assert h.shutdown.abort_rule is None, (
        "a run that COMPLETED is not a run that aborted; `abort_rule` is the only thing that "
        f"says which happened, and it must stay None here (got {h.shutdown.abort_rule!r})"
    )


def test_the_O3_shutdown_save_is_a_clean_stop() -> None:
    """A signal flipped `shutdown_save`, the loop saved and stopped. Covered separately because it
    is a SEPARATE write of `running = False`; pinning one clean site leaves half the claim
    unmeasured."""
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
    # The ordered stop is RESUMABLE, and the sidecar beside the checkpoint is what makes it so:
    # without it the next launch reads this checkpoint as a warm start and refills the ring empty.
    ckpt = h.trainer.checkpoint_dir / "fake_00000000_deadbeef.ckpt"
    side = sidecar_path_for(ckpt)
    assert side.exists(), (
        "O3 saved a checkpoint but wrote no resume sidecar — the stop is not resumable, and "
        "R343(c) forbids the resume that would follow"
    )
    state = json.loads(side.read_text(encoding="utf-8"))
    # The ring is named for ITS OWN checkpoint, not a single canonical path every save
    # overwrote — that one path is what made retaining a previous bundle impossible.
    assert state["ring"]["path"].endswith(ckpt.name + ".ring.bin"), state["ring"]["path"]
    manifest = Path(str(ckpt) + ".bundle.json")
    assert manifest.exists(), (
        "the stop wrote a sidecar and a ring but no manifest, so nothing commits the set and "
        "a resume cannot tell a complete bundle from a torn one"
    )
    assert state["ring"]["sha256"] == hashlib.sha256(b"fake-ring" * 8).hexdigest(), (
        "the sidecar must hash the ring it actually persisted, not record a placeholder"
    )


def test_a_signal_handler_stop_is_a_clean_stop() -> None:
    """The third clean write, included because it is the site a future reader is likeliest to add a
    rule to by analogy: `stop()` takes a `reason` STRING, which looks exactly like a rule name and
    is not one."""
    h = _coordinator()
    h.coord.stop("operator asked for it")
    assert h.shutdown.running is False and h.shutdown.abort_rule is None


@pytest.mark.parametrize(("rule", "has_row"), [("grad_norm_hard_abort", True),
                                              ("an_abort_rule_with_no_manifest_row", False)])
def test_an_abort_with_no_authored_code_resolves_to_None(rule: str, has_row: bool) -> None:
    """A live rule whose row carries no code, and a name with NO row, both resolve to None.

    Every fired rule has a row, so an unregistered name witnesses the no-row branch; a default
    code would fabricate a number, and a raise would crash an un-carded abort at the boundary."""
    row = [candidate for candidate in MANIFEST if candidate.name == rule]
    assert bool(row) is has_row, f"premise: {rule!r} {'has' if has_row else 'has no'} manifest row"
    assert all(candidate.exit_code is None for candidate in row), (
        "the premise: this rule has NO AUTHORED EXIT CODE. A row may exist for it — "
        "grad_norm_hard_abort gained a DEFERRED one at WPMINT Phase K-B — but a code "
        "appearing without a card is the class R84 refused, and inventing one at the "
        f"resolver is that class one layer down; got {[c.exit_code for c in row]}"
    )
    assert exit_code_for_abort(rule) is None


def test_a_registered_row_carrying_no_code_also_resolves_to_None() -> None:
    """The other source of `None`, kept distinct on purpose: `None` means "no authored exit code",
    never "no abort fired". Only `ShutdownState.abort_rule is None` means that, and conflating the
    two would read every un-coded abort as a clean run."""
    uncoded = ArmedAbort(
        name="a_registered_rule_with_no_code", config_path="monitor.actor_lag_abort_enabled",
        mechanism=Mechanism.CONFIG_BOOL, status=Status.REQUIRED, exit_code=None,
        owner=None, source_pin=None, note="oracle probe",
    )
    assert exit_code_for_abort(uncoded.name, manifest=(*MANIFEST, uncoded)) is None


def test_MUTATION_suppressing_the_assignment_makes_an_abort_look_like_a_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutation is the defect itself, restored: a `_fire_hard_abort` that stops the run and
    does NOT record the rule. Both directions in one place, and the mutation is not
    self-satisfying — it kills the production assignment, not a test helper."""
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
    """`ShutdownState` carries a rule NAME and not an exit code, so the coordinator never has to
    know which number an abort exits with. An import of `armed_aborts` into the step slice would
    give the number a second home, and gate 9's DAG check would not object because the edge is
    legal."""
    src = __import__("mantis.train.coordinator.step", fromlist=["x"]).__file__
    with open(src, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    # An IMPORT scan at any depth, not a token scan over the text: the file's own comments name
    # `armed_aborts` (they explain why it is NOT imported), and a token scan would be satisfied by
    # deleting the explanation. Function-body imports count — a lazy import is still the coupling.
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
