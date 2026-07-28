"""⊕ WPAX Phase S ORACLE — the STRICT composition root (DESIGN_S §6).

RED-at-import until IMPL lands `mantis.config.resolve.composition` (the gate:
`UnvalidatedConfigError` + `require_run_config`) and `mantis.config.resolve.run_length`
(`resolve_max_train_steps`). Both module imports below are the RED anchor; every oracle in
this file rides on them, exactly as `tests/test_run_composition.py` rides on `mantis.run`.

What this file exists to stop, in one sentence: `compose_run` currently duck-types its own
config — `getattr(config, "monitor", None) -> MonitorConfig()` silently DISARMS the
actor-lag hard abort that `configs/run5.yaml` now ships armed (ADJ-07), and five sibling
arms of the same idiom silently substitute a smoke cadence, a `None` eval section, the
literal encoding `"unknown"` (which `train/anchor.py` stamps, unvalidated, into a promoted
anchor and its `.provenance.json` sidecar — a permanent LAW-12 defect) and an empty
`full_config`. S-1/S-2 replace all six with ONE gate; this file is the instrument that
says so.

The oracles, and the defect each one is the ONLY witness to (DESIGN_S §6.8 measured six of
the ten regressions as invisible to the suite as it stands):

- O-S2   — every unvalidated shape is the SAME named error, and the gate fires before any
           subsystem is built. Ten-shape corpus (MF-5), plus the two `monitor_cfg`
           assertions (MF-1) of which the SIGNATURE CENSUS is the one that makes a silent
           re-add impossible.
- O-S1c  — the duck-typed config-getattr family is unrepresentable: a substring census
           (the CLASS defense — M-D is behaviourally invisible after the gate) and an AST
           pin that the gate is `compose_run`'s FIRST statement.
- O-S1a  — the composed encoding is the declared, REGISTERED one (LAW-11/LAW-12).
- O-S1b  — `full_config` carries the real config, not `{}` (the C-6 answer).
- O-S4-DISC — the reachability bound reads `train.max_train_steps`, not the LR-scheduler
           horizon `train.total_steps`. Every test payload in the repo sets the two equal,
           which is why the F-C revert is invisible without this oracle.
- O-AXIS — the config axis is VARIED, not re-pinned: all five minted configs through every
           composition seam, a bounded real drive on two representations, the R59 arming
           pin, and the absolute-ceiling resume arm.
- O-S5   — a drive failure is FATAL and `close_out` still ran.

DEVIATION FROM DESIGN PATH (logged in ORACLE_NOTES_S.md): DESIGN_S §6.0 places O-S2 as an
EDIT to `tests/test_run_composition.py` (inverting
`test_compose_run_falls_back_to_bare_monitor_config_when_config_has_no_monitor_section`).
ORACLE-WRITE's writable surface is NEW files only — same precedent as
`tests/test_actor_sync_composition.py`'s logged deviation — so O-S2 lives here. §6.0's
stated reason for the other placement was "two copies of one census is two authorities":
that reason does not bite, because the `dict` corpus row below is built from a MINTED
config's own `model_dump()` and this file therefore adds NO payload census at all. IMPL
must still DELETE the inverted test at its old site (it asserts the fallback that S-2
removes) — deleting it is already inside §7.3's `test_run_composition.py` fallout.

>300 justify (R8): DESIGN_S §6.0 specifies ONE new file for the §6 oracle set, and the
alternative — splitting the behavioural half out — would fork a fourth copy of the
drivable pool/trainer/buffer fakes (R5 bars cross-test imports), which is a worse LAW-03
outcome than one long file.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import ValidationError

import mantis.run
from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.composition import (  # RED-at-import anchor: module absent at HEAD
    UnvalidatedConfigError,
    require_run_config,
)
from mantis.config.resolve.run_length import (  # RED-at-import anchor: module absent at HEAD
    resolve_max_train_steps,
)
from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.monitor.config import MonitorConfig
from mantis.train.coordinator.config import StepCoordinatorConfig

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _REPO / "configs"

#: The axis, DERIVED from the minted directory rather than re-typed. A sixth minted config
#: joins every parametrized oracle below automatically; it can never be silently left off
#: the axis, which is the F-A defect one directory listing away from returning.
#: ADJ-13 F-1 corrective pass (recheck R-5): derived from the ONE discovery authority,
#: and as a path RELATIVE to configs/ so a subdirectory config is unambiguous on the
#: axis. A flat `*.yaml` glob is a sixth answer to "what is a config" and is blind to
#: exactly the `configs/prod/run6.yaml` shape both gates now make legal.
_MINTED: tuple[str, ...] = tuple(
    path.relative_to(_CONFIGS_DIR).as_posix() for path in discover_configs(_CONFIGS_DIR)
)

#: `cadence < threshold < max_train_steps` (the F-2 reachability bound) makes 3 the
#: smallest legal run at cadence 1. The drives below use 4 and 5 — IMPL may not "simplify"
#: the thresholds to values that collapse the chain; the config would stop loading.
_DRIVE_STEPS = 4
_DRIVE_THRESHOLD = 3


# ── the frozen schema oracle's payload builder, by PATH (R5: `tests` is not a package and
# no `sys.path` mutation is permitted; reading a frozen file is not editing it, R43) ───────
def _frozen_payload():
    path = _REPO / "tests" / "config" / "test_actor_sync_schema.py"
    spec = importlib.util.spec_from_file_location("_frozen_schema_for_wpax_s", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._payload


_payload = _frozen_payload()


# ── drivable fakes ───────────────────────────────────────────────────────────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    """The routing-harness pool surface + the ActorSyncTarget recorders. `games_completed`
    yields one fresh game per read so every `step()` runs exactly one burst."""

    def __init__(self) -> None:
        self._games = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self.started = False
        self.stopped = False
        self.sync_payloads: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


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


class _SentinelTrainError(RuntimeError):
    """Module-private on purpose: O-S5 must not be able to pass on an unrelated exception."""


class _ExplodingTrainer(_Trainer):
    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        raise _SentinelTrainError("the drive failed")


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None:
        return None

    def save_to_path(self, p) -> None:
        return None


def _fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


def _no_terminal_eval_config(*, stop_step, draw_rate_abort) -> StepCoordinatorConfig:
    """The ONE monkeypatch this file still applies, and only on `eval_enabled=True` drives:
    the production builder defaults `terminal_eval_enabled=True`, `close_out` therefore runs
    a terminal eval round, and `eval/snapshot.py` raises on any fake model that carries no
    declared `.arch`. That knob has NO config key — it is one of the 24 hardcoded
    `_default_step_coordinator_config` knobs owned by R-TRAINCONFIG-SCHEMA / ADJ-08
    (DESIGN_S §6.7). `stop_step` is deliberately left at 0 here: S-4 makes the config
    override it, so a patched builder that still dictated run length would hide the knob."""
    return StepCoordinatorConfig(
        terminal_eval_enabled=False,
        eval_interval=1000, log_interval=1000, checkpoint_interval=0, composition_interval=0,
        value_probe_interval=0, min_buf_size=1, capacity=100_000, buffer_schedule=(),
        training_steps_per_game=1.0, max_train_burst=1, batch_size=8, augment=False,
        recency_weight=0.0, mixing_initial_w=0.0, mixing_min_w=0.0, mixing_decay_steps=1.0,
        soft_ew_threshold=0.0, soft_ew_min_pts=0, hard_gn_threshold=1e9, hard_gn_min_steps=3,
        # WPAX Phase D: the two CONFIG-AUTHORED values arrive as required keyword
        # parameters and are passed THROUGH — a harness builder that swallowed them
        # would be a stand-in dictating a config fact, which is what this delta ends.
        instrumentation_enabled=False, stop_step=stop_step,
        draw_rate_abort=draw_rate_abort,
        final_eval_drain_timeout_sec=900.0,
    )


def _bounded(name: str = "smoke_gnn.yaml", factory=None, steps: int = _DRIVE_STEPS):
    """A real minted config, bounded so a drive terminates. The three step-clock knobs are
    co-overridden together because the reachability validator spans them: overriding
    `max_train_steps` alone leaves the config's own minted threshold of 100 above the new
    ceiling and the config stops loading (DESIGN_S §6.6 MF-3)."""
    return factory(name,
                   train={"actor_sync_cadence_steps": 1, "max_train_steps": steps},
                   monitor={"actor_lag_threshold_steps": steps - 1})


# ── O-S2 — the named error at every shape (S-1 + S-2; the RED-TEAM lens) ─────────────────
def _evasion_corpus(cfg: RunConfig) -> dict[str, Any]:
    """The MF-5 corpus. Every entry is a shape a caller could plausibly hand this root, and
    the gate's verdict on each is the whole content of "disarming can never again be the
    consequence of a shape"."""
    dumped = cfg.model_dump()

    class _Duck:
        train = cfg.train
        monitor = cfg.monitor
        eval = cfg.eval
        identity = cfg.identity

    return {
        # the shape 9 of the 12 call sites carry today
        "namespace": SimpleNamespace(),
        # the ADJ-07 shape, with the most plausible payload available: a MINTED config's own
        # model_dump(). Uses no hand-written census, so this file adds no fifteenth copy.
        "dict": dumped,
        # satisfies the RETIRED `.train` arm and has no monitor section at all
        "no_monitor_section": SimpleNamespace(
            train=SimpleNamespace(actor_sync_cadence_steps=1, max_train_steps=3)),
        "none": None,
        # the most convincing forgery available: every section is a REAL validated
        # sub-config. It must still be rejected, because the cross-field validators (the F-2
        # reachability bound among them) run on RunConfig, not on its parts.
        "namespace_with_real_sections": SimpleNamespace(
            train=cfg.train, monitor=cfg.monitor, eval=cfg.eval, identity=cfg.identity,
            model_dump=lambda: dumped),
        "duck_typed": _Duck(),
        # `unittest.mock` SETS `__class__`, so `isinstance(Mock(spec=RunConfig), RunConfig)`
        # is True for an object that has been through no validation whatsoever. An
        # isinstance gate is therefore not a validation statement, and these two rows are
        # what force the mechanism to read the object's REAL type (DESIGN_S §1.5).
        "mock_spec": Mock(spec=RunConfig),
        "magicmock_spec": MagicMock(spec=RunConfig),
    }


_REJECTED_SHAPES = ("namespace", "dict", "no_monitor_section", "none",
                    "namespace_with_real_sections", "duck_typed", "mock_spec",
                    "magicmock_spec")


@pytest.mark.parametrize("shape", _REJECTED_SHAPES)
def test_an_unvalidated_config_is_ONE_named_error_before_any_subsystem_exists(
    tmp_path, monkeypatch, smoke_run_config, shape: str
) -> None:
    """INVERSION. `tests/test_run_composition.py` used to assert the OPPOSITE of this — that
    a config with no `.monitor` section gets a bare `MonitorConfig()`. That test passed only
    because the defect existed: the bare default carries `actor_lag_abort_enabled=False`,
    so the fallback silently reverts the arming `configs/run5.yaml` ships (ADJ-07). The
    inversion IS the LAW-07 mutation pair.

    Eight shapes, ONE error type, one `match=` — so no shape is another's twin, which is
    what the RED-TEAM lens asks and what six independently-mutable error sites could not
    give. `build_run_safety` is replaced with a spy that RAISES if called, pinning that the
    gate fires before any subsystem is constructed.
    """
    def _must_not_be_called(**_kwargs):
        raise AssertionError(
            "build_run_safety was constructed for an unvalidated config — the gate must be "
            "compose_run's FIRST statement, before any subsystem exists"
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _must_not_be_called)
    subject = _evasion_corpus(smoke_run_config())[shape]

    with pytest.raises(UnvalidatedConfigError, match="schema-validated"):
        mantis.run.compose_run(
            config=subject, trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
        )


def test_the_gate_admits_a_validated_subclass_and_documents_the_model_construct_hole(
    smoke_run_config,
) -> None:
    """The two corpus rows that must NOT raise, asserted at the gate itself rather than
    through a drive (the second one cannot be driven — see below).

    `real_subclass`: it reached this root through `model_validate`, so every cross-field
    validator ran on it. Rejecting it would reject a MORE validated object than the one we
    accept (LSP). This row is what stops the mechanism from being `type(config) is
    RunConfig`.

    `model_construct`: a genuine, genuinely-typed `RunConfig` that skipped every validator.
    It PASSES **this gate**, and that is recorded rather than papered over: no type-based gate
    can see it, because it is not a spoof — it is the class. This row is asserted at
    `require_run_config` rather than through a drive because the gate is what it is about.

    CORRECTED BY WPAX R67 (RED-TEAM-2 N-3/N-4). Three sentences here were false and are struck:

    1. struck — "the first typed read (`config.monitor`) raises `AttributeError` LOUDLY". It is
       unreachable: `revalidate_run_config`, `compose_run`'s SECOND statement, rejects the
       object before any typed read happens.
    2. struck — "the only instrument that would see it is a source census banning
       `model_construct` in `tests/` … filed as WP-R §9.10". False: `revalidate_run_config`
       sees it, in-repo, and landed inside Phase S. **WP-R §9.10 is therefore CLOSED IN FACT
       and wants re-adjudication rather than continued carriage** (N-4).
    3. struck — "driving it through `compose_run` would blow up on that AttributeError".
       Measured false: `compose_run(RunConfig.model_construct(run_id="x"))` raises
       `UnvalidatedConfigError`.

    None of the three carried an assertion — this test's only two assertions are identity
    checks on the unchanged `require_run_config`, which is why the staleness was safe to carry
    until an R43 event could correct it. The live rule is now: this gate admits the object,
    and the re-validation one line later is what rejects it.
    """
    cfg = smoke_run_config()

    class _Sub(RunConfig):
        pass

    subclass = _Sub.model_validate(cfg.model_dump())
    assert require_run_config(subclass, caller="compose_run") is subclass, (
        "a validated RunConfig SUBCLASS must pass the gate — it is more validated than the "
        "object the gate accepts, not less"
    )

    constructed = RunConfig.model_construct(run_id="x")
    assert require_run_config(constructed, caller="compose_run") is constructed, (
        "model_construct builds a real RunConfig that skipped validation; the gate passes "
        "it BY DESIGN (WP-R §9.10) and the first typed read fails loud. If this ever starts "
        "raising, the gate grew a validation check and §9.10 should be closed, not patched"
    )


def test_compose_run_rejects_a_monitor_cfg_KEYWORD_at_call_time(tmp_path, smoke_run_config):
    """MF-1, assertion 1. `monitor_cfg=` was a production parameter that existed only so a
    harness could inject a `MonitorConfig`, and it short-circuited the config's own monitor
    section entirely: a fully valid, fully ARMED `RunConfig` plus `monitor_cfg=
    MonitorConfig()` reached the watchdog with `armed=False`. It is DELETED, not routed —
    routing closes the F-3 false positive and leaves the ADJ-07 disarm open."""
    with pytest.raises(TypeError, match="monitor_cfg"):
        mantis.run.compose_run(
            config=smoke_run_config(), trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
            eval_enabled=False, monitor_cfg=MonitorConfig(),
        )


def test_compose_runs_parameter_list_is_pinned_so_no_re_add_can_be_silent():
    """MF-1, assertion 2 — the SIGNATURE CENSUS, and it is the load-bearing half.

    The call-time assertion above would NOT catch a `monitor_cfg=None` parameter that is
    re-added and never exercised: zero of the suite's `compose_run` call sites pass it, so a
    re-add costs zero failures. The "no member of the duck-typed family is left" claim is
    only earned while this tuple holds.

    What it cannot see: a parameter that carries a config fact under a DIFFERENT name would
    change this tuple and fail here loudly — that is the point — but a config fact smuggled
    through an EXISTING parameter (say a `trainer` that also carries `.monitor`) is invisible
    to a signature census. That route is F9's collaborator seam, deliberately left injectable.
    """
    assert tuple(inspect.signature(mantis.run.compose_run).parameters) == (
        "config", "trainer", "pool", "buffer", "log_dir", "checkpoint_dir",
        "eval_enabled", "run_id",
    ), (
        "compose_run's parameter list is pinned: no parameter may carry a CONFIG FACT into "
        "this root (WPAX MF-1 — monitor_cfg bypassed the gate and silently disarmed the "
        "abort Phase F armed). Adding one is a design decision, not an edit."
    )


# ── O-S1c — the CLASS defense: the duck-typed config-getattr family is unrepresentable ────
def test_the_composition_root_contains_no_duck_typed_config_getattr():
    """The class defense, and the ONLY instrument that sees mutation M-D at all.

    After the gate, re-introducing the retired idiom is BEHAVIOURALLY INVISIBLE — a
    validated `RunConfig` has `.identity`, so `getattr(...)` returns the right answer and
    every behavioural oracle stays green. Measured: M-D survives the whole suite. So the
    defense here is necessarily structural.

    What this census CANNOT see, booked as KNOWN rather than scored as a kill (DESIGN_S
    §6.3): an aliased subject (`cfg = config` then `getattr(cfg, ...)`), a single space
    (`getattr( config, ...)`), `vars(config).get(...)`, `operator.attrgetter(...)(config)`,
    and an aliased builtin (`_g = getattr; _g(config, ...)`). Five of six walk through. The
    claim is narrowed to what is true: the census catches ACCIDENTAL REINTRODUCTION of the
    retired idiom — which, once the gate makes the idiom inert, is the only hazard left.
    Nobody restores a silent config default by writing `operator.attrgetter`; they restore it
    by pasting the old line back.

    The census scans PROSE too, deliberately: a docstring that quotes the retired idiom
    verbatim trips it. Describe the family, do not spell it.
    """
    banned = "getattr(" + "config"  # assembled at runtime so this census cannot trip itself
    text = Path(mantis.run.__file__).read_text(encoding="utf-8")
    assert banned not in text, (
        f"src/mantis/run.py contains `{banned}`: the composition root duck-types its own "
        "config again. After the strict gate every section is a typed attribute read; the "
        "idiom's return is a SILENT default, and one of them disarms a hard abort (ADJ-07)"
    )


def _compose_run_body() -> list[ast.stmt]:
    tree = ast.parse(inspect.getsource(mantis.run))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compose_run":
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]  # the docstring
            return body
    raise AssertionError("no `def compose_run` found in mantis.run")


def test_the_config_gate_is_compose_runs_FIRST_statement():
    """Placement is the whole safety property. A gate that runs after `build_run_safety`
    has already been handed a bare `MonitorConfig()` validates nothing that matters — the
    watchdog is built disarmed and the gate then blesses the config that would have armed
    it. House precedent for the walk: `test_actor_sync_production_posture.py`'s
    `_actor_sync_assignment()`."""
    body = _compose_run_body()
    assert body, "compose_run has no body beyond its docstring"
    first = body[0]
    assert isinstance(first, ast.Assign), (
        f"compose_run's first statement is a {type(first).__name__}, not the config gate "
        "assignment `config = require_run_config(config, caller=...)`"
    )
    assert isinstance(first.value, ast.Call), (
        f"compose_run's first statement assigns a {type(first.value).__name__}, not a call"
    )
    assert getattr(first.value.func, "id", None) == "require_run_config", (
        "compose_run's first statement must be the `require_run_config` gate; anything "
        "built before it is built from an unvalidated config"
    )


# ── O-S1a — the composed encoding is the declared, REGISTERED one (LAW-11 / LAW-12) ───────
def test_the_composed_encoding_is_the_declared_and_REGISTERED_one(
    tmp_path, monkeypatch, smoke_run_config
):
    """The retired arm substituted the literal `"unknown"` for the encoding at TWO sites,
    and the second one is permanent: `DeployTagHooks.encoding` is the value
    `train/anchor.py`'s `save_best_model_atomic` stamps into the promoted anchor's payload
    AND its `.provenance.json` sidecar, with no validation of the value whatsoever. So the
    fallback did not merely mis-route an eval round — it wrote an unregistered encoding
    literal permanently into a promoted artifact. That is LAW-12 crossed with R3.

    The `lookup()` assertion is MANDATORY, not decorative: §1.3's argument for NOT raising
    `AbsentEncodingError` here is that an unregistered encoding is UNREACHABLE once the
    section is typed, and an unreachability argument needs an oracle or it is an assertion.
    CI gate 11 carries no `"unknown"` pattern, so the gate cannot see the retired arm and
    this is the only instrument that can.
    """
    import mantis.train.anchor as _anchor

    cfg = _bounded(factory=smoke_run_config)
    captured: dict[str, Any] = {}

    def _spy_build_eval_pipeline(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            run_evaluation=lambda *a, **k: {"kicked": False, "reason": None},
            poll_completed=lambda: None, drain_pending=lambda: None,
            apply_gate_decision=lambda *a, **k: None, stop=lambda: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)
    monkeypatch.setattr(mantis.run, "build_eval_pipeline", _spy_build_eval_pipeline)
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _no_terminal_eval_config)
    # An eval pipeline makes `run_training_loop` seed the anchor, which initialises a real
    # model from `trainer.model` and reads `.arch` off it. Same patch, same reason, as
    # `tests/train/test_actor_sync_real_config.py`'s `_drive`. The assertions below observe
    # what `compose_run` COMPOSED, so the anchor seed is harness, not subject.
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )

    mantis.run.compose_run(
        config=cfg, trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
    )

    assert captured["encoding"] == cfg.identity.encoding, (
        f"the eval pipeline was composed with encoding {captured.get('encoding')!r}, not "
        f"the config's declared {cfg.identity.encoding!r}"
    )
    assert lookup(captured["encoding"]) is not None, (
        "the composed encoding is not in the registry — an unregistered literal here is "
        "written into the promoted anchor's stamp and cannot be un-written (LAW-12)"
    )
    assert captured["promotion"].encoding == cfg.identity.encoding, (
        "DeployTagHooks carries a different encoding from the eval pipeline; this is the "
        "value stamped into the promoted anchor payload and its .provenance.json sidecar"
    )


# ── O-S1b — `full_config` carries the real config (the C-6 answer) ────────────────────────
def test_full_config_carries_the_real_config_not_an_empty_dict(
    tmp_path, monkeypatch, smoke_run_config
):
    """The retired arm was `full_config=(config if isinstance(config, dict) else {})`, i.e.
    `{}` for every real `RunConfig`. Both of `{}`'s destinations are inert today (a
    hardcoded batch-size default and a pair of dead `emit_training_events` parameters), so
    nothing crashed and nothing was measurable — which is exactly why this needs an oracle
    rather than a bug report. This drive uses the PRODUCTION
    `_default_step_coordinator_config()` with NO monkeypatch: after S-4 the config authors
    `stop_step`, so a real bounded burst is finally reachable without a harness patch."""
    cfg = _bounded(factory=smoke_run_config)
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
    )

    full_config = handles.coordinator.full_config
    assert full_config, "full_config is empty for a real RunConfig — the `{}` arm is back"
    assert full_config["identity"]["encoding"] == cfg.identity.encoding
    assert full_config["train"]["max_train_steps"] == _DRIVE_STEPS, (
        "full_config does not carry the COMPOSED run length; a stale or synthetic dump "
        "would satisfy a truthiness check but not this"
    )


# ── O-S4-DISC — the bound discriminates max_train_steps from total_steps ──────────────────
def test_a_cadence_inside_the_LR_horizon_but_beyond_the_RUN_is_rejected():
    """The mandatory F-C oracle, arm (a) — the live defect.

    `total_steps` is ONLY the LR-scheduler horizon; no stop condition reads it. The real
    run-length authority is `StepCoordinatorConfig.stop_step`, now config-authored as
    `train.max_train_steps`. A bound anchored to the horizon blesses a cadence the run never
    reaches: 100 against a 50-step run, inside a `total_steps` of 1 000 000. That is run3's
    frozen actor in a config that validates clean.

    This oracle exists because EVERY test payload in the repo sets `total_steps ==
    max_train_steps`, which makes the two anchors experimentally indistinguishable and the
    F-C revert invisible to the entire suite (measured: the revert is 1701-green without
    this file). Driving the two knobs APART is the only thing that discriminates them.
    """
    with pytest.raises(ValidationError, match="must be < train.max_train_steps"):
        RunConfig(**_payload(
            train_over={"total_steps": 1_000_000, "max_train_steps": 50,
                        "actor_sync_cadence_steps": 100},
            monitor_over={"actor_lag_threshold_steps": 200},
        ))


def test_a_run_that_outlives_its_LR_horizon_is_accepted():
    """Arm (b) — the other direction, and it is not optional: without it the fix could be a
    blanket tightening that rejects any cadence beyond `total_steps` as well, and the revert
    would survive in one direction. A 1 000 000-step run whose cosine schedule anneals over
    100 steps is a strange config, but it is not an UNREACHABLE-CADENCE config, and only the
    run-length knob may decide that."""
    cfg = RunConfig(**_payload(
        train_over={"total_steps": 100, "max_train_steps": 1_000_000,
                    "actor_sync_cadence_steps": 2000},
        monitor_over={"actor_lag_threshold_steps": 3000},
    ))
    assert cfg.train.max_train_steps == 1_000_000
    assert cfg.train.actor_sync_cadence_steps == 2000


# ── O-AXIS — the config axis is VARIED, not re-pinned (F-A's lesson) ──────────────────────
def test_the_axis_is_the_whole_minted_set_and_is_not_empty():
    """Vacancy guard for every parametrized oracle below. `_MINTED` is globbed, so it cannot
    silently omit a newly minted config — but a glob that returns nothing would silently
    delete the axis instead, and a parametrized test with zero params is a green no-op."""
    assert len(_MINTED) >= 5, f"the minted-config axis collapsed to {_MINTED}"
    assert "run5.yaml" in _MINTED, f"the production config is not on the axis: {_MINTED}"


@pytest.mark.parametrize("name", _MINTED)
def test_every_minted_config_resolves_through_every_composition_seam(name: str):
    """Point 1 — the five-way resolver census. This is the cheap way to put five REAL
    production values on the axis that carried one test-only value (`SimpleNamespace()`) for
    the whole of this lineage. Two templates, two representations, five run_ids, every
    composition seam, through the ONE loader. NO drive, so run5's real 1 000 000-step run
    length is never executed."""
    cfg = load_config(_CONFIGS_DIR / name)

    assert mantis.run._resolve_actor_sync_cadence_steps(cfg) == cfg.train.actor_sync_cadence_steps
    resolved = mantis.run._resolve_monitor_cfg(cfg)
    assert resolved.actor_lag_threshold_steps == cfg.monitor.actor_lag_threshold_steps
    assert resolved.actor_lag_abort_enabled is cfg.monitor.actor_lag_abort_enabled
    assert resolve_max_train_steps(cfg.train) == cfg.train.max_train_steps
    assert lookup(cfg.identity.encoding) is not None, (
        f"configs/{name} declares encoding {cfg.identity.encoding!r}, which is not registered"
    )


def test_the_minted_PRODUCTION_config_ships_the_actor_lag_abort_ARMED():
    """Point 3 (MF-4) — the R59 per-commit pin. NOT red-before-IMPL: it is GREEN the moment
    this file collects, because Phase F armed it at `0ef05ff`. It is a regression pin, and
    the regression it pins is specific and measured: Phase S RE-MINTS run5, and an honest
    re-mint that drops `--set monitor.actor_lag_abort_enabled=true` writes `false` back with
    CI gate 7 at rc=0 and the header-truthfulness suite 9-passed — because a re-mint rewrites
    the header too, so header-truthfulness cannot see it. Every existing defense passes; this
    line is the only thing that fails.

    Its LAW-07 mutation self-test was executed at ORACLE-WRITE against a run5 re-minted
    WITHOUT the `--set` (into a scratch path, never into `configs/`): the predicate reads
    False. See ORACLE_NOTES_S.md.

    Placement: it rides on the same load as the five-way census above, deliberately — two
    files loading run5 to assert two things about it is the LAW-03 shape this run keeps
    filing. It is an INSTANCE ("this config's value"); Phase P's armed-abort manifest is the
    RULE ("which aborts a production config must arm"), and DESIGN_P must reconcile them.
    """
    assert load_config(_CONFIGS_DIR / "run5.yaml").monitor.actor_lag_abort_enabled is True, (
        "R59: the minted PRODUCTION config's actor-lag hard abort ships ARMED. Phase S "
        "re-mints run5; a dropped `--set monitor.actor_lag_abort_enabled=true` reverts "
        "Phase F (0ef05ff) with gate 7 and header-truthfulness both green."
    )


@pytest.mark.parametrize("name", ("smoke_gnn.yaml", "smoke_radius_curriculum.yaml"))
def test_a_bounded_real_config_drive_syncs_every_step_on_both_representations(
    tmp_path, monkeypatch, smoke_run_config, name: str
):
    """Point 2 — the bounded drive, on BOTH representations (`gnn_axis_v1` / graph from the
    `dev` template, `v6w25` / grid from the `grid` template). This is the behavioural half of
    the axis: point 1 proves five configs resolve, this proves two of them DRIVE.

    No `_default_step_coordinator_config` monkeypatch: the production builder runs, because
    S-4 makes the config author `stop_step`. That retires the C-6 harness patch for every
    `eval_enabled=False` drive — the axis moves from "unclosable by any test" to "closed for
    one posture, blocked on one named knob (`terminal_eval_enabled`) for the other".

    The step counts below therefore depend on the production builder's other 24 hardcoded
    knobs (`max_train_burst=1`, `min_buf_size=1`, `eval_interval=1000`). That coupling is
    deliberate — the oracle exercises the production seam — but a change to `max_train_burst`
    moves them, and the knobs have no config authority (R-TRAINCONFIG-SCHEMA / ADJ-08).
    """
    cfg = _bounded(name, factory=smoke_run_config)
    pool, trainer = _Pool(), _Trainer()
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
    )

    assert handles.eval_pipeline is None, "harness precondition: the deploy side must not exist"
    assert trainer.step == _DRIVE_STEPS, (
        f"the run length came from somewhere other than train.max_train_steps: stopped at "
        f"{trainer.step}, config says {_DRIVE_STEPS}"
    )
    assert pool.step_calls == list(range(1, _DRIVE_STEPS + 1)), (
        f"cadence 1 must sync on every step: {pool.step_calls}"
    )
    assert len(pool.sync_payloads) == _DRIVE_STEPS
    assert all(sd is trainer.inference_sd for sd in pool.sync_payloads)


def test_the_run_length_ceiling_is_ABSOLUTE_not_per_process(
    tmp_path, monkeypatch, smoke_run_config
):
    """Point 4 — the resume arm (V-NOOP's fifth row).

    WHY ZERO SYNCS IS CORRECT HERE, and this docstring is load-bearing: `StepCoordinator`
    seeds `self._train_step` from `trainer.step`, so `max_train_steps` is an ABSOLUTE
    ceiling, not a per-process budget. A run resumed at step 7 against a cap of 5 is DONE:
    O2 fires before the burst loop, the composition returns, and the actor never syncs.
    Superficially that is indistinguishable from the frozen actor this whole run exists to
    kill, which is exactly why it is pinned by name rather than left to be rediscovered as a
    bug. If a future change makes the ceiling relative, THIS test is what fails.
    """
    cfg = _bounded(factory=smoke_run_config, steps=5)
    pool, trainer = _Pool(), _Trainer(step=7)
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
    )

    assert trainer.step == 7, (
        f"a run resumed past its cap must not train: step went 7 -> {trainer.step}, which "
        "means the ceiling is being read as a per-process budget"
    )
    assert pool.sync_payloads == [], "no step ran, so no sync may have happened"
    assert handles.shutdown.running is False, (
        "composition must have RETURNED with the loop stopped, not spun"
    )


# ── O-S5 — a drive failure is FATAL, and close_out still ran ──────────────────────────────
def test_a_drive_failure_propagates_and_close_out_still_ran(
    tmp_path, monkeypatch, smoke_run_config
):
    """The two blanket `except Exception -> log -> return` blocks existed so a fakes harness
    could not crash this root — the same defect as the smoke resolver arm (accommodating
    test doubles in production code), and they also swallowed actor-SYNC failures into an
    exit-0 return. Fail-loud law wins.

    Both halves are asserted because they fail on different mutations: re-adding the
    `except` makes the raise disappear, and dropping the `finally` makes `close_out` — and
    therefore the buffer save and the guarded pool stop — vanish on the failure path. The
    error type is module-private so this oracle cannot pass on an unrelated exception.
    """
    cfg = _bounded(factory=smoke_run_config)
    pool = _Pool()
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    with pytest.raises(_SentinelTrainError):
        mantis.run.compose_run(
            config=cfg, trainer=_ExplodingTrainer(), pool=pool, buffer=_Buffer(),
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=False,
        )

    assert pool.started is True, "harness precondition: the pool was started by this run"
    assert pool.stopped is True, (
        "close_out did not run on the failure path — the epilogue must be in a `finally`, "
        "or a failed run loses its buffer save and leaves the pool running"
    )
