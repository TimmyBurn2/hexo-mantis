"""Oracles for the STRICT composition root.

`compose_run` used to duck-type its own config: an absent `monitor` section silently DISARMED
the actor-lag hard abort a shipped config arms, and five sibling arms substituted a smoke
cadence, a `None` eval section, the literal encoding `"unknown"` and an empty `full_config`.
One gate replaces all six.

>300 justify (R8): splitting the behavioural half out would fork a fourth copy of the drivable
pool/trainer/buffer fakes, cross-test imports being barred.
"""
from __future__ import annotations

import ast
import dataclasses
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
from mantis.config.resolve.disk_guard import resolve_disk_guard
from mantis.config.resolve.run_length import (  # RED-at-import anchor: module absent at HEAD
    resolve_max_train_steps,
)
from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.monitor.config import MonitorConfig
from mantis.train.coordinator.config import StepCoordinatorConfig

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _REPO / "configs"

#: The axis, DERIVED from the ONE discovery authority rather than re-typed, as a path
#: RELATIVE to configs/ so a subdirectory config is unambiguous. A sixth minted config joins
#: every parametrized oracle below automatically; a flat `*.yaml` glob would be a sixth answer
#: to "what is a config" and blind to the `configs/prod/run6.yaml` shape.
_MINTED: tuple[str, ...] = tuple(
    path.relative_to(_CONFIGS_DIR).as_posix() for path in discover_configs(_CONFIGS_DIR)
)

#: `cadence < threshold < max_train_steps` (the reachability bound) makes 3 the smallest legal
#: run at cadence 1. The drives below use 4 and 5; lowering the thresholds stops the config
#: loading.
_DRIVE_STEPS = 4
_DRIVE_THRESHOLD = 3


# The frozen schema oracle's payload builder, read BY PATH: `tests` is not a package and no
# `sys.path` mutation is permitted.
def _frozen_payload():
    path = _REPO / "tests" / "config" / "test_actor_sync_schema.py"
    spec = importlib.util.spec_from_file_location("_frozen_schema_for_wpax_s", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._payload


_payload = _frozen_payload()


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
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # the third outcome share.
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
        self.device = "cpu"
        self.inference_sd = {"w": "SENTINEL"}

    # The double conforms to the DECLARED seam (typed entry points + `device`).
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


class _SentinelTrainError(RuntimeError):
    """Module-private on purpose: O-S5 must not be able to pass on an unrelated exception."""


class _ExplodingTrainer(_Trainer):
    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        raise _SentinelTrainError("the drive failed")

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        raise _SentinelTrainError("the drive failed")


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None:
        return None

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        # The grid route's sampler; rows are opaque to _Trainer.
        return (None,) * 9


def _fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


#: The UNPATCHED production builder, captured at import so the patch below can delegate to it
#: without re-entering itself.
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


def _no_terminal_eval_config(**kwargs) -> StepCoordinatorConfig:
    """Patch `terminal_eval_enabled` off — a ONE-KNOB DELTA over the real builder, so
    `**kwargs` still forwards every CONFIG-AUTHORED value including `stop_step`. The knob has
    no config key, and left on, `close_out` runs a terminal eval that raises on a fake model
    carrying no declared `.arch`."""
    return dataclasses.replace(_PRODUCTION_BUILDER(**kwargs), terminal_eval_enabled=False)


def _bounded(name: str = "smoke_preflight_armed.yaml", factory=None, steps: int = _DRIVE_STEPS,
             eval_enabled: bool = False):
    """Load a real minted config, bounded so a drive terminates.

    The three step-clock knobs are co-overridden together because the reachability validator
    spans them, and `train.draw_rate_abort` is disarmed by that same validator: an ARMED
    `min_step` above the new ceiling is a floor the run never reaches.
    """
    return factory(name,
                   train={"actor_sync_cadence_steps": 1, "max_train_steps": steps,
                          # graph drives run the real route; 256 batch is drag.
                          "batch_size": 8, "draw_rate_abort": None},
                   monitor={"actor_lag_threshold_steps": steps - 1},
                   eval_enabled=eval_enabled)


def _evasion_corpus(cfg: RunConfig) -> dict[str, Any]:
    """Every shape a caller could plausibly hand this root; the gate's verdict on each is the
    whole content of "disarming can never again be the consequence of a shape"."""
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
        # model_dump(), so this file adds no hand-written payload census.
        "dict": dumped,
        # satisfies the RETIRED `.train` arm and has no monitor section at all
        "no_monitor_section": SimpleNamespace(
            train=SimpleNamespace(actor_sync_cadence_steps=1, max_train_steps=3)),
        "none": None,
        # the most convincing forgery available: every section is a REAL validated sub-config.
        # Still rejected, because the cross-field validators run on RunConfig, not on its parts.
        "namespace_with_real_sections": SimpleNamespace(
            train=cfg.train, monitor=cfg.monitor, eval=cfg.eval, identity=cfg.identity,
            model_dump=lambda: dumped),
        "duck_typed": _Duck(),
        # `unittest.mock` SETS `__class__`, so `isinstance(Mock(spec=RunConfig), RunConfig)` is
        # True for an object that has been through no validation. An isinstance gate is
        # therefore not a validation statement; these two rows force a read of the REAL type.
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
    """Eight unvalidated shapes all raise ONE named error, before any subsystem is built.

    The inverted predecessor asserted a config with no `.monitor` gets a bare `MonitorConfig()`
    — which silently reverted a shipped arming, since that default is `enabled=False`.
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
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        )


def test_the_gate_admits_a_validated_subclass_and_documents_the_model_construct_hole(
    smoke_run_config,
) -> None:
    """A validated subclass and a `model_construct`ed config both pass the gate.

    The subclass row stops the mechanism from being `type(config) is RunConfig`; the
    `model_construct` row is recorded rather than papered over, since no type-based gate can
    see it and `revalidate_run_config` one line later is what rejects it.
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
    """`monitor_cfg=` is DELETED from the signature, not routed: it short-circuited the
    config's own monitor section, so an ARMED `RunConfig` plus `monitor_cfg=MonitorConfig()`
    reached the watchdog with `armed=False`."""
    with pytest.raises(TypeError, match="monitor_cfg"):
        mantis.run.compose_run(
            config=smoke_run_config(), trainer=_Trainer(), pool=_Pool(), buffer=_Buffer(),
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
            # `eval_enabled=` is GONE from this call deliberately: CPython names the FIRST
            # unexpected keyword, so a second dead kwarg would break `match="monitor_cfg"`.
            monitor_cfg=MonitorConfig(),
        )


def test_compose_runs_parameter_list_is_pinned_so_no_re_add_can_be_silent():
    """No member of the duck-typed family is left in `compose_run`'s signature.

    The call-time assertion above would not catch a `monitor_cfg=None` re-added and never
    exercised. What a signature census cannot see is a config fact smuggled through an EXISTING
    parameter — that route is the collaborator seam, deliberately left injectable.
    """
    assert tuple(inspect.signature(mantis.run.compose_run).parameters) == (
        "config", "trainer", "pool", "buffer", "log_dir", "checkpoint_dir", "resume_state",
    ), (
        "compose_run's parameter list is pinned: no parameter may carry a CONFIG FACT into "
        "this root (WPAX MF-1 — monitor_cfg bypassed the gate and silently disarmed the "
        "abort Phase F armed). Adding one is a design decision, not an edit. WPMAIN shrank "
        "the tuple to six: `eval_enabled` (R120) and `run_id` (R123) were the last two "
        "parameters carrying config facts, and they are DELETED rather than merely stripped "
        "of their defaults — a required parameter is a forcing route with the default "
        "removed, not a closed one, so R64's 'the preflight may never force False' is only "
        "structurally unrepresentable with the parameter gone. R343(c) added the SEVENTH, "
        "`resume_state`, and it is admissible on this census's own terms: it carries no "
        "config fact. It is RESTORED RUNTIME STATE — a round counter, a p_hat and a ring "
        "identity read from the sidecar beside the checkpoint being resumed — and its only "
        "producer is `build_run_collaborators`, the same builder every other collaborator on "
        "this list arrives from. The alternative shapes were both worse: a second read of the "
        "sidecar inside this root would be a second authority for what the boot resumed from, "
        "and smuggling it through `trainer` is the invisible route this docstring already "
        "names as the census's blind spot. Adding it is a design decision and it is recorded "
        "as one, which is what this tuple exists to force."
    )


def test_the_composition_root_contains_no_duck_typed_config_getattr():
    """The retired attribute-probe idiom is absent from the composition root.

    Re-introducing it after the gate is BEHAVIOURALLY INVISIBLE — a validated `RunConfig` has
    `.identity`, so the probe returns the right answer — which is why the defense is
    structural. It catches ACCIDENTAL REINTRODUCTION, not an aliased subject, an extra space,
    `vars(config).get(...)`, `operator.attrgetter` or an aliased builtin.

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
    """The gate is `compose_run`'s FIRST statement, and placement is the whole safety property:
    a gate that runs after `build_run_safety` has already been handed a bare `MonitorConfig()`
    blesses the config that would have armed a watchdog built disarmed."""
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


def test_the_composed_encoding_is_the_declared_and_REGISTERED_one(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
):
    """The composed encoding is the declared, REGISTERED one.

    The retired arm substituted the literal `"unknown"`, and one of its two sites is permanent:
    `DeployTagHooks.encoding` is stamped unvalidated into the promoted anchor and its
    `.provenance.json` sidecar. The `lookup()` assertion is mandatory rather than decorative,
    because CI gate 11 carries no `"unknown"` pattern.
    """
    import mantis.train.anchor as _anchor

    cfg = _bounded(factory=smoke_run_config, eval_enabled=True)
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
    # An eval pipeline makes `run_training_loop` seed the anchor from `trainer.model`, reading
    # `.arch` off it. The assertions below observe what `compose_run` COMPOSED.
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )

    mantis.run.compose_run(
        config=cfg, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
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


def test_full_config_carries_the_real_config_not_an_empty_dict(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
):
    """`full_config` carries the real config, not `{}`.

    Both of `{}`'s destinations are inert today, so nothing crashed and nothing was
    measurable — which is why this needs an oracle rather than a bug report.
    """
    cfg = _bounded(factory=smoke_run_config)
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    full_config = handles.coordinator.full_config
    assert full_config, "full_config is empty for a real RunConfig — the `{}` arm is back"
    assert full_config["identity"]["encoding"] == cfg.identity.encoding
    assert full_config["train"]["max_train_steps"] == _DRIVE_STEPS, (
        "full_config does not carry the COMPOSED run length; a stale or synthetic dump "
        "would satisfy a truthiness check but not this"
    )


def test_a_cadence_inside_the_LR_horizon_but_beyond_the_RUN_is_rejected():
    """The reachability bound reads `train.max_train_steps`, not `train.total_steps`.

    `total_steps` is ONLY the LR-scheduler horizon; a bound anchored to it blesses a cadence of
    100 against a 50-step run. Every test payload in the repo sets the two equal, so driving
    them APART is the only thing that discriminates them.
    """
    with pytest.raises(ValidationError, match="must be < train.max_train_steps"):
        RunConfig(**_payload(
            train_over={"total_steps": 1_000_000, "max_train_steps": 50,
                        "actor_sync_cadence_steps": 100},
            monitor_over={"actor_lag_threshold_steps": 200},
        ))


def test_a_run_that_outlives_its_LR_horizon_is_accepted():
    """A cadence beyond `total_steps` but inside the run length is ACCEPTED — without this arm
    the fix could be a blanket tightening that survives in one direction."""
    cfg = RunConfig(**_payload(
        train_over={"total_steps": 100, "max_train_steps": 1_000_000,
                    "actor_sync_cadence_steps": 2000},
        monitor_over={"actor_lag_threshold_steps": 3000},
    ))
    assert cfg.train.max_train_steps == 1_000_000
    assert cfg.train.actor_sync_cadence_steps == 2000


def test_the_axis_is_the_whole_minted_set_and_is_not_empty():
    """The globbed minted-config axis is non-empty; a zero-param parametrize is a green
    no-op."""
    assert len(_MINTED) >= 3, f"the minted-config axis collapsed to {_MINTED}"
    assert "run6.yaml" in _MINTED, f"the production config is not on the axis: {_MINTED}"


@pytest.mark.parametrize("name", _MINTED)
def test_every_minted_config_resolves_through_every_composition_seam(name: str):
    """Every minted config resolves through EVERY composition seam, with no drive.

    The name says EVERY seam, so a seam added without a line here makes the name false and the
    fix is one line, not a rename.
    """
    cfg = load_config(_CONFIGS_DIR / name)

    assert mantis.run._resolve_actor_sync_cadence_steps(cfg) == cfg.train.actor_sync_cadence_steps
    resolved = mantis.run._resolve_monitor_cfg(cfg)
    assert resolved.actor_lag_threshold_steps == cfg.monitor.actor_lag_threshold_steps
    assert resolved.actor_lag_abort_enabled is cfg.monitor.actor_lag_abort_enabled
    assert resolve_max_train_steps(cfg.train) == cfg.train.max_train_steps
    guard = resolve_disk_guard(cfg.monitor)
    assert (guard.interval_sec, guard.warn_gb, guard.fail_gb) == (
        cfg.monitor.disk_guard.interval_sec, cfg.monitor.disk_guard.warn_gb,
        cfg.monitor.disk_guard.fail_gb,
    ), f"configs/{name} does not resolve through the disk-guard seam (LAW-16 leg 3, R122)"
    assert cfg.train.device in ("cpu", "cuda"), (
        f"configs/{name} declares train.device {cfg.train.device!r}: the device is a CONFIG "
        "FACT read by the composition root (R126), and every minted config must author it"
    )
    assert lookup(cfg.identity.encoding) is not None, (
        f"configs/{name} declares encoding {cfg.identity.encoding!r}, which is not registered"
    )


def test_the_minted_PRODUCTION_config_ships_the_actor_lag_abort_ARMED():
    """run5 ships `monitor.actor_lag_abort_enabled: true`, pinned per commit.

    A re-mint of run5 that drops the `--set` writes `false` back with CI gate 7 at rc 0 and the
    header-truthfulness suite green, because a re-mint rewrites the header too. Mutation
    self-test: against a run5 re-minted without the `--set`, the predicate reads False.
    """
    assert load_config(_CONFIGS_DIR / "run6.yaml").monitor.actor_lag_abort_enabled is True, (
        "R59: the minted PRODUCTION config's actor-lag hard abort ships ARMED. Phase S "
        "re-mints run5; a dropped `--set monitor.actor_lag_abort_enabled=true` reverts "
        "Phase F (0ef05ff) with gate 7 and header-truthfulness both green."
    )


@pytest.mark.parametrize("name", ("smoke_preflight_armed.yaml",))
def test_a_bounded_real_config_drive_syncs_every_step_on_the_declared_representation(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, name: str
):
    """A bounded drive on a REAL minted config — the behavioural half of the axis.

    `smoke_preflight_armed.yaml` is the only CPU config in the committed set; a `device: cuda`
    config does not terminate on a CPU box. The step counts below depend on the production
    builder's other hardcoded knobs (`max_train_burst=1`, `min_buf_size=1`,
    `eval_interval=1000`), which have no config authority.
    """
    cfg = _bounded(name, factory=smoke_run_config)
    pool, trainer = _Pool(), _Trainer()
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool,
        # The declared route samples a REAL HexgBuffer — the typed dispatcher refuses a
        # shapeless fake at the route, by design.
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
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
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
):
    """A run resumed past its ceiling performs ZERO syncs, and that is correct.

    `StepCoordinator` seeds `self._train_step` from `trainer.step`, so `max_train_steps` is an
    ABSOLUTE ceiling: a run resumed at step 7 against a cap of 5 is DONE before the burst loop.
    Superficially indistinguishable from the frozen actor, hence pinned by name.
    """
    cfg = _bounded(factory=smoke_run_config, steps=5)
    pool, trainer = _Pool(), _Trainer(step=7)
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    handles = mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert trainer.step == 7, (
        f"a run resumed past its cap must not train: step went 7 -> {trainer.step}, which "
        "means the ceiling is being read as a per-process budget"
    )
    assert pool.sync_payloads == [], "no step ran, so no sync may have happened"
    assert handles.shutdown.running is False, (
        "composition must have RETURNED with the loop stopped, not spun"
    )


def test_a_drive_failure_propagates_and_close_out_still_ran(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
):
    """A drive failure is FATAL and `close_out` still ran.

    Both halves are asserted because they fail on different mutations: re-adding the blanket
    `except` makes the raise disappear, dropping the `finally` makes `close_out` — and with it
    the buffer save and the guarded pool stop — vanish on the failure path.
    """
    cfg = _bounded(factory=smoke_run_config)
    pool = _Pool()
    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)

    with pytest.raises(_SentinelTrainError):
        mantis.run.compose_run(
            config=cfg, trainer=_ExplodingTrainer(), pool=pool, buffer=mk_graph_buffer(n_records=32),
            log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        )

    assert pool.started is True, "harness precondition: the pool was started by this run"
    assert pool.stopped is True, (
        "close_out did not run on the failure path — the epilogue must be in a `finally`, "
        "or a failed run loses its buffer save and leaves the pool running"
    )
